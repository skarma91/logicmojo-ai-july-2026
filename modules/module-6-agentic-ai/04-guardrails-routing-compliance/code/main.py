"""Class 6.4 build: a guardrail gateway in front of the team, measured on labeled requests.

Runs every request in data/requests.jsonl through gateway.py and reports:

  - each request's outcome (answer, blocked, refused, canned, fallback), the stage
    that decided it, its route and similarity score, and what PII was redacted,
  - decision accuracy against the labels, block rate (unsafe requests stopped) and
    false-block rate (safe requests wrongly stopped),
  - routing accuracy,
  - PII precision and recall over labeled spans,
  - a compliance check: the audit log must contain no raw PII value,
  - model calls and tokens per agent (classifier versus team).

Run (native tool calling: groq or ollama):
    pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq
    python main.py              # full run: the team answers the allowed requests
    python main.py --no-team    # gateway only: stops at the team's door (far fewer calls)
    python main.py --tune       # print router scores and a suggested threshold, no model calls
    python main.py --tau 0.4    # run with a different router threshold
"""

from __future__ import annotations
import argparse
import json
import logging
import pathlib

import llm
import pii
from gateway import Gateway, AUDIT
from router import SemanticRouter, FALLBACK

HERE = pathlib.Path(__file__).parent
BLOCKING = {"blocked", "refused"}


def load_requests() -> list[dict]:
    rows = []
    for line in (HERE / "data" / "requests.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["text"] = r["text"] * r.get("repeat", 1)
            rows.append(r)
    return rows


def tune(router: SemanticRouter, rows: list[dict]):
    """Suggest TAU from data: midway between the weakest in-scope best score and the
    strongest out-of-scope best score. If they overlap, no threshold separates them,
    and the route examples need work."""
    print(f"\nROUTER SCORES (current tau = {router.tau})")
    inside, outside = [], []
    for r in rows:
        if r["route"] is None:
            continue                                   # blocked by rules before routing
        red = pii.redact(r["text"])[0]
        _, best, sc = router.route(red)
        top = max(sc, key=sc.get)
        (outside if r["route"] == FALLBACK else inside).append(best)
        print(f"  {r['id']}  best={best:.3f} ({top:<16}) label={r['route']}")
    lo, hi = min(inside), max(outside) if outside else 0.0
    if lo > hi:
        print(f"\n  in-scope min {lo:.3f} > out-of-scope max {hi:.3f}: suggested tau = {(lo + hi) / 2:.3f}")
    else:
        print(f"\n  overlap: in-scope min {lo:.3f} <= out-of-scope max {hi:.3f}; add route examples")


def report(rows: list[dict], decisions: list):
    print(f"\n{'id':<4} {'outcome':<9} {'stage':<11} {'route (score)':<26} {'pii':<16} label")
    for r, d in zip(rows, decisions):
        mark = "" if d.outcome == r["expect"] else "   <- mismatch"
        route = f"{d.route} ({d.score:.2f})" if d.route else "-"
        kinds = ",".join(k for k, _ in d.pii) or "-"
        print(f"{r['id']:<4} {d.outcome:<9} {d.stage:<11} {route:<26} {kinds:<16} {r['expect']}{mark}")

    n = len(rows)
    correct = sum(d.outcome == r["expect"] for r, d in zip(rows, decisions))
    unsafe = [d for r, d in zip(rows, decisions) if r["expect"] in BLOCKING]
    safe = [d for r, d in zip(rows, decisions) if r["expect"] not in BLOCKING]
    block_rate = sum(d.outcome in BLOCKING for d in unsafe) / len(unsafe)
    false_block = sum(d.outcome in BLOCKING for d in safe) / len(safe)
    routed = [(r, d) for r, d in zip(rows, decisions) if r["route"] is not None and d.route]
    route_acc = sum(d.route == r["route"] for r, d in routed) / len(routed) if routed else 0.0
    m = pii.span_metrics([d.pii for d in decisions], [[tuple(p) for p in r["pii"]] for r in rows])

    print(f"\nDECISIONS   {correct}/{n} match the label")
    print(f"  block rate        {block_rate:.0%}  (unsafe requests stopped: blocked or refused)")
    print(f"  false-block rate  {false_block:.0%}  (safe requests wrongly stopped)")
    print(f"  routing accuracy  {route_acc:.0%}  over {len(routed)} routed requests")
    print(f"PII         precision {m['precision']:.2f}  recall {m['recall']:.2f}  "
          f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")

    raw_values = [p[1] for r in rows for p in r["pii"]]
    audit = AUDIT.read_text() if AUDIT.exists() else ""
    leaked = [v for v in raw_values if v in audit]
    print(f"COMPLIANCE  audit log lines={len(audit.splitlines())}  raw PII values found in it: "
          f"{len(leaked)} {'(PASS)' if not leaked else '(LEAK: ' + ', '.join(leaked) + ')'}")
    if leaked:
        print("            every leak is a span the detector missed (see PII recall). Rules plus")
        print("            validators cannot find a name with no context cue; production adds a")
        print("            named-entity model, for example Microsoft Presidio with spaCy.")

    print("\nMODEL CALLS PER AGENT")
    for agent, u in llm.usage_by_agent().items():
        print(f"  {agent:<17} calls={u['calls']:<3} in={u['prompt']:<6} cached={u['cached']:<6} out={u['completion']}")

    print("\nSAMPLE REPLIES")
    for r, d in zip(rows, decisions):
        if d.outcome in ("answer", "refused", "canned") and r["id"] in ("r03", "r09", "r11"):
            print(f"  {r['id']} [{d.outcome}] sent onward: {d.redacted[:90]}")
            print(f"      reply: {d.answer[:140]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-team", action="store_true", help="stop at the team's door")
    ap.add_argument("--tune", action="store_true", help="print router scores, suggest tau")
    ap.add_argument("--tau", type=float, default=None, help="override the router threshold")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    rows = load_requests()
    router = SemanticRouter() if args.tau is None else SemanticRouter(tau=args.tau)
    if args.tune:
        tune(router, rows)
        return

    AUDIT.unlink(missing_ok=True)                      # fresh audit log per run
    llm.reset_usage()
    if args.no_team:
        gw = Gateway(router=router)
        decisions = [gw.handle(r["text"]) for r in rows]
    else:
        from mcp_bridge import McpBridge
        from tax_worker import build_tax_worker
        from team import build_team, run_team
        with McpBridge() as bridge:
            gw = Gateway(team=build_team(build_tax_worker(bridge)), run_team=run_team, router=router)
            decisions = [gw.handle(r["text"]) for r in rows]
    report(rows, decisions)


if __name__ == "__main__":
    main()
