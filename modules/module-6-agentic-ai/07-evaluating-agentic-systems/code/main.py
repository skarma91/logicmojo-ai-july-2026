"""Class 6.7 build: evaluate the team's path, not just its answers, and gate regressions.

Live mode runs every task in data/tasks.jsonl through the class 6.1 (Multi-agent
architectures) team with tracing on (class 6.6, Observability and tracing), then reads
each run's trajectory back from the trace and scores:

  - the PATH:    in-order match against the reference (required steps, in order,
                 extras allowed), exact and set match for comparison, step precision
                 and recall, and forbidden steps
  - TOOL USE:    schema validity of every call, expected argument values, redundant calls
  - the OUTCOME: a concrete success check per task (cites the right source, or
                 mentions the right fact)

A task PASSES only if the outcome succeeds and the path is right. "Right answer,
wrong path" and "wrong answer, right path" are flagged separately. The pass rate is
reported with Wald and Wilson 95% confidence intervals.

    python main.py                           # live: run the tasks (paced for the free tier)
    python main.py --save-baseline           # live, then store results/baseline.json
    python main.py --gate                    # live, then fail (exit 1) on any regression
    python main.py --from-trace traces/sample_eval_trace.jsonl        # offline, no model calls
    python main.py --from-trace traces/sample_regressed_trace.jsonl \\
                   --baseline traces/sample_eval_trace.jsonl          # offline regression gate
"""

from __future__ import annotations
import argparse
import functools
import json
import logging
import pathlib
import sys
import time
from collections import defaultdict

import evals
import llm

HERE = pathlib.Path(__file__).parent
TASKS = [json.loads(l) for l in (HERE / "data" / "tasks.jsonl").read_text().splitlines() if l.strip()]
BASELINE = HERE / "results" / "baseline.json"


# ---------------------------------------------------------------------------
# Live runs: instrument the team so every step lands in the trace
# ---------------------------------------------------------------------------
def run_live(pace: float) -> pathlib.Path:
    from langgraph.types import Command
    from mcp_bridge import McpBridge
    from tax_worker import build_tax_worker, search_tax_docs
    from team import build_team, run_team
    from tracing import Tracer

    path = HERE / "traces" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    tracer = Tracer(path)

    def search(query: str) -> str:
        with tracer.span("search_tax_docs", "retriever", inputs=query, metadata={"args": {"query": query}}) as s:
            out = search_tax_docs(query)
            tracer.set_output(s, out)
            return out

    def wrap(name, fn):
        @functools.wraps(fn)
        def node(state):
            with tracer.span(name, "chain", inputs=state.get("task", "")) as s:
                out = fn(state)
                if isinstance(out, Command):            # record where control went next
                    s.metadata["goto"] = str(out.goto)
                tracer.set_output(s, s.metadata.get("goto", "ok"))
                return out
        return node

    with McpBridge() as bridge:
        original = bridge.call

        def call(name, args):
            with tracer.span(f"mcp:{name}", "tool", inputs=args, metadata={"args": args}) as s:
                out = original(name, args)
                tracer.set_output(s, out)
                return out
        bridge.call = call

        team = build_team(build_tax_worker(bridge, search_fn=search), wrap=wrap)
        for i, task in enumerate(TASKS):
            if i:
                time.sleep(pace)                          # stay inside the free tier's per-minute limits
            with tracer.span("run", "chain", inputs=task["question"],
                             metadata={"task_id": task["id"], "question": task["question"]}) as root:
                state = run_team(team, task["question"])
                tracer.set_output(root, state.get("explanation") or state.get("findings") or "")
            print(f"  ran {task['id']}")
    return path


# ---------------------------------------------------------------------------
# Scoring a trace
# ---------------------------------------------------------------------------
def score_trace(path: pathlib.Path) -> dict:
    spans = [json.loads(l) for l in open(path) if l.strip()]
    by_trace = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
    tasks = {t["id"]: t for t in TASKS}
    results = {}
    for tspans in by_trace.values():
        root = next(s for s in tspans if s["parent_id"] is None)
        task = tasks.get(root["metadata"].get("task_id"))
        if task is None:
            continue
        steps, calls = evals.trajectory(tspans)
        p, r = evals.step_precision_recall(steps, task["reference"])
        bad = evals.forbidden_steps(steps, task["forbidden"])
        in_order = evals.match_in_order(steps, task["reference"])
        results[task["id"]] = {
            "steps": steps,
            "exact": evals.match_exact(steps, task["reference"]),
            "in_order": in_order,
            "set": evals.match_set(steps, task["reference"]),
            "precision": round(p, 3), "recall": round(r, 3),
            "forbidden": bad,
            "path_ok": in_order and not bad,
            "tools": evals.tool_use_report(calls, task["expected_args"]),
            "success": evals.task_success(root["outputs"], task["success"]),
            "trace_id": root["trace_id"], "langsmith_run_id": root.get("langsmith_run_id", ""),
            "recorded_with": root["metadata"].get("recorded_with", ""),
        }
    return results


def report(results: dict):
    print(f"\n{'task':<5}{'pass':<6}{'success':<9}{'path':<6}{'P':>5}{'R':>6}  {'args':<6}trajectory")
    for tid in sorted(results):
        r = results[tid]
        t = r["tools"]
        args = f"{t['args_ok']}/{t['args_checked']}" if t["args_checked"] else "-"
        flag = ""
        if r["success"] and not r["path_ok"]:
            flag = "   <- RIGHT ANSWER, WRONG PATH"
        elif r["path_ok"] and not r["success"]:
            flag = "   <- wrong answer, right path"
        print(f"{tid:<5}{'yes' if evals.passed(r) else 'no':<6}{str(r['success']):<9}{'ok' if r['path_ok'] else 'bad':<6}"
              f"{r['precision']:>5.2f}{r['recall']:>6.2f}  {args:<6}{' > '.join(r['steps'])}{flag}")
        if r["forbidden"]:
            print(f"{'':<11}forbidden step(s): {r['forbidden']}")

    n = len(results)
    k = sum(evals.passed(r) for r in results.values())
    ks = sum(r["success"] for r in results.values())
    wl, wh = evals.wald_ci(k, n)
    sl, sh = evals.wilson_ci(k, n)
    print(f"\nPASS RATE     {k}/{n} = {k / n:.0%}   Wald 95% [{wl:.0%}, {wh:.0%}]   Wilson 95% [{sl:.0%}, {sh:.0%}]")
    print(f"answer only   {ks}/{n} = {ks / n:.0%}   (what an answer-only eval would have reported)")
    calls = sum(r["tools"]["calls"] for r in results.values())
    print(f"TOOL USE      calls={calls}  schema ok={sum(r['tools']['schema_ok'] for r in results.values())}"
          f"  expected args ok={sum(r['tools']['args_ok'] for r in results.values())}"
          f"/{sum(r['tools']['args_checked'] for r in results.values())}"
          f"  redundant={sum(r['tools']['redundant'] for r in results.values())}")
    print(f"STEPS         mean precision {sum(r['precision'] for r in results.values()) / n:.2f}"
          f"   mean recall {sum(r['recall'] for r in results.values()) / n:.2f}")
    rec = next((r["recorded_with"] for r in results.values() if r["recorded_with"]), "")
    if rec:
        print(f"recorded with: {rec}")


def send_feedback(results: dict):
    """With LangSmith on, attach every score to its run (class 6.6, Observability and tracing)."""
    from tracing import langsmith_enabled
    if not langsmith_enabled():
        return
    from langsmith import Client
    client = Client()
    for tid, r in results.items():
        if r["langsmith_run_id"]:
            for key, score in (("path_ok", int(r["path_ok"])), ("task_success", int(r["success"])),
                               ("step_precision", r["precision"]), ("step_recall", r["recall"])):
                client.create_feedback(r["langsmith_run_id"], key, score=score, comment=tid)


def load_results(path: pathlib.Path) -> dict:
    return score_trace(path) if path.suffix == ".jsonl" else json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-trace", type=pathlib.Path, help="score a recorded trace; no model calls")
    ap.add_argument("--baseline", type=pathlib.Path, help="results .json or trace .jsonl to gate against")
    ap.add_argument("--gate", action="store_true", help=f"gate against {BASELINE.relative_to(HERE)}")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--pace", type=float, default=2.0, help="seconds between tasks in live mode")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

    if args.from_trace:
        trace = args.from_trace
    else:
        llm.reset_usage()
        print(f"running {len(TASKS)} tasks live (pace {args.pace}s)")
        trace = run_live(args.pace)
    results = score_trace(trace)
    report(results)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "latest.json").write_text(json.dumps(results, indent=2))
    if not args.from_trace:
        send_feedback(results)
    if args.save_baseline:
        BASELINE.write_text(json.dumps(results, indent=2))
        print(f"saved baseline: {BASELINE.relative_to(HERE)}")

    base_path = args.baseline or (BASELINE if args.gate else None)
    if base_path:
        reg = evals.regressions(load_results(base_path), results)
        print(f"\nREGRESSION GATE against {base_path.name}: "
              + ("PASS (nothing that passed before fails now)" if not reg else f"FAIL: {reg}"))
        if reg:
            sys.exit(1)


if __name__ == "__main__":
    main()
