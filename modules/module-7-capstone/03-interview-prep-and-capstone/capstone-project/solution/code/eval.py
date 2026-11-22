"""Evaluate the planner on a fixed set of requests (data/eval_requests.jsonl).

    python eval.py --retrieval      # retrieval alone: candidate chunks per interest, no model
    python eval.py                  # every request, real model calls
    python eval.py --limit 5

Metrics, in plain words:
  constraint pass rate  of requests that should get a plan, the share whose returned plan
                        meets every hard constraint (95% Wilson interval)
  grounding rate        of all places in all drafted plans, the share that exist in the pool;
                        the critic rejects the rest, so this measures the planner, not the output
  interest coverage     of the interests requested, the share covered by the returned plan
  revisions             extra plans drafted after the first, per planned request
  correct refusals      of requests that must be refused, the share refused
  cost                  model calls, tokens, and seconds per request
"""

from __future__ import annotations
import argparse
import json
import math
import pathlib
import time

import llm
from agents.critic import exact_checks
from agents.researcher import ResearcherAgent
from guides import INTERESTS
from schemas import TripRequest
from supervisor import Supervisor
from tracing import Tracer

REQUESTS = pathlib.Path(__file__).parent / "data" / "eval_requests.jsonl"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load() -> list[dict]:
    return [json.loads(l) for l in REQUESTS.read_text().splitlines() if l.strip()]


def retrieval_only():
    """How many distinct candidate chunks each interest finds, per city. No model."""
    from guides import DESTINATIONS
    r = ResearcherAgent()
    print(f"{'city':<14}" + "".join(f"{i[:9]:>10}" for i in INTERESTS))
    for city in DESTINATIONS:
        row = []
        for interest in INTERESTS:
            found = r.retrieve(TripRequest(city, 1, "moderate", [interest]))
            row.append(len(found))
        print(f"{city:<14}" + "".join(f"{n:>10}" for n in row))
    print("cells: chunks retrieved for that interest (at most 4); 0 means the guide has nothing on it")


def full(items: list[dict]):
    desk = Supervisor(Tracer("traces/eval.jsonl"))
    llm.reset_usage()
    rows, t0 = [], time.perf_counter()
    for it in items:
        start = time.perf_counter()
        res = desk.handle(it["request"], it["id"])
        rows.append((it, res, time.perf_counter() - start))
        print(f"  {it['id']}  expect={it['expect']:<8} got={res.status:<11} plans={res.rounds}  {res.reason[:60]}")

    planned = [(it, r) for it, r, _ in rows if it["expect"] == "plan"]
    ok = sum(r.plan is not None and not exact_checks(r.plan, r.request, r.pool) for _, r in planned)
    lo, hi = wilson(ok, len(planned))
    print(f"\nconstraint pass rate  {ok}/{len(planned)}  (95% interval {lo:.0%} to {hi:.0%})")

    ids = [(a.place_id, r.pool) for _, r, _ in rows for d in r.drafts for day in d.days for a in day]
    grounded = sum(pid in pool.by_id() for pid, pool in ids)
    print(f"grounding rate        {grounded}/{len(ids)} places in drafted plans exist in the pool")

    want = got = 0
    for _, r in planned:
        if r.plan is not None:
            places = r.pool.by_id()
            covered = {i for d in r.plan.days for a in d if a.place_id in places for i in places[a.place_id].interests}
            want += len(r.request.interests)
            got += len(set(r.request.interests) & covered)
    print(f"interest coverage     {got}/{want}")

    revisions = [max(0, r.rounds - 1) for _, r in planned if r.rounds]
    if revisions:
        print(f"revisions per plan    {sum(revisions) / len(revisions):.2f}")
    refuse = [(it, r) for it, r, _ in rows if it["expect"] == "refused"]
    print(f"correct refusals      {sum(r.status == 'refused' for _, r in refuse)}/{len(refuse)}")
    n = len(rows)
    tokens = sum(u["prompt_tokens"] + u["completion_tokens"] for u in llm.USAGE)
    print(f"cost per request      {len(llm.USAGE) / n:.1f} model calls, {tokens / n:.0f} tokens, "
          f"{(time.perf_counter() - t0) / n:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.retrieval:
        retrieval_only()
        return
    items = load()
    full(items[:args.limit] if args.limit else items)


if __name__ == "__main__":
    main()
