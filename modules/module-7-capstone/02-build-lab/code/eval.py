"""Evaluate the desk on the labeled tickets.

    python eval.py --retrieval        # knowledge agent alone: no model, runs in seconds
    python eval.py --limit 30         # the full desk on 30 tickets (real model calls)
    python eval.py                    # every ticket

What each number means:
  intent accuracy      triage picked the labeled intent, with a 95% Wilson interval
  escalation recall    of the tickets policy says need a person, the share escalated
                       (a miss means an upset or sensitive customer got a bot reply)
  escalation precision of the tickets escalated, the share that really needed a person
                       (low precision wastes human time)
  auto-resolved        share of tickets answered without a person; every one passed the checks
  cost                 model calls and tokens per ticket

Policy (the gold label): a ticket needs a person when its intent is in
guards.ALWAYS_HUMAN, or when Bitext tagged it as offensive language (flag "W").
"""

from __future__ import annotations
import argparse
import math
from collections import Counter

import guards
import llm
from agents.knowledge import KnowledgeAgent
from data import load_tickets
from schemas import Ticket
from supervisor import Supervisor
from tracing import Tracer


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate measured on a small sample."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def needs_person(t: dict) -> bool:
    return t["intent"] in guards.ALWAYS_HUMAN or "W" in t["flags"]


def retrieval_only(tickets: list[dict]):
    k = KnowledgeAgent()
    top1 = top3 = 0
    for t in tickets:
        order = k.scores(t["text"]).argsort()[::-1]
        gold = k.index[t["intent"]]
        top1 += order[0] == gold
        top3 += gold in order[:3]
    n = len(tickets)
    print(f"retrieval alone on {n} tickets: top-1 {top1}/{n} = {top1 / n:.0%}, top-3 {top3}/{n} = {top3 / n:.0%}")
    print("so retrieval is a second opinion, not the decider: triage names the intent")


def full_desk(tickets: list[dict]):
    desk = Supervisor(Tracer("traces/eval.jsonl"))
    llm.reset_usage()
    rows = [(t, desk.handle(Ticket(t["id"], t["text"]))) for t in tickets]
    n = len(rows)

    correct = sum(o.triage is not None and o.triage.intent == t["intent"] for t, o in rows)
    lo, hi = wilson(correct, n)
    print(f"intent accuracy      {correct}/{n} = {correct / n:.0%}  (95% interval {lo:.0%} to {hi:.0%})")

    tp = sum(needs_person(t) and o.action == "escalate" for t, o in rows)
    fn = sum(needs_person(t) and o.action == "reply" for t, o in rows)
    fp = sum(not needs_person(t) and o.action == "escalate" for t, o in rows)
    print(f"escalation recall    {tp}/{tp + fn}" + (f" = {tp / (tp + fn):.0%}" if tp + fn else ""))
    print(f"escalation precision {tp}/{tp + fp}" + (f" = {tp / (tp + fp):.0%}" if tp + fp else ""))

    replied = sum(o.action == "reply" for _, o in rows)
    print(f"auto-resolved        {replied}/{n} = {replied / n:.0%}")
    reasons = Counter(o.handoff.reason.split(":")[0].split(" (")[0] for _, o in rows if o.handoff)
    for reason, count in reasons.most_common():
        print(f"  escalated: {reason:<45} {count}")

    misses = Counter((t["intent"], o.triage.intent) for t, o in rows if o.triage and o.triage.intent != t["intent"])
    for (gold, got), count in misses.most_common(5):
        print(f"  confused: {gold} -> {got}  x{count}")

    calls = len(llm.USAGE)
    tokens = sum(u["prompt_tokens"] + u["completion_tokens"] for u in llm.USAGE)
    print(f"cost per ticket      {calls / n:.1f} model calls, {tokens / n:.0f} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true", help="score the knowledge agent alone (no model)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tickets")
    args = ap.parse_args()
    tickets = load_tickets()
    if args.limit:
        tickets = tickets[:args.limit]
    retrieval_only(tickets) if args.retrieval else full_desk(tickets)


if __name__ == "__main__":
    main()
