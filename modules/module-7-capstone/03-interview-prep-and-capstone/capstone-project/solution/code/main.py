"""Plan a trip from the command line.

    python main.py "Four relaxed days in Lisbon, I love food and old churches"
    python main.py                       # a sample request
"""

from __future__ import annotations
import logging
import sys

import llm
from schemas import Result
from supervisor import Supervisor
from tracing import Tracer

SAMPLE = "Hi, I'm Priya Shah (priya@mail.com). Three relaxed days in Lisbon, I love food and old churches."


def show(r: Result):
    print("=" * 78)
    if r.request:
        q = r.request
        print(f"{q.days} days in {q.destination} | pace {q.pace} | interests: {', '.join(q.interests)}")
    print(f"status: {r.status} | plans drafted: {r.rounds} | steps: {' > '.join(r.trail)}")
    if r.reason:
        print(f"note: {r.reason}")
    for note in r.notes:
        print(f"note: {note}")
    if r.plan:
        places = r.pool.by_id()
        for n, day in enumerate(r.plan.days, 1):
            print(f"\nDay {n}")
            for a in day:
                p = places[a.place_id]
                print(f"  - {p.name} [{p.section}] {a.note}")
        print("\nSources: " + ", ".join(sorted({places[a.place_id].url for d in r.plan.days for a in d})))
        print("Text from Wikivoyage, CC BY-SA 4.0.")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    text = " ".join(sys.argv[1:]) or SAMPLE
    show(Supervisor(Tracer("traces/planner.jsonl")).handle(text))
    print("=" * 78)
    for agent, u in llm.usage_by_agent().items():
        print(f"  {agent:<10} calls={u['calls']:<3} tokens in={u['prompt']:<6} out={u['completion']}")
    print("trace: traces/planner.jsonl")


if __name__ == "__main__":
    main()
