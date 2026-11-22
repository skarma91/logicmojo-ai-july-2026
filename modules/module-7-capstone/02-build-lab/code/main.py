"""Run the support desk on sample tickets, or on a message you type.

    python main.py                       # 6 tickets spread across the sample
    python main.py --n 12
    python main.py --ticket "I'm Ana Ruiz, ana@example.com. How do I cancel order 4812?"
"""

from __future__ import annotations
import argparse
import logging

import llm
from data import load_tickets
from schemas import Outcome, Ticket
from supervisor import Supervisor
from tracing import Tracer


def show(text: str, o: Outcome):
    print("=" * 78)
    print(f"[{o.ticket_id}] {text}")
    if o.triage:
        t = o.triage
        print(f"  triage   : {t.intent} | urgency {t.urgency} | {t.sentiment} | confidence {t.confidence:.2f}")
    print(f"  steps    : {' > '.join(o.trail)}")
    if o.action == "reply":
        print(f"  REPLY    : {o.reply}")
    else:
        print(f"  ESCALATE : {o.handoff.reason}")
        print(f"  handoff  : {o.handoff.summary}")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    ap = argparse.ArgumentParser(description="Customer Support Desk")
    ap.add_argument("--ticket", help="your own customer message")
    ap.add_argument("--n", type=int, default=6, help="how many sample tickets to run")
    args = ap.parse_args()

    if args.ticket:
        tickets = [Ticket("you", args.ticket)]
    else:
        sample = load_tickets()
        step = max(1, len(sample) // args.n)
        tickets = [Ticket(t["id"], t["text"]) for t in sample[::step][:args.n]]

    desk = Supervisor(Tracer("traces/desk.jsonl"))
    for ticket in tickets:
        show(ticket.text, desk.handle(ticket))

    print("=" * 78)
    print("model calls per agent")
    for agent, u in llm.usage_by_agent().items():
        print(f"  {agent:<10} calls={u['calls']:<3} tokens in={u['prompt']:<6} out={u['completion']}")
    print("trace: traces/desk.jsonl")


if __name__ == "__main__":
    main()
