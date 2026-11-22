"""The supervisor: a plain Python loop that decides which agent works next.

    redact -> triage -> (escalate?) -> knowledge -> (escalate?) -> write -> check
                                                                   ^        |
                                                                   +-retry--+   at most MAX_ATTEMPTS
           -> send the reply, or hand the ticket to a human

Routing is rules, not a model: every decision is cheap, testable, and explainable.
The agents do the language work; the supervisor only reads the state and picks a step.
"""

from __future__ import annotations
import guards
import pii
from agents.knowledge import KnowledgeAgent
from agents.triage import TriageAgent
from agents.writer import ReplyWriterAgent
from data import CATEGORY
from schemas import CaseState, Handoff, Outcome, Ticket
from tracing import Tracer

MAX_STEPS = 8               # a hard stop, whatever the rules say


class Supervisor:
    def __init__(self, tracer: Tracer | None = None):
        self.triage = TriageAgent()
        self.knowledge = KnowledgeAgent()
        self.writer = ReplyWriterAgent()
        self.tracer = tracer or Tracer(None)

    def next_step(self, s: CaseState) -> str:
        """Read the state, return the next step's name."""
        if s.triage is None:
            return "triage"
        if guards.escalation_reason(s):
            return "escalate"
        if s.article is None:
            return "knowledge"
        if s.draft is None or s.draft.problems:
            return "write"
        return "send"

    def handle(self, ticket: Ticket) -> Outcome:
        redacted, vault, _ = pii.redact(ticket.text)        # before any model sees the text
        s = CaseState(ticket=ticket, redacted=redacted, vault=vault)
        with self.tracer.span("ticket", ticket_id=ticket.id) as root:
            for _ in range(MAX_STEPS):
                step = self.next_step(s)
                s.trail.append(step)
                with self.tracer.span(step, ticket_id=ticket.id):
                    if step == "triage":
                        s.triage = self.triage.run(s.redacted)
                    elif step == "knowledge":
                        s.article = self.knowledge.run(s.redacted, s.triage.intent)
                    elif step == "write":
                        feedback = s.draft.problems if s.draft else None
                        s.draft = self.writer.run(s.redacted, s.article, feedback)
                        s.draft.problems = guards.check_reply(s.draft.text, s.article, s.redacted)
                        s.attempts += 1
                    elif step == "escalate":
                        outcome = self.escalate(s, guards.escalation_reason(s))
                        break
                    elif step == "send":
                        outcome = Outcome(ticket.id, "reply", reply=pii.restore(s.draft.text, s.vault),
                                          triage=s.triage, trail=s.trail)
                        break
            else:
                outcome = self.escalate(s, f"step limit of {MAX_STEPS} reached")
            root["action"] = outcome.action
        return outcome

    def escalate(self, s: CaseState, reason: str) -> Outcome:
        """Hand the ticket to a person with a summary they can act on at once."""
        t = s.triage
        about = f"{t.intent} ({CATEGORY.get(t.intent, '?')}), urgency {t.urgency}, " \
                f"sentiment {t.sentiment}" if t else "not triaged"
        summary = f"{about}. Customer wrote: {s.redacted[:300]}"
        return Outcome(s.ticket.id, "escalate", handoff=Handoff(reason, summary), triage=t, trail=s.trail)
