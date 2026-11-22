"""The supervisor: intake, research once, then a bounded plan-and-critique loop.

    redact -> intake -> (refuse?) -> research -> (pool too thin?) -> plan -> critique
                                                                       ^          |
                                                                       +-revise---+  at most MAX_REVISIONS
           -> an approved plan, the best plan that passed the exact checks, or a refusal
"""

from __future__ import annotations
import pii
from agents.critic import CriticAgent
from agents.intake import IntakeAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from schemas import Result, TripState
from tracing import Tracer

MAX_REVISIONS = 2           # so at most 1 + 2 = 3 plans per request
MAX_STEPS = 10              # a hard stop, whatever the rules say


class Supervisor:
    def __init__(self, tracer: Tracer | None = None):
        self.intake, self.researcher = IntakeAgent(), ResearcherAgent()
        self.planner, self.critic = PlannerAgent(), CriticAgent()
        self.tracer = tracer or Tracer(None)

    def next_step(self, s: TripState) -> str:
        if s.request is None:
            return "intake"
        if s.pool is None:
            return "research"
        if len(s.pool.places) < s.request.days:
            return "refuse"
        if s.critique is None:
            return "plan" if s.plan is None else "critique"
        if s.critique.approved:
            return "finish"
        if s.rounds > MAX_REVISIONS:
            return "finish"
        return "plan"

    def handle(self, request_text: str, request_id: str = "r") -> Result:
        text, _, _ = pii.redact(request_text)              # before any model sees it
        s = TripState(text=text)
        with self.tracer.span("request", request_id=request_id) as root:
            result = None
            for _ in range(MAX_STEPS):
                step = self.next_step(s)
                s.trail.append(step)
                with self.tracer.span(step, request_id=request_id):
                    if step == "intake":
                        s.request, reason = self.intake.run(s.text)
                        if s.request is None:
                            result = Result("refused", reason, trail=s.trail)
                    elif step == "research":
                        s.pool = self.researcher.run(s.request)
                        found = {i for p in s.pool.places for i in p.interests}
                        dropped = [i for i in s.request.interests if i not in found]
                        if dropped:                         # plan what the guide supports, and say so
                            s.notes.append(f"The {s.request.destination} guide has nothing usable on: {', '.join(dropped)}.")
                            s.request.interests = [i for i in s.request.interests if i in found]
                        if not s.request.interests:
                            result = Result("refused", s.notes[-1], s.request, pool=s.pool, trail=s.trail)
                    elif step == "refuse":
                        result = Result("refused", f"The {s.request.destination} guide gives only "
                                        f"{len(s.pool.places)} usable places; not enough for "
                                        f"{s.request.days} days.", s.request, pool=s.pool, trail=s.trail)
                    elif step == "plan":
                        feedback = None
                        if s.critique and not s.critique.approved:
                            feedback = s.critique.problems or [s.critique.feedback]
                        s.plan, s.critique = self.planner.run(s.request, s.pool, feedback), None
                        s.drafts.append(s.plan)
                        s.rounds += 1
                    elif step == "critique":
                        s.critique = self.critic.run(s.plan, s.request, s.pool)
                        if not s.critique.problems:
                            s.best = s.plan                 # valid, even if the judge wants more
                    elif step == "finish":
                        result = self.finish(s)
                if result is not None:
                    break
            if result is None:
                result = self.finish(s)
            result.drafts, result.notes = s.drafts, s.notes
            root["status"] = result.status
        return result

    def finish(self, s: TripState) -> Result:
        if s.critique is not None and s.critique.approved:
            return Result("plan", "", s.request, s.plan, s.pool, s.rounds, s.trail)
        if s.best is not None:
            return Result("best_effort", "Meets every hard constraint; the reviewer still "
                          f"suggested: {s.critique.feedback if s.critique else 'more polish'}",
                          s.request, s.best, s.pool, s.rounds, s.trail)
        reason = "; ".join(s.critique.problems) if s.critique else "no valid plan"
        return Result("refused", f"Could not produce a plan that meets the constraints: {reason}",
                      s.request, pool=s.pool, rounds=s.rounds, trail=s.trail)
