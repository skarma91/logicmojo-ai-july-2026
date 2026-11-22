"""Critic agent: exact checks in code first, a model's judgment second.

The judgment call is made only when every exact check passes, so no tokens are spent
on an opinion about a plan that is already broken.
"""

from __future__ import annotations
import llm
from guides import PACE_LIMIT
from schemas import Critique, Plan, Pool, TripRequest

JUDGE = (
    "You review a city itinerary that already meets the hard rules (days, pace, real places). "
    "Judge only: do the days make sense, is there variety, are the notes useful and free of "
    'invented facts? Reply with one JSON object only: {"approve": true|false, "feedback": "<one '
    'specific fix, or empty>"}'
)


def exact_checks(plan: Plan, req: TripRequest, pool: Pool) -> list[str]:
    """Hard-constraint failures. An empty list means the plan is valid."""
    problems = []
    places = pool.by_id()
    if len(plan.days) != req.days:
        problems.append(f"the plan has {len(plan.days)} days; the request is for {req.days}")
    limit = PACE_LIMIT[req.pace]
    for n, day in enumerate(plan.days, 1):
        if not day:
            problems.append(f"day {n} is empty")
        elif len(day) > limit:
            problems.append(f"day {n} has {len(day)} activities; the {req.pace} limit is {limit}")
    ids = [a.place_id for day in plan.days for a in day]
    unknown = sorted({i for i in ids if i not in places})
    if unknown:
        problems.append(f"places not in the pool: {unknown}")
    repeats = sorted({i for i in ids if ids.count(i) > 1})
    if repeats:
        problems.append(f"places scheduled twice: {repeats}")
    covered = {interest for i in ids if i in places for interest in places[i].interests}
    missing = [i for i in req.interests if i not in covered]
    if missing:
        problems.append(f"interests not covered: {missing}")
    return problems


class CriticAgent:
    name = "critic"

    def run(self, plan: Plan, req: TripRequest, pool: Pool) -> Critique:
        problems = exact_checks(plan, req, pool)
        if problems:
            return Critique(problems, approved=False, feedback="; ".join(problems))
        places = pool.by_id()
        text = "\n".join(f"Day {n}: " + "; ".join(f"{places[a.place_id].name} ({a.note})" for a in day)
                         for n, day in enumerate(plan.days, 1))
        raw = llm.chat_json([{"role": "system", "content": JUDGE},
                             {"role": "user", "content": f"{req.days} days in {req.destination}, "
                                                         f"pace {req.pace}.\n{text}"}], agent=self.name)
        approved = raw.get("approve") is True
        return Critique([], approved=approved, feedback="" if approved else str(raw.get("feedback") or "improve the plan"))
