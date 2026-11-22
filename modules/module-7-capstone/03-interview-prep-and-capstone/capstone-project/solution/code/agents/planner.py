"""Planner agent: arrange places from the pool into days, and only from the pool.

The planner refers to places by pool id (P1, P2, ...), never by free-text name, so an
invented place shows up as an id that does not exist: easy for code to catch.
"""

from __future__ import annotations
import llm
from guides import PACE_LIMIT
from schemas import Activity, Plan, Pool, TripRequest

SYSTEM = (
    "You plan city trips. Use ONLY places from the numbered pool, by their id. Reply with one "
    "JSON object only:\n"
    '{"days": [{"day": 1, "activities": [{"place": "P3", "note": "<one short sentence>"}]}]}\n'
    "Rules: exactly the requested number of days; at most the stated number of activities per "
    "day; cover every interest; never repeat a place; mix sections across a day. Notes must not "
    "invent opening hours, prices, or travel times."
)


class PlannerAgent:
    name = "planner"

    def run(self, req: TripRequest, pool: Pool, feedback: list[str] | None = None) -> Plan:
        lines = "\n".join(f"{p.id} | {p.name} | {p.section} | interests: {', '.join(p.interests)}"
                          for p in pool.places)
        prompt = (f"Trip: {req.days} days in {req.destination}, pace {req.pace} "
                  f"(at most {PACE_LIMIT[req.pace]} activities per day), "
                  f"interests: {', '.join(req.interests)}.\n\nPool:\n{lines}")
        if feedback:
            prompt += "\n\nYour previous plan was rejected. Fix these problems:\n- " + "\n- ".join(feedback)
        raw = llm.chat_json([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}], agent=self.name)
        days = []
        for d in raw.get("days") or []:                 # validate the shape; drop what is malformed
            acts = d.get("activities") if isinstance(d, dict) else None
            days.append([Activity(str(a.get("place", "")).strip(), str(a.get("note", "")).strip()[:200])
                         for a in (acts or []) if isinstance(a, dict)])
        return Plan(days)
