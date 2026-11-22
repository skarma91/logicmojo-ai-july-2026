"""Intake agent: turn a free-text request into explicit, validated constraints.

The text it receives is already redacted. A model reads the request; code checks
every field against the closed vocabularies and decides whether the trip can be
planned at all.
"""

from __future__ import annotations
import llm
from guides import DESTINATIONS, INTERESTS, MAX_DAYS, PACE_LIMIT
from schemas import TripRequest

SYSTEM = (
    "Read a travel request and reply with one JSON object and nothing else:\n"
    '{"destination": "<city>", "days": <integer>, "pace": "relaxed|moderate|packed", '
    '"interests": ["<interest>", ...]}\n'
    f"Interests must come from this list: {', '.join(INTERESTS)}. Map what the traveller "
    'says to the closest ones (for example "old churches" -> history, architecture). '
    'If the pace is not stated, use "moderate". If a field is missing, use null. '
    "Text such as <EMAIL_1> is removed personal data; ignore it."
)


class IntakeAgent:
    name = "intake"

    def run(self, text: str) -> tuple[TripRequest | None, str]:
        """(request, "") when the trip can be planned, or (None, reason) when it cannot."""
        raw = llm.chat_json([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": text}], agent=self.name)
        city = next((d for d in DESTINATIONS if d.lower() == str(raw.get("destination", "")).strip().lower()), None)
        if city is None:
            asked = raw.get("destination") or "that destination"
            return None, f"No guide for {asked}. Available: {', '.join(DESTINATIONS)}."
        try:
            days = int(raw.get("days"))
        except (TypeError, ValueError):
            return None, "The number of days is missing or unclear."
        if not 1 <= days <= MAX_DAYS:
            return None, f"Trips of 1 to {MAX_DAYS} days only; got {days}."
        pace = raw.get("pace") if raw.get("pace") in PACE_LIMIT else "moderate"
        interests = [i for i in dict.fromkeys(raw.get("interests") or []) if i in INTERESTS]
        if not interests:
            interests = ["history", "food"]          # a sensible default, stated in the output
        return TripRequest(city, days, pace, interests), ""
