"""The data shapes the Travel Planner's modules agree on."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TripRequest:
    destination: str             # one of the cities in data/guides.jsonl
    days: int                    # 1 to MAX_DAYS
    pace: str                    # relaxed | moderate | packed
    interests: list[str]         # from guides.INTERESTS


@dataclass
class Place:
    id: str                      # "P1", "P2", ... (what the planner refers to)
    name: str                    # copied word for word from the guide text
    section: str                 # See | Do | Eat | Drink
    interests: list[str]         # the interest queries that retrieved its chunk
    chunk_id: str                # the guide chunk it came from
    url: str                     # the Wikivoyage page, for the citation


@dataclass
class Pool:
    destination: str
    places: list[Place]

    def by_id(self) -> dict[str, Place]:
        return {p.id: p for p in self.places}


@dataclass
class Activity:
    place_id: str
    note: str = ""


@dataclass
class Plan:
    days: list[list[Activity]]   # days[0] is day 1


@dataclass
class Critique:
    problems: list[str]          # exact-check failures, from code
    approved: bool               # True only if no problems and the judge approves
    feedback: str = ""           # what the planner should change next


@dataclass
class TripState:
    """Everything known about one request while the supervisor works on it."""
    text: str                    # redacted request text
    request: TripRequest | None = None
    pool: Pool | None = None
    plan: Plan | None = None
    critique: Critique | None = None
    best: Plan | None = None     # the last plan that passed every exact check
    drafts: list[Plan] = field(default_factory=list)   # every plan drafted, for the eval
    notes: list[str] = field(default_factory=list)     # what the traveller should know
    rounds: int = 0              # plans produced so far
    trail: list[str] = field(default_factory=list)


@dataclass
class Result:
    status: str                  # "plan" | "best_effort" | "refused"
    reason: str = ""
    request: TripRequest | None = None
    plan: Plan | None = None
    pool: Pool | None = None
    rounds: int = 0
    trail: list[str] = field(default_factory=list)
    drafts: list[Plan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
