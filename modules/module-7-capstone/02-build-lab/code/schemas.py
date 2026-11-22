"""The data shapes every module agrees on.

Agents never pass loose dicts to each other: each one returns one of these, so the
supervisor (and a reader) always knows exactly what a step produced.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Ticket:
    id: str
    text: str                    # exactly what the customer wrote; never sent to a model


@dataclass
class Triage:
    intent: str                  # one of the 27 intents, or "unknown"
    urgency: str                 # low | normal | high
    sentiment: str               # calm | frustrated | angry
    confidence: float            # the model's own estimate, 0 to 1


@dataclass
class Article:
    intent: str
    title: str
    body: str
    score: float                 # how similar the ticket is to this article (cosine, 0 to 1)
    agrees: bool                 # was this also retrieval's best match on its own?


@dataclass
class Draft:
    text: str
    problems: list[str] = field(default_factory=list)   # filled by guards.check_reply


@dataclass
class Handoff:
    reason: str                  # why a human must take it
    summary: str                 # what the human needs to know, redacted


@dataclass
class CaseState:
    """Everything known about one ticket while the supervisor works on it."""
    ticket: Ticket
    redacted: str                # the ticket with personal data replaced by placeholders
    vault: dict                  # placeholder -> original value; stays in memory
    triage: Triage | None = None
    article: Article | None = None
    draft: Draft | None = None
    attempts: int = 0            # how many drafts the writer has produced
    trail: list[str] = field(default_factory=list)       # the steps taken, in order


@dataclass
class Outcome:
    ticket_id: str
    action: str                  # "reply" (sent to the customer) or "escalate" (to a human)
    reply: str = ""
    handoff: Handoff | None = None
    triage: Triage | None = None
    trail: list[str] = field(default_factory=list)
