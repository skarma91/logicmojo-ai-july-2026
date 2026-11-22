"""The desk's rules, in plain code: when a human must take over, and what a reply may say.

Rules live here, not in prompts, so they are cheap, testable, and never ignored.
"""

from __future__ import annotations
import re

import pii
from schemas import Article, CaseState

ALWAYS_HUMAN = {"complaint", "payment_issue", "contact_human_agent"}   # company policy
MIN_CONFIDENCE = 0.6        # below this, triage is a guess
TRUST_CONFIDENCE = 0.8      # at or above this, triage outweighs a retrieval disagreement
MIN_SCORE = 0.05            # the chosen article barely matches the ticket's words
MAX_ATTEMPTS = 2            # drafts before a human takes over
MAX_WORDS = 150

NUMBER = re.compile(r"\d{3,}")                    # order numbers, amounts, phone fragments
PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")         # {{Order Number}} and friends


def escalation_reason(state: CaseState) -> str | None:
    """Why this ticket must go to a human now, or None to keep going."""
    t, a, d = state.triage, state.article, state.draft
    if t is not None:
        if t.intent == "unknown":
            return "intent unclear"
        if t.intent in ALWAYS_HUMAN:
            return f"policy: {t.intent} always goes to a person"
        if t.sentiment == "angry":
            return "customer is angry"
        if t.urgency == "high":
            return "high urgency"
        if t.confidence < MIN_CONFIDENCE:
            return f"low triage confidence ({t.confidence:.2f})"
    if a is not None:
        if a.score < MIN_SCORE:
            return f"help article barely matches the ticket (score {a.score:.2f})"
        if not a.agrees and t.confidence < TRUST_CONFIDENCE:
            return "triage and knowledge disagree"
    if d is not None and d.problems and state.attempts >= MAX_ATTEMPTS:
        return "reply failed the checks twice: " + "; ".join(d.problems)
    return None


def check_reply(reply: str, article: Article, ticket_text: str) -> list[str]:
    """Problems that make a draft unsafe to send. An empty list means send it."""
    if not reply:
        return ["empty reply"]
    problems = []
    if len(reply.split()) > MAX_WORDS:
        problems.append(f"too long ({len(reply.split())} words; limit {MAX_WORDS})")
    source = article.body + " " + ticket_text
    invented = set(NUMBER.findall(reply)) - set(NUMBER.findall(source))
    if invented:
        problems.append(f"numbers that are not in the article or ticket: {sorted(invented)}")
    made_up = set(PLACEHOLDER.findall(reply)) - set(PLACEHOLDER.findall(source))
    if made_up:
        problems.append(f"placeholders that are not in the article: {sorted(made_up)}")
    if pii.detect(reply):                          # the writer never saw real personal data
        problems.append("reply contains personal data the writer could not have known")
    return problems
