"""Triage agent: what does the customer want, how urgent is it, how upset are they."""

from __future__ import annotations
import llm
from data import INTENTS
from schemas import Triage

URGENCY = ("low", "normal", "high")
SENTIMENT = ("calm", "frustrated", "angry")

SYSTEM = (
    "You triage customer-support tickets. Reply with one JSON object and nothing else:\n"
    '{"intent": "<one allowed intent>", "urgency": "low|normal|high", '
    '"sentiment": "calm|frustrated|angry", "confidence": <number from 0 to 1>}\n'
    'If no intent fits, use "unknown" with a low confidence. Text such as <EMAIL_1> is '
    "personal data that was removed; ignore it.\n"
    "Allowed intents: " + ", ".join(INTENTS)
)


class TriageAgent:
    name = "triage"

    def run(self, text: str) -> Triage:
        raw = llm.chat_json([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": text}], agent=self.name)
        # Never trust model JSON: check every field and fall back to a safe value.
        intent = raw.get("intent") if raw.get("intent") in INTENTS else "unknown"
        urgency = raw.get("urgency") if raw.get("urgency") in URGENCY else "normal"
        sentiment = raw.get("sentiment") if raw.get("sentiment") in SENTIMENT else "calm"
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0                       # unreadable confidence counts as none
        return Triage(intent, urgency, sentiment, confidence)
