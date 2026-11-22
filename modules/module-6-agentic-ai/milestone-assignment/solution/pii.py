"""PII detection and redaction with transparent rules plus validators.

A regex alone over-matches: any 16 digits look like a card, any 3-2-4 digits look
like a US Social Security number. So every pattern here is paired with a VALIDATOR
that rejects strings which cannot be real:

  EMAIL  a standard address shape
  PHONE  North American numbers; the area code cannot start with 0 or 1
  SSN    3-2-4 digits; area 000, 666 and 900-999, group 00, and serial 0000 are never issued
  CARD   13 to 19 digits (spaces or dashes allowed) that pass the Luhn checksum
  PERSON a capitalized name right after a context cue ("my neighbor", "patient",
         "my name is", "Dr."); a name with no cue is MISSED, on purpose: names need
         a named-entity model, which is exactly what Presidio adds (with spaCy)

Detected spans are replaced with TYPED PLACEHOLDERS (<EMAIL_1>, <SSN_1>, ...), so an
agent can still reason about the request ("the user gave an email address") without
ever seeing the value. The mapping from placeholder to value stays in memory in the
gateway; it is never logged, traced, or sent to a model.

Production systems use a library such as Microsoft Presidio (recognizers plus
context words plus optional NER models). The idea is the same: detect, validate,
replace.
"""

from __future__ import annotations
import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
SSN_RE = re.compile(r"\b(\d{3})-(\d{2})-(\d{4})\b")
PERSON_RE = re.compile(r"(?i:\b(?:neighbou?r|patient|coworker|colleague|client|my name is|i am|i'm|"
                       r"mr\.|ms\.|mrs\.|dr\.))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?([2-9]\d{2})\)?[ .-]?(\d{3})[ .-](\d{4})(?!\d)")


def luhn_ok(digits: str) -> bool:
    """Luhn checksum: from the right, double every second digit (subtract 9 if the
    result exceeds 9), sum everything; a valid number has a sum divisible by 10."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def ssn_ok(area: str, group: str, serial: str) -> bool:
    a = int(area)
    return a != 0 and a != 666 and a < 900 and group != "00" and serial != "0000"


@dataclass
class Span:
    kind: str
    start: int
    end: int
    text: str


def detect(text: str) -> list[Span]:
    """All validated PII spans, non-overlapping, in order of position."""
    found: list[Span] = []

    def free(s, e):
        return all(e <= f.start or s >= f.end for f in found)

    for m in EMAIL_RE.finditer(text):
        found.append(Span("EMAIL", m.start(), m.end(), m.group()))
    for m in CARD_RE.finditer(text):                       # before phone and SSN: longest first
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and luhn_ok(digits) and free(m.start(), m.end()):
            found.append(Span("CARD", m.start(), m.end(), m.group()))
    for m in SSN_RE.finditer(text):
        if ssn_ok(*m.groups()) and free(m.start(), m.end()):
            found.append(Span("SSN", m.start(), m.end(), m.group()))
    for m in PHONE_RE.finditer(text):
        if free(m.start(), m.end()):
            found.append(Span("PHONE", m.start(), m.end(), m.group()))
    for m in PERSON_RE.finditer(text):                     # the name only, not the cue word
        if free(m.start(1), m.end(1)):
            found.append(Span("PERSON", m.start(1), m.end(1), m.group(1)))
    return sorted(found, key=lambda s: s.start)


def redact(text: str) -> tuple[str, dict, list[Span]]:
    """Replace every span with a typed placeholder.

    Returns (redacted_text, vault, spans). The vault maps placeholder -> original
    value and must never leave the process. The same value gets the same
    placeholder, so "<EMAIL_1> ... <EMAIL_1>" still reads as one address.
    """
    spans = detect(text)
    vault, seen, counters, out, pos = {}, {}, {}, [], 0
    for s in spans:
        if s.text not in seen:
            counters[s.kind] = counters.get(s.kind, 0) + 1
            seen[s.text] = f"<{s.kind}_{counters[s.kind]}>"
            vault[seen[s.text]] = s.text
        out.append(text[pos:s.start])
        out.append(seen[s.text])
        pos = s.end
    out.append(text[pos:])
    return "".join(out), vault, spans


def span_metrics(predicted: list[list[tuple]], expected: list[list[tuple]]) -> dict:
    """Precision and recall over (kind, text) spans, summed across requests.

    precision = flagged spans that really are PII / all flagged spans
    recall    = real PII spans that were flagged  / all real PII spans
    For redaction, recall matters more: a missed SSN leaks; an extra mask only
    hides a harmless number.
    """
    tp = fp = fn = 0
    for pred, exp in zip(predicted, expected):
        p, e = set(pred), set(exp)
        tp += len(p & e)
        fp += len(p - e)
        fn += len(e - p)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0}
