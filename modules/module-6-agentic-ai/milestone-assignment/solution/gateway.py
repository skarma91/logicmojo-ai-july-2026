"""The guardrail gateway around the research team (class 6.4, Guardrails, routing, and compliance).

    question -> admit (length and pattern rules, then PII redaction) -> team -> output guard

Everything after admit() sees placeholders, never a raw email, phone number, or ID,
and that includes the model provider, the trace, and the audit log. The output guard
checks the answer before the user sees it: every cited arXiv id must be one the team
actually retrieved, and personal data is redacted again.
"""

from __future__ import annotations
import json
import pathlib
import re
import time

import pii
from team import CITE, NO_RESULTS

MAX_CHARS = 1500
AUDIT = pathlib.Path(__file__).parent / "logs" / "audit.jsonl"
RULES = [re.compile(r"\bignore (all |any |your |the )?(previous |prior |above )?(instructions|rules)\b", re.I),
         re.compile(r"\b(system|hidden|developer) prompt\b", re.I)]
BLOCKED = "I can only help with questions about research papers in this corpus."


# ---------------------------------------------------------------------------
# TODO 4: admission, in the right order
# ---------------------------------------------------------------------------
def admit(question: str) -> tuple[bool, str, list[str]]:
    """(admitted, text, pii_kinds).

    Rules run first, on the raw text, locally: too long, or matching a RULES pattern,
    means (False, reason, []). Otherwise redact, and return (True, redacted_text, kinds)
    so nothing downstream ever sees the raw value."""
    if len(question) > MAX_CHARS:
        return False, f"too long ({len(question)} chars > {MAX_CHARS})", []
    if any(r.search(question) for r in RULES):
        return False, "matches a blocked pattern", []
    redacted, _vault, spans = pii.redact(question)
    return True, redacted, sorted({s.kind for s in spans})


def output_guard(answer: str, retrieved: set) -> tuple[str, list[str]]:
    """Redact the answer and drop it if it cites a paper the team never retrieved."""
    notes = []
    clean, _, spans = pii.redact(answer)
    if spans:
        notes.append(f"redacted {len(spans)} PII span(s) in the answer")
    cited = set(CITE.findall(clean))
    if cited - retrieved:
        notes.append(f"cited unretrieved papers {sorted(cited - retrieved)}: answer withheld")
        clean = NO_RESULTS
    return clean, notes


def handle(team_run, question: str) -> dict:
    """Run one question through the gateway and the team; audit only redacted text."""
    ok, text, kinds = admit(question)
    if not ok:
        result = {"outcome": "blocked", "answer": BLOCKED, "reason": text, "pii": [], "papers": []}
    else:
        state = team_run(text)
        retrieved = {f["id"] for f in state["findings"]}
        answer, notes = output_guard(state["answer"], retrieved)
        outcome = "answer" if CITE.search(answer) else "no_results"
        result = {"outcome": outcome, "answer": answer, "reason": "; ".join(notes), "pii": kinds,
                  "papers": [{"id": f["id"], "title": f["title"]} for f in state["findings"]],
                  "route": state["route_log"], "reviews": state["reviews"]}
    AUDIT.parent.mkdir(exist_ok=True)
    with open(AUDIT, "a") as f:
        f.write(json.dumps({"ts": round(time.time(), 3), "outcome": result["outcome"],
                            "request": text if ok else pii.redact(question)[0],
                            "pii_kinds": kinds, "answer": result["answer"][:300]}) + "\n")
    return result
