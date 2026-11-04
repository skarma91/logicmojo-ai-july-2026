"""The guardrail gateway: one safety and compliance layer around the whole team.

Every request passes these stages, cheapest first, and stops at the first one that
decides its fate:

  1. RULES       (local, free)     length cap and obvious prompt-attack patterns
  2. REDACT      (local, free)     PII -> typed placeholders; from here on nothing
                                   downstream (log, classifier, router, team, trace)
                                   ever sees a raw value
  3. ROUTE       (local, cheap)    semantic router picks a path by meaning:
                                   team | refuse | canned | fallback
  4. CLASSIFY    (one model call)  only for requests headed to the TEAM, the one
                                   path that can act: gpt-oss-safeguard-20b against
                                   a written policy
  5. TEAM        (several calls)   the class 6.1 (Multi-agent architectures) team, given the redacted request
  6. OUTPUT GUARD (local)          re-scan the answer for PII; require a citation

Why route before classifying: the refuse, canned, and fallback paths never reach a
model, so a classifier call there buys nothing. The expensive check guards the only
path with power. Class 5.6 (Reliability and failure modes) put rails INSIDE one agent;
here they wrap the team from outside, so the team code does not change at all.

The audit log (logs/audit.jsonl) records every decision, and only redacted text.
"""

from __future__ import annotations
import json
import logging
import pathlib
import re
import time
from dataclasses import dataclass, field

import pii
import policy
from router import SemanticRouter, FALLBACK

log = logging.getLogger("course.gateway")
MAX_CHARS = 1500                              # a request longer than this is rejected outright
AUDIT = pathlib.Path(__file__).parent / "logs" / "audit.jsonl"
CITE = re.compile(r"\[[a-z0-9]+(?:-[a-z0-9]+)+\]")

RULES = [   # (pattern, category): only patterns obvious enough to need no model
    (re.compile(r"\bignore (all |any |your |the )?(previous |prior |above )?(instructions|rules)\b", re.I), "prompt_attack"),
    (re.compile(r"\b(system|hidden|developer) prompt\b", re.I), "prompt_attack"),
]

REPLIES = {
    "refuse:personal_records": "I can only help with your own tax questions. I cannot look up or "
                               "share another person's tax, financial, or identity records.",
    "refuse:medical_records": "I cannot access anyone's medical records. If you are asking whether "
                              "your own medical costs are deductible, ask me that directly.",
    "canned:account_help": "Payments and account changes are handled in the secure account portal, "
                           "not in this chat. Please do not share card numbers here.",
    "fallback": "I answer questions about US federal tax rules from IRS publications. "
                "Could you rephrase your question in those terms?",
    "blocked": "I cannot help with that request.",
    "no_citation": "I could not find a cited source for that, so I will not guess. "
                   "Try rephrasing the question.",
}


@dataclass
class Decision:
    outcome: str                  # answer | blocked | refused | canned | fallback
    stage: str                    # where the decision was made
    reason: str = ""
    redacted: str = ""
    pii: list = field(default_factory=list)          # [(kind, original_text)] stays in memory only
    route: str = ""
    score: float = 0.0
    verdict: dict = field(default_factory=dict)
    answer: str = ""
    output_notes: list = field(default_factory=list)


class Gateway:
    def __init__(self, team=None, run_team=None, router: SemanticRouter | None = None,
                 audit_path: pathlib.Path = AUDIT):
        self.team, self.run_team = team, run_team    # team=None: stop before the team (dry run)
        self.router = router or SemanticRouter()
        self.audit_path = audit_path
        self.audit_path.parent.mkdir(exist_ok=True)

    # --- the pipeline ---------------------------------------------------------
    def handle(self, text: str) -> Decision:
        # 1. Rules: on the raw text, locally, before anything leaves the process.
        if len(text) > MAX_CHARS:
            return self._done(Decision("blocked", "rules", f"too long ({len(text)} chars > {MAX_CHARS})"), text)
        for pattern, category in RULES:
            if pattern.search(text):
                return self._done(Decision("blocked", "rules", category), text)

        # 2. Redact: everything after this line sees placeholders only.
        redacted, vault, spans = pii.redact(text)
        found = [(s.kind, s.text) for s in spans]

        # 3. Route by meaning.
        route, score, _ = self.router.route(redacted)
        action = self.router.actions.get(route, "fallback")
        if route == FALLBACK:
            return self._done(Decision("fallback", "router", f"best score {score:.2f} < tau",
                                       redacted, found, route, score), text)
        if action in ("refuse", "canned"):
            outcome = "refused" if action == "refuse" else "canned"
            return self._done(Decision(outcome, "router", route, redacted, found, route, score,
                                       answer=REPLIES[f"{action}:{route}"]), text)

        # 4. Classify: only traffic headed to the team pays for a model call.
        verdict = policy.classify(redacted)
        if verdict["violation"]:
            return self._done(Decision("blocked", "classifier", verdict.get("category") or "violation",
                                       redacted, found, route, score, verdict, REPLIES["blocked"]), text)

        d = Decision("answer", "team", "", redacted, found, route, score, verdict)
        if self.team is None:                          # dry run: stop at the door of the team
            d.answer = "(team not run)"
            return self._done(d, text)

        # 5. The team sees only the redacted request.
        state = self.run_team(self.team, redacted)
        # 6. Output guard.
        d.answer, d.output_notes = self.output_guard(state.get("explanation") or state.get("findings") or "")
        return self._done(d, text)

    def output_guard(self, answer: str) -> tuple[str, list]:
        notes = []
        clean, _, spans = pii.redact(answer)
        if spans:
            notes.append(f"redacted {len(spans)} PII span(s) in the answer")
        if not CITE.search(clean):
            notes.append("no citation: answer withheld")
            clean = REPLIES["no_citation"]
        return clean, notes

    # --- audit log: redacted text and decisions only ------------------------------
    def _done(self, d: Decision, raw_text: str) -> Decision:
        if not d.redacted:                               # blocked at the rules stage
            d.redacted = pii.redact(raw_text)[0]
        if d.outcome == "blocked" and not d.answer:
            d.answer = REPLIES["blocked"]
        record = {"ts": round(time.time(), 3), "outcome": d.outcome, "stage": d.stage,
                  "reason": d.reason, "route": d.route, "score": round(d.score, 3),
                  "pii_kinds": sorted({k for k, _ in d.pii}),
                  "request": d.redacted[:300], "answer": d.answer[:300]}
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        log.info("gateway %s at %s (%s)", d.outcome, d.stage, d.reason)
        return d
