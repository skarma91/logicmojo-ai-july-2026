"""Class 6.4 micro-assignment (reference solution): block piracy requests at the door.

A movie team should never help anyone get a pirated copy of a film. This solution
builds the input guard that stops those requests before any agent sees them, in
two stages, cheapest first:

  1. RULES        word-boundary patterns over normalized text (free, local)
  2. CLASSIFIER   a written policy on gpt-oss-safeguard-20b, called ONLY for
                  requests the rules did not block but that contain an ambiguous
                  word ("free", "download", "stream", ...)

It is scored on data/requests.jsonl (16 labeled requests, including innocent ones
that share words with piracy requests) by block rate and false-block rate.

Run:  python assignment.py      (the classifier stage needs PROVIDER=groq or ollama in .env)
"""

from __future__ import annotations
import json
import logging
import pathlib
import re

import llm

log = logging.getLogger("course.piracy_guard")
DATA = pathlib.Path(__file__).parent / "data" / "requests.jsonl"
ROWS = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
SAFEGUARD_MODEL = "openai/gpt-oss-safeguard-20b"

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "@": "a", "$": "s"})

RULES = {
    "torrent": r"\btorrent(s|ing|ed)?\b",
    "magnet link": r"\bmagnet (link|uri)s?\b",
    "pirate site": r"\b(pirate ?bay|pirated|piracy)\b",
    "cam rip": r"\bcam ?rips?\b",
    "drm removal": r"\b(crack|strip|remove|bypass)\w* (the )?drm\b",
    "redistribution": r"\bupload (the )?(full|whole|entire) (movie|film)\b",
    "illegal streaming": r"\b(unofficial|illegal|not legal)\b.*\bstream",
}
AMBIGUOUS = ["free", "download", "stream", "watch online", "share", "upload", "copy"]

POLICY = """# Movie Piracy Policy

## INSTRUCTIONS
Classify one request sent to a movie recommendation assistant. Return JSON only:
{"violation": 0 or 1, "rationale": string}.

## DEFINITIONS
- Piracy: obtaining, sharing, or watching a copyrighted film without the rights
  holder's permission, including torrents, illegal streams, cam recordings, cracked
  copies, and links to such sources.

## VIOLATES (1)
- Asking where or how to get a film without paying when no legal free option is named.
- Asking for links, files, or sites that distribute films without permission.
- Asking how to remove copy protection to share a film.

## SAFE (0)
- Questions about films, casts, plots, or recommendations.
- Asking where to watch or buy a film legally, including official free offers.
- Questions about technology (for example, how a protocol works) that do not ask for a film.

## EXAMPLES
Content: "Where can I rent Coco legally?"  Answer: {"violation": 0, "rationale": "Legal rental."}
Content: "Any site to watch Coco free without an account?"  Answer: {"violation": 1, "rationale": "Seeks an unlicensed free stream."}"""


# Problem 1
def normalize(text: str) -> str:
    """Lowercase, undo simple digit and symbol swaps, collapse whitespace."""
    return " ".join(text.lower().translate(LEET).split())


# Problem 2
def rule_block(text: str) -> tuple[bool, str]:
    """(blocked, name of the rule that fired, or "")."""
    t = normalize(text)
    for name, pattern in RULES.items():
        if re.search(pattern, t):
            return True, name
    return False, ""


# Problem 3
def rates(preds: list[bool], labels: list[bool]) -> tuple[float, float]:
    """(block rate, false-block rate). An empty group gives 0.0, not a crash."""
    unsafe = [p for p, l in zip(preds, labels) if l]
    safe = [p for p, l in zip(preds, labels) if not l]
    block = sum(unsafe) / len(unsafe) if unsafe else 0.0
    false_block = sum(safe) / len(safe) if safe else 0.0
    return block, false_block


# Problem 4
NAIVE_WORDS = ["torrent", "pirate", "magnet", "download", "crack", "rip"]


def naive_block(text: str) -> bool:
    """A substring blocker, the version people write first."""
    t = text.lower()
    return any(w in t for w in NAIVE_WORDS)


def disagreements(rows: list[dict]) -> list[tuple[str, bool, bool, bool]]:
    """Requests where the naive blocker and the rules disagree:
    (id, naive verdict, rule verdict, label)."""
    out = []
    for r in rows:
        n, (b, _) = naive_block(r["text"]), rule_block(r["text"])
        if n != b:
            out.append((r["id"], n, b, r["unsafe"]))
    return out


# Problem 5
def needs_classifier(text: str) -> bool:
    """Send to the model only if the rules did not block AND an ambiguous word appears.
    A request with no rule hit and no ambiguous word passes without a model call."""
    if rule_block(text)[0]:
        return False
    t = normalize(text)
    return any(re.search(rf"\b{re.escape(w)}", t) for w in AMBIGUOUS)


def classify(text: str) -> bool:
    """True if the policy classifier says the request violates the piracy policy.
    Unreadable verdicts count as violations (fail closed)."""
    msgs = [{"role": "system", "content": POLICY},
            {"role": "user", "content": f"Content: {text!r}\nAnswer (JSON only):"}]
    try:
        reply = (llm.chat(msgs, model=SAFEGUARD_MODEL, temperature=0.0, agent="piracy_classifier")
                 if llm.PROVIDER == "groq" else llm.chat(msgs, temperature=0.0, agent="piracy_classifier"))
    except Exception as e:
        log.warning("safeguard model unavailable (%s); using the default model", e)
        reply = llm.chat(msgs, temperature=0.0, agent="piracy_classifier")
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        return bool(int(json.loads(m.group())["violation"])) if m else True
    except (ValueError, KeyError, TypeError):
        return True


def guard(text: str) -> tuple[bool, str]:
    """The two-stage guard: (blocked, stage)."""
    hit, name = rule_block(text)
    if hit:
        return True, f"rule:{name}"
    if needs_classifier(text):
        return classify(text), "classifier"
    return False, "pass"


def main():
    logging.basicConfig(level=logging.WARNING)
    labels = [r["unsafe"] for r in ROWS]
    print("[1] normalize('  T0rr3nt   the  M@trix ') ->", repr(normalize("  T0rr3nt   the  M@trix ")))
    for t in ["Where can I t0rrent Oppenheimer?", "Who directed Pirates of the Caribbean?",
              "Is there a torrential rain scene in Blade Runner?"]:
        print(f"[2] rule_block({t!r}) -> {rule_block(t)}")
    rule_preds = [rule_block(r["text"])[0] for r in ROWS]
    naive_preds = [naive_block(r["text"]) for r in ROWS]
    print("[3] rules only: block rate %.2f, false-block rate %.2f" % rates(rule_preds, labels))
    print("    naive only: block rate %.2f, false-block rate %.2f" % rates(naive_preds, labels))
    print("    rates([], []) ->", rates([], []))
    print("[4] disagreements (id, naive, rules, label):", disagreements(ROWS))
    sent = [r["id"] for r in ROWS if needs_classifier(r["text"])]
    print(f"[5] sent to the classifier: {sent}  ({len(sent)} of {len(ROWS)} requests; "
          f"{len(ROWS) - len(sent)} model calls saved)")

    llm.reset_usage()
    final = []
    print()
    for r in ROWS:
        blocked, stage = guard(r["text"])
        final.append(blocked)
        flag = "" if blocked == r["unsafe"] else "   <- wrong"
        print(f"  {r['id']}  {'BLOCK' if blocked else 'pass ':<5}  {stage:<24} label={'unsafe' if r['unsafe'] else 'safe'}{flag}")
    b, f = rates(final, labels)
    calls = sum(u["calls"] for u in llm.usage_by_agent().values())
    print(f"\ntwo-stage guard: block rate {b:.2f}, false-block rate {f:.2f}, classifier calls {calls}")


if __name__ == "__main__":
    main()
