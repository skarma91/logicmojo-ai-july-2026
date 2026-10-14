"""Build an evaluation set for the assistant (given).

RAG is not trained on the corpus, so we do NOT hold documents out of the retriever.
What we hold out is a 20% slice of items used to WRITE eval questions, so the eval set
is a fixed, representative sample rather than questions you tuned against by hand.

It writes data/eval.jsonl with two kinds of items:
  - answerable: a question whose answer is in a specific corpus passage. We record the
    gold passage id, so eval.py can measure retrieval hit@k, and a keyword the answer
    should contain, so it can check the outcome without a model.
  - unanswerable: a question the corpus does not cover, so eval.py can measure whether
    the assistant correctly DECLINES instead of inventing an answer.

Run:
    python build_eval.py                 # writes data/eval.jsonl
    python build_eval.py --frac 0.2      # holdout fraction (default 0.2)
"""

from __future__ import annotations
import argparse
import json
import pathlib
import random
import re

import assistant   # reuses assistant.DATA (prefers corpus.jsonl, else example_corpus)

OUT = pathlib.Path(__file__).parent / "data" / "eval.jsonl"

# Questions the health corpus does not answer, to test correct refusal.
UNANSWERABLE = [
    "What is the capital of Australia?",
    "How do I change a car tire?",
    "What year was the telephone invented?",
    "How do I file a US tax return?",
]


def _keyword(text: str) -> str:
    """Pick a distinctive content word from a passage for an offline outcome check."""
    words = re.findall(r"[A-Za-z]{5,}", text)
    stop = {"blood", "health", "which", "there", "these", "their", "about", "often",
            "usually", "people", "cause", "risk", "important", "helps"}
    for w in words:
        if w.lower() not in stop:
            return w
    return words[0] if words else "health"


def main():
    ap = argparse.ArgumentParser(description="Build the milestone eval set.")
    ap.add_argument("--frac", type=float, default=0.2, help="holdout fraction for questions")
    ap.add_argument("--seed", type=int, default=13, help="random seed (reproducible split)")
    args = ap.parse_args()

    docs = assistant.load_corpus()
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_hold = max(1, round(len(docs) * args.frac))
    held = docs[:n_hold]

    items = []
    for d in held:
        items.append({
            "question": f"According to MedlinePlus, what should I know about {d['title']} ({d['section']})?",
            "gold_id": d["id"],
            "answerable": True,
            "keyword": _keyword(d["text"]),
        })
    for q in UNANSWERABLE:
        items.append({"question": q, "gold_id": None, "answerable": False, "keyword": None})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} eval items to {OUT} "
          f"({n_hold} answerable from a {args.frac:.0%} holdout, {len(UNANSWERABLE)} unanswerable)")


if __name__ == "__main__":
    main()
