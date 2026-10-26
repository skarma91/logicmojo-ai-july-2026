"""Build an evaluation set for the assistant (given).

RAG is not trained on the corpus, so we do NOT hold documents out of the retriever.
What we hold out is a 20% slice of items used to WRITE eval questions, so the eval set
is a fixed, representative sample rather than questions you tuned against by hand.

It writes data/eval.jsonl with two kinds of items:
  - answerable: a question whose answer is in a specific corpus passage. We record the
    gold passage id, so eval.py can measure retrieval hit@k, and a keyword the answer
    should contain, so it can check the outcome without a model.
  - unanswerable: a question the corpus does not cover, so eval.py can measure whether
    the assistant correctly DECLINES instead of inventing an answer (class 5.6).

Run:
    python build_eval.py                 # writes data/eval.jsonl (needs a corpus)
    python build_eval.py --frac 0.2      # holdout fraction (default 0.2)
"""

from __future__ import annotations
import argparse
import json
import pathlib
import random
import re

import nasa_retriever   # reuses the same corpus the retriever uses

OUT = pathlib.Path(__file__).parent / "data" / "eval.jsonl"

# Questions the NASA corpus does not answer, to test correct refusal.
UNANSWERABLE = [
    "What is the recommended daily intake of vitamin C?",
    "How do I file a US tax return?",
    "What year did the Berlin Wall fall?",
    "What is the best recipe for sourdough bread?",
]


def _keyword(text: str) -> str:
    """Pick a distinctive content word from a passage for an offline outcome check."""
    words = re.findall(r"[A-Za-z]{5,}", text)
    stop = {"image", "space", "which", "shows", "these", "their", "using", "known",
            "about", "there", "where", "other", "small", "large", "first"}
    for w in words:
        if w.lower() not in stop:
            return w
    return words[0] if words else "NASA"


def main():
    ap = argparse.ArgumentParser(description="Build the milestone eval set.")
    ap.add_argument("--frac", type=float, default=0.2, help="holdout fraction for questions")
    ap.add_argument("--seed", type=int, default=13, help="random seed (reproducible split)")
    args = ap.parse_args()

    docs = list(nasa_retriever.DOCS)
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_hold = max(1, round(len(docs) * args.frac))
    held = docs[:n_hold]

    items = []
    for d in held:
        items.append({
            "question": f"What does NASA's '{d['title']}' show?",
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
