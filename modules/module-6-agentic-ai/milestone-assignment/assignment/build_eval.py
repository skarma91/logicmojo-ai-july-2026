"""Build the trajectory eval set from a 20% holdout of the corpus (given).

The team retrieves; it is not trained on the corpus, so no paper is removed from the
index. What we hold out is a 20% slice of papers used to WRITE tasks, so the task set
is a fixed, representative sample rather than questions you tuned the prompts against.

Each task in data/eval.jsonl says what a good run looks like, not only a good answer
(class 6.7, Evaluating agentic systems):

  answerable    one held-out paper is the gold source.
                reference path:  paper_finder > tool:search_papers > summarizer > critic
                success:         the answer cites the gold arXiv id and mentions a key term
  unanswerable  an off-topic question the corpus cannot answer.
                reference path:  paper_finder > tool:search_papers
                forbidden:       summarizer (summarizing unrelated papers is the failure)
                success:         the answer is exactly the team's NO_RESULTS message

    python build_eval.py               # 20% holdout, seed 13
    python build_eval.py --frac 0.3
"""

from __future__ import annotations
import argparse
import json
import pathlib
import random
import re
from collections import Counter

import retriever
from team import NO_RESULTS

OUT = pathlib.Path(__file__).parent / "data" / "eval.jsonl"
ANSWER_PATH = ["paper_finder", "tool:search_papers", "summarizer", "critic"]
DECLINE_PATH = ["paper_finder", "tool:search_papers"]

# Questions a corpus of LLM and agent papers does not answer.
UNANSWERABLE = [
    "What is the boiling point of ethanol at sea level?",
    "Which team won the 2018 FIFA World Cup final?",
    "How long should sourdough bread proof before baking?",
    "What is the capital gains tax rate on collectibles in the United States?",
    "How do monarch butterflies navigate during migration?",
]
STOP = set("""about above after again against also among based because before being below
between both could does doing during each first from further have having here however
into itself large language model models more most much only other over paper propose
proposed same should show shows some such than that their them then there these they
this those through under until using very well what when where which while with within
would results approach method methods tasks task performance work also llms large""".split())


def key_terms(text: str, title: str, n: int = 3) -> list[str]:
    """Distinctive words from the abstract that are not in the question itself."""
    in_title = {w.lower() for w in re.findall(r"[A-Za-z]+", title)}
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{4,}", text)]
    counts = Counter(w for w in words if w not in STOP and w not in in_title)
    return [w for w, _ in counts.most_common(n)]


def main():
    ap = argparse.ArgumentParser(description="Build the milestone trajectory eval set.")
    ap.add_argument("--frac", type=float, default=0.2, help="holdout fraction used to write tasks")
    ap.add_argument("--seed", type=int, default=13, help="random seed (a reproducible split)")
    args = ap.parse_args()

    papers = retriever.load_corpus()
    rng = random.Random(args.seed)
    rng.shuffle(papers)
    held = papers[:max(1, round(len(papers) * args.frac))]

    tasks = []
    for i, p in enumerate(held, 1):
        tasks.append({"id": f"a{i}", "kind": "answerable",
                      "question": f'What does the paper "{p["title"]}" propose, and how is it evaluated?',
                      "gold_id": p["id"], "reference": ANSWER_PATH, "forbidden": [],
                      "expected_args": {"search_papers": {"contains_any": key_terms(p["title"], "", 4)}},
                      "success": {"cites_any": [p["id"]], "mentions_any": key_terms(p["text"], p["title"])}})
    for i, q in enumerate(UNANSWERABLE, 1):
        tasks.append({"id": f"u{i}", "kind": "unanswerable", "question": q, "gold_id": None,
                      "reference": DECLINE_PATH, "forbidden": ["summarizer"], "expected_args": {},
                      "success": {"declines": NO_RESULTS}})

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"corpus: {len(papers)} papers ({retriever.corpus_path().name})")
    print(f"wrote {len(tasks)} tasks to data/eval.jsonl: {len(held)} answerable "
          f"(a {args.frac:.0%} holdout, seed {args.seed}) + {len(UNANSWERABLE)} unanswerable")


if __name__ == "__main__":
    main()
