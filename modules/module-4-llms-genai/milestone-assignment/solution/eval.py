"""Evaluate the assistant's retrieval and answers (given).

This is the milestone's evidence step. It reads data/eval.jsonl (from build_eval.py)
and reports the numbers that tell you whether the RAG core works, reusing the metrics
from class 4.8 (Evaluating LLM and RAG). It is also where the RAG-versus-fine-tuning
claim gets settled: run it with the base model and again with your tuned model
(--model dka-assistant) and compare.

Metrics
-------
Always (no model needed):
  - hit@1, hit@3: is the gold passage in the top-1 / top-3 retrieved? (retrieval quality)

When a provider is set in .env (real model calls):
  - keyword recall: does the grounded answer contain the expected fact word? (outcome)
  - faithfulness: an LLM judge checks the answer is supported by the retrieved passages
  - refusal accuracy: on unanswerable questions, does the assistant correctly decline?

Run:
    python eval.py                       # base model (or the .env default)
    python eval.py --model dka-assistant # your fine-tuned model, to compare
"""

from __future__ import annotations
import argparse
import json
import pathlib

import assistant

EVAL = pathlib.Path(__file__).parent / "data" / "eval.jsonl"
JUDGE_PROMPT = (
    "You are a strict grader. Is the ANSWER fully supported by the CONTEXT? Reply with "
    "one word: YES or NO.\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{ans}"
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the milestone assistant.")
    ap.add_argument("--model", default=None, help="MODEL_NAME override (e.g. dka-assistant)")
    ap.add_argument("--k", type=int, default=3, help="passages to retrieve")
    args = ap.parse_args()

    items = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    answerable = [it for it in items if it["answerable"]]
    unanswerable = [it for it in items if not it["answerable"]]
    store = assistant.build_store(assistant.load_corpus())

    # --- retrieval metrics (no model) ---
    hit1 = hit3 = 0
    for it in answerable:
        docs = assistant.retrieve(store, it["question"], k=args.k, rerank=False)
        ids = [d.metadata["id"] for d in docs]
        hit1 += int(bool(ids) and ids[0] == it["gold_id"])
        hit3 += int(it["gold_id"] in ids)
    n = len(answerable)
    print(f"retrieval:  hit@1 = {hit1}/{n} = {hit1/n:.0%}   hit@3 = {hit3}/{n} = {hit3/n:.0%}")

    # --- model metrics (only if a provider is configured) ---
    try:
        import llm
        model = llm.get_chat_model(model=args.model) if args.model else llm.get_chat_model()
        kw_ok = faithful = 0
        for it in answerable:
            text, _ = assistant.answer(store, it["question"], model=model, k=args.k, rerank=False)
            kw_ok += int(it["keyword"] and it["keyword"].lower() in text.lower())
            docs = assistant.retrieve(store, it["question"], k=args.k, rerank=False)
            ctx = assistant.format_context(docs)
            verdict = assistant.message_text(model.invoke(JUDGE_PROMPT.format(ctx=ctx, ans=text)))
            faithful += int("yes" in verdict.strip().lower()[:5])
        declined = 0
        for it in unanswerable:
            text, _ = assistant.answer(store, it["question"], model=model, k=args.k, rerank=False)
            declined += int(any(p in text.lower() for p in ("do not find", "not in the", "don't find")))
        print(f"outcome:    keyword recall = {kw_ok}/{n} = {kw_ok/n:.0%}")
        print(f"faithful:   supported = {faithful}/{n} = {faithful/n:.0%}")
        print(f"refusal:    declined = {declined}/{len(unanswerable)} = {declined/len(unanswerable):.0%}")
    except Exception as e:
        print(f"\nmodel metrics skipped (set PROVIDER and a key in .env to enable): {e}")


if __name__ == "__main__":
    main()
