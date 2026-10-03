"""Evaluate the assistant's retrieval and answers (given).

This is the milestone's evidence step. It reads data/eval.jsonl (from build_eval.py)
and reports the numbers that tell you whether the RAG core actually works, reusing the
metrics from class 4.8 (Evaluating LLM and RAG) and the agent-eval idea from class 5.6
(outcome, not just a vibe check).

Metrics
-------
Always (no model needed):
  - hit@1, hit@3: is the gold passage in the top-1 / top-3 retrieved? (retrieval quality)

When a provider is set in .env (real model calls):
  - keyword recall: does the grounded answer contain the expected fact word? (outcome)
  - faithfulness: an LLM judge checks the answer is supported by the retrieved passages
  - refusal accuracy: on unanswerable questions, does the assistant correctly decline?

Run:
    python eval.py                 # hit@k always; model metrics if PROVIDER is set
"""

from __future__ import annotations
import json
import pathlib

import nasa_retriever

EVAL = pathlib.Path(__file__).parent / "data" / "eval.jsonl"

ANSWER_PROMPT = (
    "You are a NASA information assistant. Answer the question using ONLY the context "
    "passages, and cite them like [1]. If the answer is not in the context, say you do "
    "not find it in the documents. Be concise.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)
JUDGE_PROMPT = (
    "You are a strict grader. Is the ANSWER fully supported by the CONTEXT? Reply with "
    "one word: YES or NO.\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{ans}"
)


def _context(docs) -> str:
    return "\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, 1))


def rag_answer(question: str, chat, k: int = 3):
    """Retrieve, then generate one grounded answer (a plain RAG call, no agent loop)."""
    docs = nasa_retriever.search(question, k=k)
    text = chat([{"role": "user", "content": ANSWER_PROMPT.format(ctx=_context(docs), q=question)}])
    return text.strip(), docs


def main():
    items = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    answerable = [it for it in items if it["answerable"]]
    unanswerable = [it for it in items if not it["answerable"]]

    # --- retrieval metrics (no model) ---
    hit1 = hit3 = 0
    for it in answerable:
        ids = [d["id"] for d in nasa_retriever.search(it["question"], k=3)]
        hit1 += int(ids[:1] == [it["gold_id"]] or (ids and ids[0] == it["gold_id"]))
        hit3 += int(it["gold_id"] in ids)
    n = len(answerable)
    print(f"retrieval:  hit@1 = {hit1}/{n} = {hit1/n:.0%}   hit@3 = {hit3}/{n} = {hit3/n:.0%}")

    # --- model metrics (only if a provider is configured) ---
    try:
        import llm
        chat = llm.chat
        kw_ok = faithful = 0
        for it in answerable:
            ans, docs = rag_answer(it["question"], chat)
            kw_ok += int(it["keyword"] and it["keyword"].lower() in ans.lower())
            verdict = chat([{"role": "user", "content": JUDGE_PROMPT.format(
                ctx=_context(docs), ans=ans)}])
            faithful += int("yes" in verdict.strip().lower()[:5])
        declined = 0
        for it in unanswerable:
            ans, _ = rag_answer(it["question"], chat)
            declined += int(any(p in ans.lower() for p in ("do not find", "not in the", "don't find")))
        print(f"outcome:    keyword recall = {kw_ok}/{n} = {kw_ok/n:.0%}")
        print(f"faithful:   supported = {faithful}/{n} = {faithful/n:.0%}")
        print(f"refusal:    declined = {declined}/{len(unanswerable)} = {declined/len(unanswerable):.0%}")
    except Exception as e:
        print(f"\nmodel metrics skipped (set PROVIDER and a key in .env to enable): {e}")


if __name__ == "__main__":
    main()
