# Class 4.8 code: evaluating LLM and RAG

`main.py` evaluates the class 4.5 RAG assistant over the real IRS corpus on a
30-item eval set, scoring four metrics: faithfulness, answer relevance, context
relevance (all hand-built LLM-as-judge calls), and context recall (a deterministic
check). It prints the four means and names the weakest, the one to fix first.

## What ships here

```
main.py                     the eval harness (run this)
llm.py                      unified LLM access: gemini | groq | ollama
data/corpus.jsonl           the real IRS corpus (same as 4.4-4.6)
data/eval_set.jsonl         the 30-item eval set (question, gold_id, reference)
data/make_eval_set.py       regenerates eval_set.jsonl from a curated list
data/make_corpus.py         regenerates the shipped sample corpus
data/fetch_corpus.py        builds the full corpus from the original PDFs
```

## Run

```
pip install sentence-transformers faiss-cpu numpy python-dotenv
pip install ollama      # or set GEMINI_API_KEY / GROQ_API_KEY in .env
python main.py
```

The judge is a real model, so use a capable one: a small local model is a weak
grader, so this is where a hosted or frontier model in `.env` helps. Embeddings
are cached to `data/embeddings.npy` on first run (git-ignored).

## The metrics

- **faithfulness** = supported claims / total claims in the answer (judge).
- **answer relevance** = how directly the answer addresses the question (judge).
- **context relevance** = relevant retrieved passages / retrieved (judge).
- **context recall** = was the gold passage retrieved (deterministic, from the
  labeled `gold_id`), the same idea as hit@k in class 4.6.

Read them together: low recall points at the retriever; low faithfulness at the
prompt or model. Exact judge scores vary run to run; the method is the point.

## RAGAS

The standard library (RAGAS) computes the same ideas. A reference block at the
bottom of `main.py` shows the call, and notes the dependency pitfall (RAGAS pins
LangChain versions that often clash with a current install), which is why the core
here is hand-built and RAGAS is a reference.

## Milestone

This eval set is part of the Module 4 milestone. Rerun it whenever you change the
retriever, the model, or add a fine-tune (class 4.10), to see if the numbers moved.
