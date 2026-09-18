# Micro-assignment 4.8: evaluate the movie-plot assistant

Apply the class metrics to a **new** dataset: the movie-plot assistant over `data/movies.jsonl`, not the IRS corpus. Work in `assignment.py`. You will score faithfulness, answer relevance, and context relevance with an LLM-as-judge (real `llm.chat` calls through the shipped `llm.py`), plus context recall as a deterministic check, then read the metrics together. The embedding model downloads on first run, and the judge needs a provider (Ollama local, or a Groq/Gemini key in `.env`).

## Setup

```
pip install sentence-transformers faiss-cpu numpy python-dotenv
pip install ollama        # or set GEMINI_API_KEY / GROQ_API_KEY in .env
```

## Problems

1. **The assistant.** Write `build_retriever()` (`all-MiniLM-L6-v2` + `faiss.IndexFlatIP` over the plots) and `answer(model, index, question, k)` that retrieves the top-k plots and answers **only** from them, replying `"I do not find that in these plots."` when the answer is not present. **Expected:** for the Blade Runner question, the answer names Blade Runner and the retrieved set includes `blade-runner-1982`.

2. **Context recall.** Write `context_recall(gold_id, retrieved_ids)` returning 1.0 if the gold id was retrieved, else 0.0. **Expected:** `context_recall("coco-2017", ["coco-2017", ...])` is `1.0`.

3. **Judge metrics.** Write `faithfulness`, `answer_relevance`, and `context_relevance` as LLM-as-judge calls that each return a number in [0, 1]. **Expected:** each returns a float in [0, 1]; a correct "not found" answer scores high on faithfulness.

4. **Evaluate.** Write `evaluate(model, index)` that averages each metric over `MOVIE_EVAL` (compute recall only where a gold id exists) and print the four means. **Expected:** four numbers in [0, 1].

5. **Weakest and a fix (reasoning).** Name the lowest metric and state one change you would try, matched to it. **Expected:** the weakest metric plus a justified fix (for example, low recall points at the retriever, so raise k or add a re-ranker from class 4.6).

## How this is checked

A reference solution is in the `solution/` folder. Because the judge and the answers are real model calls, exact scores vary; the metric definitions, the deterministic recall, and reading the metrics together are what matter.
