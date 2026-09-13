# Micro-assignment 4.6: measure a retrieval-quality delta

Prove an upgrade helps, do not assume it. A/B two retrievers on a **new** dataset, real movie plots (`data/movies.jsonl`), not the IRS tax corpus, and report the hit@k delta. Work in `assignment.py`. Both retrievers use the class 4.6 framework stack, not the raw FAISS of class 4.5: the naive one is dense retrieval over a **LangChain Chroma** store (`all-MiniLM-L6-v2`); the upgraded one retrieves a wider pool from the same store and re-ranks it with a cross-encoder. The models download on first run (internet).

## Setup

```
pip install langchain-chroma langchain-huggingface chromadb sentence-transformers
```

## Problems

1. **Chroma store.** Write `build_store()` that loads the plots into a LangChain **Chroma** store, each plot a `Document` with metadata `{"id","title","year"}`, embedded with `all-MiniLM-L6-v2` (`HuggingFaceEmbeddings`). Keep it in memory (no `persist_directory`, the corpus is tiny). **Expected:** a store over 18 plots.

2. **Naive retriever.** Write `make_naive(store, k)` returning a function that gives the dense top-k ids via `store.similarity_search`. **Expected:** `naive("hunting artificial people in a rainy future city")` includes `blade-runner-1982`.

3. **Upgraded retriever.** Write `make_upgraded(store, reranker, k, pool)` that retrieves a wider `pool` from the store, then re-scores `(query, plot)` pairs with a `CrossEncoder` and keeps the top k. **Expected:** returns top-k ids after re-ranking.

4. **hit@k and evaluate.** Write `hit_at_k(ids, gold, k)` and `evaluate(retrieve_fn, k)` (mean over the labeled set). **Expected:** a float in [0, 1].

5. **Report the delta (reasoning).** Print naive and upgraded hit@3 and write one or two sentences on when re-ranking helps. **Expected:** upgraded hit@3 is at least as high as naive; the cross-encoder helps most on ambiguous queries where the first pass put the right film just outside the top-k. Exact numbers depend on the models, so measure them rather than assume.

## How this is checked

A reference solution is in the `solution/` folder. Because both retrievers use real models, exact scores vary; the two-stage pattern and the measurement are what matter.
