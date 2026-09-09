# Micro-assignment 4.4: retrieval over a movie-plot corpus

Apply the class pattern (chunk, embed, FAISS, metadata filter, hybrid) to a **new** dataset you have not seen: real movie plots (`data/movies.jsonl`), not the IRS tax corpus from the class notebook. Encoding is transformer-based (`all-MiniLM-L6-v2`), exactly as in class, so the embedding cells download the model on first run and need internet. `rank_bm25` is used only as the keyword baseline for the hybrid comparison.

Each record has `id`, `title`, `year`, `genre`, `director`, `source_url`, and `plot`. Note the two `The Italian Job` entries (1969 and 2003), a remake pair.

## Setup

```
pip install sentence-transformers faiss-cpu rank-bm25 numpy
```

## Problems

1. **Chunk a page.** Word-window chunker (size 20, overlap 5) on the first three plots joined. This part runs offline. **Expected:** page words 71, chunks 5, overlap check `True`.

2. **Embed and search.** Encode the plots with `all-MiniLM-L6-v2`, build a `faiss.IndexFlatIP`, and write `search(query, k)`. **Expected:** `search("a detective hunting artificial humans in a future city", 3)` returns `blade-runner-1982` at the top, matched by meaning despite few shared words.

3. **Metadata filter (pick the remake version).** Write `search_filtered(query, year, k)` (over-fetch, then keep the year). **Expected:** without a filter both `italian-job-1969` and `italian-job-2003` appear for "gold heist with mini coopers"; `search_filtered(..., 2003)` returns `italian-job-2003`. Same idea as the tax-year filter in class.

4. **Hybrid: meaning plus keywords.** Add a BM25 keyword baseline and a min-max blend of BM25 and dense scores. **Expected:** for `"the one about the dream heist"`, keyword-only ranks `italian-job-2003` first (the wrong film, matched on "heist"), while the dense/hybrid ranking lifts `inception-2010`.

5. **Why meaning wins (reasoning).** Show the dense search finds Inception for that query where BM25 does not, and explain why in one or two sentences. **Expected:** `inception-2010` near the top of the dense results (BM25 ranked it last), plus your reasoning.

## How this is checked

A reference solution is in the `solution/` folder. The chunking and BM25 parts are exact; the dense-embedding rankings depend on the model, so compare the ordering and behavior rather than exact scores.
