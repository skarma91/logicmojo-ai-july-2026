# Micro-assignment 4.5: a RAG loop over movie plots

Build the RAG loop from class (retrieve, cited prompt, generate) on a **new** dataset: real movie plots (`data/movies.jsonl`), not the IRS tax corpus. Work in `assignment.py`. Encoding is transformer-based (`all-MiniLM-L6-v2`) + FAISS, exactly as in class; generation goes through the real `llm.chat` (shipped as `llm.py`). The embedding model downloads on first run (internet), and generation needs a provider (Ollama local, or a Groq/Gemini key in `.env`); without one, fall back to an extractive answer.

Each record has `id`, `title`, `year`, `genre`, `director`, `source_url`, `plot`. Note the two `The Italian Job` entries (1969 and 2003), a remake pair.

## Setup

```
pip install sentence-transformers faiss-cpu numpy python-dotenv streamlit
pip install ollama        # or set GEMINI_API_KEY / GROQ_API_KEY in .env
```

## Problems

1. **Retriever.** Write `build_retriever()` (encode plots with `all-MiniLM-L6-v2`, index with `faiss.IndexFlatIP`) and `retrieve(model, index, query, k, year=None)` that over-fetches, keeps the given year, and attaches each hit's cosine `score`. **Expected:** `retrieve(..., "hunting replicants", 1)` returns `blade-runner-1982` with a `score` field.

2. **Build the prompt.** Write `build_prompt(question, hits)` listing the retrieved plots as numbered, sourced context (cite title and year) with a grounding instruction. **Expected:** the prompt contains `[1]` and a source like `The Italian Job 2003`.

3. **Answer.** Write `answer(model, index, question, k, year=None)` that retrieves, builds the prompt, and calls `llm.chat` (falling back to an extractive answer if no provider). **Expected:** a grounded reply citing `[1]`.

4. **Trigger the remake ambiguity.** Ask `"In The Italian Job, how do the thieves get away?"` with `k=2` and no filter. Because the corpus has both the 1969 and 2003 versions, retrieval mixes them. **Expected:** the retrieved set includes both `italian-job-1969` and `italian-job-2003`.

5. **Fix it, and explain (reasoning).** Re-run with `year=2003`, so only the remake is retrieved and the answer is grounded in that film. Write one or two sentences on why retrieval, not the model, decided which film you answered about. **Expected:** the retrieved set is only the 2003 version, plus your reasoning.

6. **A UI over the loop.** In `app.py`, build a small Streamlit front end over your functions: a question box, the grounded answer with its sources, and an expander that lists each retrieved plot with its title, year, and `score`. Add no new retrieval logic; import and reuse `build_retriever`, `retrieve`, and `answer`. **Expected:** `streamlit run app.py` answers a question and, with no year filter, shows both `italian-job-1969` and `italian-job-2003` in the panel; setting the year to 2003 leaves only the remake.

## How this is checked

A reference solution is in the `solution/` folder (`solution.py` plus `solution/app.py`). Because retrieval uses a real model and generation is a real call, your exact text will vary; the loop structure, the version filter, the citations, and the visible retrieved-plots panel are what matter.
