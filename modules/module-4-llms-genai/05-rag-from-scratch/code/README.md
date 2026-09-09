# Class 4.5 code: RAG from scratch

`main.py` is the Domain Knowledge Assistant: a hand-built retrieval-augmented
generation loop over a real corpus of IRS tax publications (public domain). It
retrieves the relevant passages, builds a cited prompt, and answers using only
those passages, citing the publication, tax year, and page. It then shows the
stale-document failure (last year's figure retrieved) and fixes it with a
metadata filter.

## What ships here

```
main.py                         the RAG build (run this)
app.py                          a Streamlit UI over main.py (optional)
llm.py                          unified LLM access: gemini | groq | ollama
data/corpus.jsonl               the real corpus, ready to use (shipped)
data/make_corpus.py             regenerates the shipped sample corpus
data/fetch_corpus.py            builds the FULL corpus from the original PDFs
data/precompute_embeddings.py   caches dense embeddings for the corpus
```

`corpus.jsonl` is the same schema as class 4.4 (Embeddings and vector
databases): `id`, `pub`, `title`, `tax_year`, `page`, `source_url`, `text`.

## Run

```
pip install sentence-transformers faiss-cpu numpy python-dotenv
python main.py
```

The embedding step downloads a small model (`all-MiniLM-L6-v2`) on first use, so
it needs internet. No GPU or paid API.

## Run the UI (optional)

`app.py` is a thin Streamlit front end over the same functions in `main.py`. It
adds no RAG logic; it draws a question box, the grounded answer with sources, and
an expander that shows the retrieved chunks and their similarity scores, so the
bad-chunk failure is visible on screen instead of only in the console.

```
pip install streamlit
streamlit run app.py
```

Ask the standard-deduction question with the tax-year filter on "All years",
then switch it to 2025 and open the panel: the stale 2024 passage drops out.

## Choosing the LLM provider

Generation goes through `llm.chat` in `llm.py`, which supports three providers.
Pick one with a `.env` file at the project root (copy `.env.example`):

```
PROVIDER=ollama            # gemini | groq | ollama   (default: ollama)
MODEL_NAME=                # optional; blank uses the per-provider default
GEMINI_API_KEY=...         # only if PROVIDER=gemini
GROQ_API_KEY=...           # only if PROVIDER=groq
```

Per-provider default model: gemini -> `gemini-3.5-flash-lite`, groq ->
`openai/gpt-oss-20b`, ollama -> `llama3.2`. Install only the SDK you use:

```
pip install ollama         # local default; then: ollama serve && ollama pull llama3.2
pip install google-genai   # for PROVIDER=gemini
pip install groq           # for PROVIDER=groq
```

If no provider is reachable, `main.py` falls back to an extractive answer (the
top retrieved passage), so retrieval, citations, and the stale-document fix are
all visible even with no model configured.

## The data workflow

The same three steps whether or not you use the real PDFs. The only difference
is whether you run `fetch_corpus.py`. Run from `data/`.

```
cd data

# 1. Get corpus.jsonl. Either use the shipped sample (skip this line) or rebuild
#    it from the real PDFs:
pip install requests docling     # or: pip install pypdf (lightweight fallback)
python fetch_corpus.py           # downloads the PDFs -> data/pdfs/, writes corpus.jsonl

# 2. Embed the corpus once and cache the vectors:
python precompute_embeddings.py  # writes embeddings.npy and ids.json

# 3. Run the build (from the code/ folder): it loads the cache.
```

`main.py` always reads `corpus.jsonl`, and `build_retriever` loads
`embeddings.npy`/`ids.json` when they exist and match that corpus, so the run is
instant. If the cache is missing or stale, it embeds once and writes the cache
itself, so step 2 is optional but recommended. `make_corpus.py` regenerates the
shipped sample `corpus.jsonl` if you want to reset it.

## What is committed vs generated

Committed: `main.py`, `data/corpus.jsonl`, and the three `data/*.py` scripts.
Generated locally and git-ignored: `data/pdfs/`, `data/embeddings.npy`,
`data/ids.json`.

## Note on the shipped corpus

`data/corpus.jsonl` is a small, hand-built sample condensed from public-domain
IRS publications; the figures were verified but the wording is paraphrased and
page numbers are representative. Run `fetch_corpus.py` for the exact text and
true page numbers.
