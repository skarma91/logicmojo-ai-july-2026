# Class 4.6 code: RAG with frameworks (LangChain + Chroma)

`main.py` refactors the hand-built RAG loop from class 4.5 onto LangChain's LCEL
with a **Chroma** vector store, over the same real IRS tax-publication corpus. It
adds query rewriting and a cross-encoder re-ranker, and A/Bs retrieval quality
(hit@3) against the naive baseline. It also includes a small **HyDE** demo
(`make_hyde_retrieve`): draft a hypothetical answer with the model, then retrieve
the real passages nearest to that draft. HyDE is a query-time technique, so it
costs one extra model call per query; the A/B prints its hit@3 next to the others
so you measure it rather than assume.

## Why Chroma (not the FAISS index from 4.4 and 4.5)

FAISS is a vector *index*: it stores only vectors, so in 4.5 we kept text and
metadata in a parallel list and filtered by hand. Chroma is a vector *database*:
it stores vectors, text, and metadata together, filters on metadata natively in
the query, persists to disk, runs embedded (free, no server), and has a
first-class LangChain integration. That removes the plumbing we wrote by hand.
FAISS is still fine for a fast in-memory index; Qdrant, Weaviate, or pgvector are
the step up for a networked or managed store.

## What ships here

```
main.py                         the framework build (run this)
llm.py                          unified LLM access; get_chat_model() for LangChain
data/corpus.jsonl               the real corpus, ready to use (shipped)
data/make_corpus.py             regenerates the shipped sample corpus
data/fetch_corpus.py            builds the FULL corpus from the original PDFs (Docling)
```

Chroma persists its index to `data/chroma/` on first build; later runs reopen
that store instead of re-embedding, so there is no separate embeddings cache here
(unlike classes 4.4 and 4.5, which cache a numpy file). `data/chroma/` is
generated and git-ignored.

## Run

```
pip install langchain langchain-community langchain-chroma langchain-huggingface \
            sentence-transformers chromadb python-dotenv
python main.py
```

## Choosing the LLM provider

The chat model comes from `llm.get_chat_model()`, which uses LangChain's
`init_chat_model` selected from the same `.env` as the rest of the course (copy
`.env.example` at the project root):

```
PROVIDER=ollama            # gemini | groq | ollama   (default: ollama)
MODEL_NAME=                # optional; blank uses the per-provider default
GEMINI_API_KEY=...         # only if PROVIDER=gemini
GROQ_API_KEY=...           # only if PROVIDER=groq
```

Install the matching LangChain integration for your provider:

```
pip install langchain-ollama          # PROVIDER=ollama (default); run `ollama serve`
pip install langchain-groq            # PROVIDER=groq
pip install langchain-google-genai    # PROVIDER=gemini
```

These are **real** model calls (chat model and the query rewrite), so a provider
must be set up. Every call is logged: `llm.chat` logs directly, and the LangChain
chain logs through a callback (`build_callbacks`).

## The data workflow

From `data/`: `make_corpus.py` regenerates the shipped sample; `fetch_corpus.py`
builds the full corpus from the real PDFs (Docling primary, pypdf fallback).
`main.py` reads `corpus.jsonl` either way, and caching is handled by Chroma's
persisted `data/chroma/` store (no separate precompute step here, unlike classes
4.4 and 4.5).
