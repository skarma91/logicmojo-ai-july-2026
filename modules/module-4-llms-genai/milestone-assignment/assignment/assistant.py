"""Module 4 milestone (starter): the Domain Knowledge Assistant.

Complete the TODOs to build an assistant that answers questions from a private
corpus WITH citations (RAG, classes 4.5 and 4.6), generated through the one model
wrapper you have used all module (llm.py). The same code should answer with a base
model or your fine-tuned house-style model (class 4.10): you only change which model
llm.py selects.

    RAG supplies the FACTS (retrieved, cited, updatable).
    Fine-tuning supplies the STYLE (concise, always cites the source and section).

The corpus is real, public-domain health information condensed from MedlinePlus
(U.S. National Library of Medicine). This assistant gives GENERAL health information,
not medical advice; print that disclaimer with every answer.

Run:
    pip install langchain langchain-community langchain-chroma langchain-huggingface \\
                sentence-transformers chromadb python-dotenv
    # plus your provider, e.g.:  pip install langchain-ollama
    python assistant.py "What is a normal blood pressure reading?"
    python assistant.py --demo

A worked reference is in ../solution/. Compare your outputs to it.
"""

from __future__ import annotations
import argparse
import json
import logging
import pathlib

import llm  # get_chat_model(): the LangChain chat model, selected from .env

_DATADIR = pathlib.Path(__file__).parent / "data"
# Prefer the downloaded corpus (fetch_data.py); fall back to the committed sample.
DATA = _DATADIR / "corpus.jsonl" if (_DATADIR / "corpus.jsonl").exists() else _DATADIR / "example_corpus.jsonl"
log = logging.getLogger("course.assistant")


def message_text(resp) -> str:
    """Read the text out of a LangChain chat reply, for any provider.

    `.content` is a string for Ollama and Groq, but a list of parts for some
    Gemini replies. This returns the text in both cases. (Given, so the provider
    switch stays transparent; you do not need to change this.)
    """
    content = resp.content
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict):
            parts.append(p.get("text", ""))
        else:
            parts.append(getattr(p, "text", "") or "")
    return "".join(parts)


def load_corpus(path=DATA) -> list[dict]:
    """Load the private passages with their metadata (id, title, source, section, url)."""
    return [json.loads(line) for line in open(path)]


# ---------------------------------------------------------------------------
# Part 1: retrieval (the RAG half)
# ---------------------------------------------------------------------------
def build_store(corpus):
    """TODO: load the corpus into a persisted Chroma store.

    Use HuggingFaceEmbeddings("all-MiniLM-L6-v2"). Each Document should carry the
    text as page_content and {id, title, source, section, url} as metadata. Persist
    to data/chroma and reopen it on later runs instead of re-embedding (classes 4.6).
    Return the Chroma store.
    """
    raise NotImplementedError


def retrieve(store, query: str, k: int = 3, pool: int = 6, rerank: bool = True):
    """TODO: return the top-k Documents for a query.

    rerank=False: plain top-k similarity (the naive baseline).
    rerank=True: pull a wider `pool` by similarity, then re-score (query, passage)
    pairs with a CrossEncoder ("cross-encoder/ms-marco-MiniLM-L-6-v2") and keep the
    best k (classes 4.6).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Citation assembly (pure Python)
# ---------------------------------------------------------------------------
def format_context(docs) -> str:
    """TODO: number the passages so the model can cite them as [1], [2], ...
    Include the source, title, and section in each line so a citation is traceable.
    """
    raise NotImplementedError


def sources_list(docs) -> list[str]:
    """TODO: return one human-readable source line per retrieved passage, in
    citation order (e.g. "[1] MedlinePlus: High Blood Pressure - Blood pressure
    categories (https://medlineplus.gov/ency/article/000468.htm)").
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: generation (the model half)
# ---------------------------------------------------------------------------
PROMPT = (
    "You are a health information assistant. Answer the question using ONLY the "
    "context passages. Cite the passages you use like [1]. If the answer is not in "
    "the context, say you do not find it in the documents. Be concise, and give "
    "general information only, not medical advice.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

DISCLAIMER = ("General health information from MedlinePlus, not medical advice. "
              "For personal medical concerns, talk to a healthcare professional.")


def answer(store, question: str, model=None, k: int = 3, rerank: bool = True):
    """TODO: retrieve, then generate a grounded, cited answer.

    Steps: get a model with llm.get_chat_model() if none was passed; retrieve the
    passages; build PROMPT with format_context; call the model (use message_text on
    the reply); return (answer_text, sources_list(docs)).
    """
    raise NotImplementedError


DEMO_QUESTIONS = [
    "What is a normal blood pressure reading?",
    "How much sleep do adults need?",
    "How can I lower my risk of heart disease?",
]


def main():
    ap = argparse.ArgumentParser(description="Domain Knowledge Assistant (RAG + your model).")
    ap.add_argument("question", nargs="?", help="a question to answer")
    ap.add_argument("--model", default=None, help="override MODEL_NAME (e.g. dka-assistant for the tuned model)")
    ap.add_argument("--k", type=int, default=3, help="passages to retrieve")
    ap.add_argument("--no-rerank", action="store_true", help="use the naive baseline retriever")
    ap.add_argument("--demo", action="store_true", help="run the demo questions")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    store = build_store(load_corpus())
    model = llm.get_chat_model(model=args.model) if args.model else llm.get_chat_model()

    questions = DEMO_QUESTIONS if args.demo or not args.question else [args.question]
    for q in questions:
        text, sources = answer(store, q, model=model, k=args.k, rerank=not args.no_rerank)
        print(f"\nQ: {q}\n{text}\n\nSources:")
        for s in sources:
            print("  " + s)


if __name__ == "__main__":
    main()
