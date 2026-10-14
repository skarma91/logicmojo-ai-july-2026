"""Module 4 milestone (reference solution): the Domain Knowledge Assistant.

This ties the whole module together. It answers questions from a private corpus,
with citations (RAG, classes 4.5 and 4.6), and generates the answer through the
one model wrapper we have used all module (llm.py, classes 4.1 onward). The SAME
code answers with a plain base model or with your fine-tuned house-style model:
you only change which model llm.py selects.

    RAG supplies the FACTS (retrieved, cited, updatable).
    Fine-tuning supplies the STYLE (concise, always cites the source and section).

The corpus here is real, public-domain health information condensed from MedlinePlus
(U.S. National Library of Medicine). This assistant gives GENERAL health information,
not medical advice; that disclaimer is part of the house style and is printed with
every answer.

How the two halves plug together
--------------------------------
Part 1 (RAG): retrieve the most relevant passages from a Chroma store, then ask
the model to answer using ONLY those passages and cite them.

Part 2 (fine-tuned component): the generator is chosen by llm.py from .env. To use
your class 4.10 model, convert the merged model to GGUF, register it with Ollama
(`ollama create dka-assistant -f Modelfile`), then run with PROVIDER=ollama and
MODEL_NAME=dka-assistant (or pass --model dka-assistant). Nothing else changes,
which is exactly the point: the fine-tuned model is a drop-in generator.

Run:
    pip install langchain langchain-community langchain-chroma langchain-huggingface \\
                sentence-transformers chromadb python-dotenv
    # plus your provider, e.g.:  pip install langchain-ollama
    python assistant.py "What is a normal blood pressure reading?"
    python assistant.py --demo
    python assistant.py --model dka-assistant "..."   # answer with the tuned model

Model calls are REAL (through llm.get_chat_model()); there is no mocking. The
retrieval and the citation assembly are pure Python and unit-testable offline.
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

    `.content` is a plain string for Ollama and Groq, but for some Gemini replies
    it is a list of content parts (dicts with a 'text' key). This returns the text
    in both cases, so the provider switch stays transparent.
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
# Part 1: retrieval (the RAG half) -- Chroma store + optional cross-encoder rerank
# ---------------------------------------------------------------------------
def build_store(corpus):
    """Load the corpus into a persisted Chroma store (vectors + text + metadata)."""
    import os
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    persist_dir = str(DATA.parent / "chroma")
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        log.info("loaded persisted Chroma store from %s", persist_dir)
        return Chroma(persist_directory=persist_dir, embedding_function=emb)

    docs = [Document(page_content=r["text"],
                     metadata={"id": r["id"], "title": r["title"],
                               "source": r["source"], "section": r["section"],
                               "url": r["url"]})
            for r in corpus]
    store = Chroma.from_documents(docs, emb, persist_directory=persist_dir)
    log.info("built Chroma store with %d passages", len(docs))
    return store


def retrieve(store, query: str, k: int = 3, pool: int = 6, rerank: bool = True):
    """Return the top-k Documents for a query.

    With rerank on (default): pull a wider `pool` by vector similarity, then
    re-score (query, passage) pairs with a cross-encoder and keep the best k
    (classes 4.6). With rerank off: plain top-k similarity, the naive baseline.
    """
    if not rerank:
        return store.similarity_search(query, k=k)

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = store.similarity_search(query, k=pool)
    scores = reranker.predict([(query, c.page_content) for c in candidates])
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]
    return ranked[:k]


# ---------------------------------------------------------------------------
# Citation assembly (pure Python, unit-testable offline)
# ---------------------------------------------------------------------------
def format_context(docs) -> str:
    """Number the passages so the model can cite them as [1], [2], ..."""
    return "\n".join(
        f"[{i}] {d.page_content} (source: {d.metadata['source']}: {d.metadata['title']} - {d.metadata['section']})"
        for i, d in enumerate(docs, 1)
    )


def sources_list(docs) -> list[str]:
    """A human-readable source line per retrieved passage, in citation order."""
    return [f"[{i}] {d.metadata['source']}: {d.metadata['title']} - {d.metadata['section']} ({d.metadata['url']})"
            for i, d in enumerate(docs, 1)]


# ---------------------------------------------------------------------------
# Part 2: generation (the model half) -- provider or fine-tuned model via llm.py
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
    """Retrieve, then generate a grounded, cited answer. Returns (text, sources)."""
    if model is None:
        model = llm.get_chat_model()
    docs = retrieve(store, question, k=k, rerank=rerank)
    prompt = PROMPT.format(context=format_context(docs), question=question)
    text = message_text(model.invoke(prompt)).strip()   # REAL model call
    log.info("answered %r using %d passages", question, len(docs))
    return text, sources_list(docs)


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
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
