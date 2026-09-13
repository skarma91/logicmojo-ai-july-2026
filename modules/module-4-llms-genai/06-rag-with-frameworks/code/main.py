"""Class 4.6 build: RAG on a framework (LangChain), plus the upgrades that help.

This refactors the hand-built loop from class 4.5 onto LangChain's LCEL with a
Chroma vector store, over the same real IRS tax-publication corpus. It then adds
two upgrades, query rewriting and a cross-encoder re-ranker, and A/Bs the
upgraded retrieval against the naive baseline on a small labeled set.

Run with:
    pip install langchain langchain-community langchain-chroma langchain-huggingface \\
                sentence-transformers chromadb python-dotenv
    # plus the chat provider you use, e.g.:  pip install langchain-ollama
    python main.py

WHY CHROMA (and not the FAISS index from 4.4 and 4.5)
-----------------------------------------------------
FAISS is a pure vector INDEX: it stores only the vectors, so in 4.5 we had to
keep the text and metadata in a parallel Python list and filter by hand after
over-fetching. Chroma is a vector DATABASE: it stores the vectors, the document
text, AND the metadata together; it filters on metadata natively inside the
query (a `where`/`filter` clause, no hand-rolled loop); it persists to disk so
you build the index once; and it has a first-class LangChain integration
(langchain-chroma) that drops straight into a retriever and an LCEL chain. It is
also free and runs embedded (no server). For a framework-based build that is
exactly what we want. (FAISS is still great when you only need a fast in-memory
index; Qdrant, Weaviate, or pgvector are the step up when you need a networked or
managed store, which we note but do not require.)

The model calls are REAL and all go through LangChain: the chat model comes from
llm.get_chat_model() (selected from PROVIDER/MODEL_NAME/.env), and the query
rewrite invokes that same model. The evaluation metric (hit_at_k) is pure Python.
"""

from __future__ import annotations
import json
import logging
import pathlib

import llm   # get_chat_model(): the LangChain chat model, selected from .env

DATA = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"
log = logging.getLogger("course.rag")


def message_text(resp) -> str:
    """Read the text out of a LangChain chat reply, for any provider.

    `model.invoke(...)` returns a message whose `.content` is a plain string for
    Ollama and Groq, but for some Gemini responses it is a list of content parts
    (each a dict like {"type": "text", "text": "..."}). Calling `.content.strip()`
    then fails on the list. This helper returns the text in both cases, so the
    provider switch stays transparent. (The LCEL chain uses StrOutputParser, which
    already handles this; we only need it for direct model.invoke calls.)
    """
    content = resp.content
    if isinstance(content, str):
        return content
    parts = []
    for p in content:                      # a list of content parts
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict):
            parts.append(p.get("text", ""))
        else:
            parts.append(getattr(p, "text", "") or "")
    return "".join(parts)


def load_corpus(path=DATA) -> list[dict]:
    """Load the real IRS passages with their metadata."""
    return [json.loads(line) for line in open(path)]


# A small labeled set: question -> the id of the passage that best answers it.
# This lets us MEASURE retrieval quality (hit@k) instead of eyeballing it.
LABELED = [
    ("how much of my medical bills can I deduct", "p502-2025-threshold"),
    ("can I deduct dental work", "p502-2025-dental"),
    ("how much can I contribute to my HSA", "p969-2025-limit"),
    ("standard deduction for a single filer", "p501-2025-stdded"),
    ("mortgage interest deduction limit", "p936-2025-limit"),
    ("the amount went up this year, what is it now", "p501-2025-stdded"),  # vague: rewrite helps
]


# ===========================================================================
# Pure evaluation metric (no model needed, unit-testable offline)
# ===========================================================================
def hit_at_k(retrieved_ids: list[str], gold_id: str, k: int = 3) -> int:
    """1 if the gold passage is among the top-k retrieved ids, else 0."""
    return int(gold_id in retrieved_ids[:k])


def evaluate(retrieve_fn, labeled=LABELED, k: int = 3) -> float:
    """Mean hit@k of a retrieve function over the labeled set."""
    return sum(hit_at_k(retrieve_fn(q), gold, k) for q, gold in labeled) / len(labeled)


# ===========================================================================
# Upgrade 1: query rewriting with a REAL model
# ===========================================================================
def rewrite_query(query: str, model) -> str:
    """Ask the LangChain chat model to turn a vague question into a search query."""
    prompt = (
        "Rewrite the user's question into a short, specific search query with the "
        "key terms a document search would match. Return ONLY the query.\n"
        f"Question: {query}"
    )
    # model.invoke returns a message; message_text reads its text for any provider.
    new = message_text(model.invoke(prompt)).strip().strip('"')
    log.info("query rewritten %r -> %r", query, new)
    return new or query


# ===========================================================================
# The Chroma store and the two retrievers
# ===========================================================================
def build_store(corpus):
    """Load the corpus into a Chroma vector database (vectors + text + metadata).

    Chroma persists to disk, so we point it at a persist_directory and build once.
    On later runs (whether corpus.jsonl is the shipped sample or was rebuilt from
    the real PDFs by fetch_corpus.py) we reopen the persisted store instead of
    re-embedding. This is Chroma's framework-native equivalent of the embeddings
    cache used in classes 4.4 and 4.5 (there a numpy file; here Chroma's own store).
    """
    import os
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    persist_dir = str(DATA.parent / "chroma")         # code/data/chroma
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Reopen the persisted store if we already built it; otherwise build and persist.
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        log.info("loaded persisted Chroma store from %s", persist_dir)
        return Chroma(persist_directory=persist_dir, embedding_function=emb)

    # Each Document carries the text AND the metadata, so Chroma can filter on it.
    docs = [Document(page_content=r["text"],
                     metadata={"id": r["id"], "pub": r["pub"],
                               "tax_year": r["tax_year"], "page": r["page"]})
            for r in corpus]
    store = Chroma.from_documents(docs, emb, persist_directory=persist_dir)
    log.info("built Chroma store with %d passages (persisted to %s)", len(docs), persist_dir)
    return store


def make_naive_retrieve(store, k: int = 3):
    """Baseline: plain similarity search, top-k ids."""
    def retrieve(query: str) -> list[str]:
        hits = store.similarity_search(query, k=k)
        ids = [h.metadata["id"] for h in hits]
        log.info("naive retrieve %r -> %s", query, ids)
        return ids
    return retrieve


def make_upgraded_retrieve(store, model, k: int = 3, pool: int = 6):
    """Upgraded: rewrite the query, retrieve a wider pool, then cross-encoder rerank."""
    from sentence_transformers import CrossEncoder
    # A cross-encoder reads (query, passage) TOGETHER and scores relevance more
    # accurately than the bi-encoder used for the initial vector search. It is too
    # slow to score the whole corpus, but perfect for re-scoring a small pool.
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(query: str) -> list[str]:
        q2 = rewrite_query(query, model)                      # upgrade 1
        candidates = store.similarity_search(q2, k=pool)       # retrieve wide
        pairs = [(q2, c.page_content) for c in candidates]
        scores = reranker.predict(pairs)                      # upgrade 2: rerank
        ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]
        ids = [c.metadata["id"] for c in ranked[:k]]
        log.info("upgraded retrieve %r -> %s (reranked from %d)", query, ids, len(candidates))
        return ids
    return retrieve


# ===========================================================================
# HyDE (Hypothetical Document Embeddings): a query-side architecture
# ===========================================================================
def make_hyde_retrieve(store, model, k: int = 3):
    """HyDE: instead of searching with the short question, ask the model to DRAFT
    a hypothetical answer, then retrieve the real passages nearest to that draft.

    Why it helps: a terse question and the passage that answers it often use
    different words, so their embeddings sit far apart. The draft reads like the
    target documents, so it lands in the right neighborhood. The draft may be
    wrong in its facts; that does not matter, because we only use it to search,
    then answer from the REAL passages it retrieves.

    Cost: this is a query-time technique, so it adds one LLM call before every
    retrieval (the draft), on top of the final answer call.
    """
    def retrieve(query: str) -> list[str]:
        prompt = (
            "Write a short, factual paragraph that would answer the question, as if "
            "quoting a tax guide. Being approximate is fine.\n"
            f"Question: {query}"
        )
        draft = message_text(model.invoke(prompt)).strip()   # the hypothetical answer (REAL call)
        hits = store.similarity_search(draft, k=k)      # retrieve near the DRAFT, not the question
        ids = [h.metadata["id"] for h in hits]
        log.info("hyde retrieve %r -> %s (via a %d-char draft)", query, ids, len(draft))
        return ids
    return retrieve


def current_year_retrieve(store, tax_year=2025, k=3):
    """Show Chroma's NATIVE metadata filter: keep only one tax year, inside the query.

    In 4.5 with FAISS we over-fetched and filtered by hand; here the store does it.
    """
    def retrieve(query: str) -> list[str]:
        hits = store.similarity_search(query, k=k, filter={"tax_year": tax_year})
        return [h.metadata["id"] for h in hits]
    return retrieve


# ===========================================================================
# The answer chain: the LCEL refactor of the 4.5 loop
# ===========================================================================
def format_docs(docs):
    return "\n".join(f"[{i}] {d.page_content} (source: {d.metadata['pub']} {d.metadata['tax_year']})"
                     for i, d in enumerate(docs, 1))


def build_answer_chain(store, model):
    """retriever | prompt | model | parser, all LangChain Runnables."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY the context. Cite sources like [1].\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    # The dict builds the prompt's two inputs; then pipe through prompt, model, parser.
    return ({"context": store.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | prompt | model | StrOutputParser())


def build_callbacks():
    """A LangChain callback that logs each model call, keeping the 'log every call'
    habit inside the framework path (the model runs inside the chain, so we hook it
    with a callback). LangSmith is the fuller framework-native tracing tool used later."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _Log(BaseCallbackHandler):
        def on_llm_end(self, response, **kw):
            log.info("llm call (langchain) provider=%s", llm.PROVIDER)
    return [_Log()]


# ===========================================================================
# A/B the two retrievers and show one grounded answer
# ===========================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    corpus = load_corpus()
    store = build_store(corpus)
    model = llm.get_chat_model()          # one LangChain chat model, from .env

    naive = make_naive_retrieve(store, k=3)
    upgraded = make_upgraded_retrieve(store, model, k=3)
    hyde = make_hyde_retrieve(store, model, k=3)          # query-side architecture demo
    naive_score, upgraded_score = evaluate(naive), evaluate(upgraded)
    hyde_score = evaluate(hyde)
    log.info("A/B hit@3 naive=%.2f upgraded=%.2f hyde=%.2f", naive_score, upgraded_score, hyde_score)
    print(f"naive     hit@3: {naive_score:.2f}")
    print(f"upgraded  hit@3: {upgraded_score:.2f}")
    print(f"hyde      hit@3: {hyde_score:.2f}")
    print("Expected: the upgraded pipeline scores at least as high, and higher on the")
    print("vague query, which query rewriting rescues. HyDE is a second query-side")
    print("option; measure it rather than assume, since it costs an extra call.\n")

    # Native metadata filter (Chroma does it in the query, no hand-rolled loop).
    print("current-year only:", current_year_retrieve(store)("standard deduction for a single filer"))

    # One grounded, cited answer through the LCEL chain (a real model call).
    chain = build_answer_chain(store, model)
    answer = chain.invoke("How much can I contribute to my HSA?", config={"callbacks": build_callbacks()})
    print("\nAnswer:\n" + answer)


if __name__ == "__main__":
    main()
