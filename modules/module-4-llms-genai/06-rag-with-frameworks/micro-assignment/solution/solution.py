"""Micro-assignment 4.6: measure a retrieval-quality delta (movie plots), on the framework.

Same idea as the class (A/B two retrievers, report hit@k) on a NEW dataset: real
movie plots (data/movies.jsonl), not the IRS tax corpus. Both retrievers use the
class 4.6 framework stack, not the raw FAISS of class 4.5: the naive one is dense
retrieval over a LangChain Chroma store (all-MiniLM-L6-v2); the upgraded one
retrieves a wider pool from the same store and re-ranks it with a cross-encoder,
exactly as in class. The models download on first run, so this needs internet.

Run:  python solution.py
    pip install langchain-chroma langchain-huggingface chromadb sentence-transformers
"""

import json
import pathlib

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "movies.jsonl"
ROWS = [json.loads(l) for l in open(DATA)]

# Labeled set: a question and the id of the film it is about.
LABELED = [
    ("hunting artificial people in a rainy future city", "blade-runner-1982"),
    ("a boy travels to the land of the dead to find family", "coco-2017"),
    ("scientists bring dinosaurs back and it goes wrong", "jurassic-park-1993"),
    ("a con man robs three casinos in one night", "oceans-eleven-2001"),
    ("entering people's dreams to plant an idea", "inception-2010"),
    ("the fancy-car heist crew that steals gold", "italian-job-2003"),
]


# ---- 1. Chroma store: the framework's vector database (in-memory here) ----
def build_store():
    """Load the movie plots into a LangChain Chroma store.

    Each Document carries the plot text AND its metadata (id, title, year), so the
    store returns everything we need. The corpus is tiny (18 plots), so we keep the
    store in memory, no persist_directory, unlike the class build over the larger
    IRS corpus.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=r["plot"],
                     metadata={"id": r["id"], "title": r["title"], "year": r["year"]})
            for r in ROWS]
    return Chroma.from_documents(docs, emb)


# ---- 2. Naive retriever: plain dense similarity over the Chroma store ----
def make_naive(store, k=3):
    def retrieve(query):
        hits = store.similarity_search(query, k=k)
        return [h.metadata["id"] for h in hits]
    return retrieve


# ---- 3. Upgraded retriever: wide dense pool from Chroma, then cross-encoder rerank ----
def make_upgraded(store, reranker, k=3, pool=6):
    def retrieve(query):
        # Stage 1: dense retrieve a wider pool from the same Chroma store.
        cand = store.similarity_search(query, k=pool)
        # Stage 2: the cross-encoder re-scores (query, plot) pairs together, then keep top-k.
        scores = reranker.predict([(query, d.page_content) for d in cand])
        ranked = [d for _, d in sorted(zip(scores, cand), key=lambda x: -x[0])]
        return [d.metadata["id"] for d in ranked[:k]]
    return retrieve


# ---- 4. hit@k and evaluate (pure Python, no model) ----
def hit_at_k(retrieved_ids, gold_id, k=3):
    return int(gold_id in retrieved_ids[:k])


def evaluate(retrieve_fn, k=3):
    return sum(hit_at_k(retrieve_fn(q), g, k) for q, g in LABELED) / len(LABELED)


# ---- 5. A/B the two retrievers and report the delta ----
if __name__ == "__main__":
    from sentence_transformers import CrossEncoder

    store = build_store()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    naive = make_naive(store, k=3)
    upgraded = make_upgraded(store, reranker, k=3)
    print(f"naive     hit@3: {evaluate(naive):.2f}")
    print(f"upgraded  hit@3: {evaluate(upgraded):.2f}")
    # Expected: the upgraded (rerank) pipeline scores at least as high as the naive
    # one, and higher on ambiguous queries where the first-pass ranking puts the
    # right film just outside the top-k. The exact numbers depend on the models; the
    # point is to MEASURE the delta on a labeled set, not to assume the upgrade helped.
