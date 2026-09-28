"""The Module 4 retriever, packaged as a reusable tool for the agent.

This is the dense retriever from classes 4.4 and 4.5 (embed the corpus, embed the
query, take the top-k by cosine similarity), wrapped in a single `search(query)`
function so the agent can call it as one of its tools. It caches the corpus
embeddings to `embeddings.npy` plus `ids.json` on first run and reuses them after,
so the workflow is identical whether the corpus is the shipped sample or a real one
(the class 4.4 cache pattern).

Install:
    pip install sentence-transformers numpy
"""

from __future__ import annotations
import json
import logging
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"
EMB = DATA.parent / "embeddings.npy"
IDS = DATA.parent / "ids.json"
MODEL_NAME = "all-MiniLM-L6-v2"
log = logging.getLogger("course.retriever")

_STATE = {"model": None, "matrix": None, "corpus": None}


def _load_corpus() -> list[dict]:
    return [json.loads(line) for line in open(DATA)]


def _embedder():
    """Load the embedding model once and reuse it."""
    if _STATE["model"] is None:
        from sentence_transformers import SentenceTransformer
        _STATE["model"] = SentenceTransformer(MODEL_NAME)
    return _STATE["model"]


def _normalize(m: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product equals cosine similarity (class 4.4)."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def _build_index():
    """Embed the corpus once, cache to disk, and hold it in memory.

    On later runs (whether corpus.jsonl is the shipped sample or a real fetch) we
    reopen the cached vectors instead of re-embedding, so the workflow is identical.
    """
    corpus = _load_corpus()
    ids = [r["id"] for r in corpus]

    if EMB.exists() and IDS.exists() and json.load(open(IDS)) == ids:
        matrix = np.load(EMB)
        log.info("loaded cached embeddings for %d passages", len(ids))
    else:
        model = _embedder()
        matrix = _normalize(np.asarray(model.encode([r["text"] for r in corpus])))
        np.save(EMB, matrix)
        json.dump(ids, open(IDS, "w"))
        log.info("embedded and cached %d passages", len(ids))

    _STATE["matrix"] = matrix
    _STATE["corpus"] = corpus


def search(query: str, k: int = 3) -> list[dict]:
    """Return the top-k passages most similar to the query.

    Each result is a dict with id, text, pub, tax_year, and page, so the agent can
    quote and cite it. This is the single function the agent registers as a tool.
    """
    if _STATE["matrix"] is None:
        _build_index()
    q = _normalize(np.asarray(_embedder().encode([query])))[0]
    scores = _STATE["matrix"] @ q                       # cosine, since rows are normalized
    top = np.argsort(-scores)[:k]
    hits = []
    for i in top:
        r = _STATE["corpus"][int(i)]
        hits.append({"id": r["id"], "text": r["text"], "pub": r["pub"],
                     "tax_year": r["tax_year"], "page": r["page"], "score": float(scores[i])})
    log.info("retriever.search %r -> %s", query, [h["id"] for h in hits])
    return hits
