"""A small dense retriever over the NASA corpus (given).

The Module 4 idea (class 4.4, 4.5), on NASA passages: embed each description once with
sentence-transformers, cache the vectors to embeddings.npy, and answer a query by
cosine similarity. This is the retriever the agent wraps as a tool.

DATA SOURCE. It reads data/corpus.jsonl if present (the larger corpus you download
with fetch_data.py); otherwise it falls back to data/example_corpus.jsonl, the small
real sample committed to the repo. So the code runs out of the box, and gets more
interesting once you fetch the full corpus.

Install:  pip install sentence-transformers numpy
"""

from __future__ import annotations
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
FULL = HERE / "data" / "corpus.jsonl"
SAMPLE = HERE / "data" / "example_corpus.jsonl"
CACHE = HERE / "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def corpus_path() -> pathlib.Path:
    """Prefer the downloaded corpus; fall back to the committed sample."""
    return FULL if FULL.exists() else SAMPLE


DOCS = [json.loads(l) for l in corpus_path().read_text().splitlines() if l.strip()]
BY_ID = {d["id"]: d for d in DOCS}
_model = None
_matrix = None


def _encode(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    v = np.asarray(_model.encode(list(texts)), dtype=float)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def _index():
    """Build (and cache) the normalized passage embeddings once. The cache is keyed
    to the corpus size, so it rebuilds automatically if you swap in the full corpus."""
    global _matrix
    if _matrix is not None:
        return _matrix
    if CACHE.exists():
        cached = np.load(CACHE)
        if cached.shape[0] == len(DOCS):
            _matrix = cached
            return _matrix
    _matrix = _encode([d["text"] for d in DOCS])
    np.save(CACHE, _matrix)
    return _matrix


def search(query: str, k: int = 3) -> list[dict]:
    """Return the top-k NASA passages whose description best matches the query."""
    q = _encode([query])[0]
    sims = _index() @ q
    order = np.argsort(-sims)[:k]
    return [dict(DOCS[i], score=float(sims[i])) for i in order]
