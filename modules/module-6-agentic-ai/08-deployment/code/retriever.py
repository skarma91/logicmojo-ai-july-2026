"""The retriever, put on a diet for a 512 MB container.

Same interface as the retriever from classes 4.4 (Embeddings and vector databases)
and 4.5 (RAG from scratch): search(query, k) returns passages with id, text, pub,
tax_year, page. Two changes make it fit Render's free tier:

  1. FastEmbed instead of sentence-transformers. FastEmbed runs the same MiniLM model
     (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions) on the ONNX runtime,
     with no PyTorch. PyTorch alone can exceed 512 MB of memory.
  2. The corpus vectors are computed ONCE, at image build time (build_index.py), and
     baked into the image as data/embeddings.npy plus data/ids.json. At request time
     only the query is embedded.

Build and query must use the same backend: FastEmbed's vectors are close to, but not
bit-identical with, sentence-transformers' vectors for the same model, so never mix a
cache from one with queries from the other.
"""

from __future__ import annotations
import json
import logging
import os
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"
EMB = DATA.parent / "embeddings.npy"
IDS = DATA.parent / "ids.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
log = logging.getLogger("course.retriever")

_STATE = {"model": None, "matrix": None, "corpus": None}


class _FastEmbedder:
    """Wraps FastEmbed with the .encode(list_of_texts) -> array interface the rest of
    the code (retriever, semantic router) already uses."""

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(MODEL_NAME, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))

    def encode(self, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        return np.asarray(list(self.model.embed(texts)), dtype="float32")


def _embedder():
    if _STATE["model"] is None:
        _STATE["model"] = _FastEmbedder()
    return _STATE["model"]


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def _load_corpus() -> list[dict]:
    return [json.loads(line) for line in open(DATA)]


def build_index() -> int:
    """Embed the whole corpus and write the cache. Run at image build time."""
    corpus = _load_corpus()
    matrix = _normalize(_embedder().encode([r["text"] for r in corpus]))
    np.save(EMB, matrix)
    json.dump([r["id"] for r in corpus], open(IDS, "w"))
    return len(corpus)


def load_index() -> int:
    """Load the baked vectors. Rebuild only if they are missing or stale (slow: avoid
    this in production by running build_index.py in the Dockerfile)."""
    corpus = _load_corpus()
    ids = [r["id"] for r in corpus]
    if not (EMB.exists() and IDS.exists() and json.load(open(IDS)) == ids):
        log.warning("no baked index matching the corpus; embedding %d passages now", len(ids))
        build_index()
    _STATE["matrix"], _STATE["corpus"] = np.load(EMB), corpus
    return len(corpus)


def search(query: str, k: int = 3) -> list[dict]:
    if _STATE["matrix"] is None:
        load_index()
    q = _normalize(_embedder().encode([query]))[0]
    scores = _STATE["matrix"] @ q
    hits = []
    for i in np.argsort(-scores)[:k]:
        r = _STATE["corpus"][int(i)]
        hits.append({"id": r["id"], "text": r["text"], "pub": r["pub"], "tax_year": r["tax_year"],
                     "page": r["page"], "score": float(scores[i])})
    return hits
