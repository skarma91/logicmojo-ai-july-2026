"""Dense retrieval over arXiv abstracts (given).

The corpus is data/corpus.jsonl when you have run fetch_data.py, otherwise the
committed data/example_corpus.jsonl (20 real papers). Embeddings use FastEmbed, the
same MiniLM model as the class builds on the ONNX runtime, with no PyTorch, so the
exact same code fits the 512 MB container in class 6.8 (Deployment).

Vectors are cached to data/embeddings.npy plus data/ids.json and reused while the
corpus is unchanged; the Dockerfile builds the cache at image build time.
"""

from __future__ import annotations
import json
import os
import pathlib

import numpy as np

DATA_DIR = pathlib.Path(__file__).parent / "data"
EMB, IDS = DATA_DIR / "embeddings.npy", DATA_DIR / "ids.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_STATE = {"model": None, "matrix": None, "corpus": None}


def corpus_path() -> pathlib.Path:
    full = DATA_DIR / "corpus.jsonl"
    return full if full.exists() else DATA_DIR / "example_corpus.jsonl"


class _FastEmbedder:
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


def load_corpus() -> list[dict]:
    return [json.loads(l) for l in corpus_path().read_text().splitlines() if l.strip()]


def load_index() -> int:
    corpus = load_corpus()
    ids = [p["id"] for p in corpus]
    if EMB.exists() and IDS.exists() and json.loads(IDS.read_text()) == ids:
        matrix = np.load(EMB)
    else:
        docs = [f"{p['title']}. {p['text']}" for p in corpus]     # title + abstract
        matrix = _normalize(_embedder().encode(docs))
        np.save(EMB, matrix)
        IDS.write_text(json.dumps(ids))
    _STATE["matrix"], _STATE["corpus"] = matrix, corpus
    return len(corpus)


def search(query: str, k: int = 4) -> list[dict]:
    """Top-k papers with id, title, year, url, abstract, and cosine score."""
    if _STATE["matrix"] is None:
        load_index()
    q = _normalize(_embedder().encode([query]))[0]
    scores = _STATE["matrix"] @ q
    out = []
    for i in np.argsort(-scores)[:k]:
        p = _STATE["corpus"][int(i)]
        out.append({**{key: p[key] for key in ("id", "title", "year", "url", "text")},
                    "score": float(scores[i])})
    return out
