"""Precompute dense embeddings for corpus.jsonl, once, where you have internet.

The class build embeds on the fly, but caching the vectors makes later runs
instant and lets an offline machine reuse them.

    pip install sentence-transformers numpy
    python precompute_embeddings.py     # writes embeddings.npy and ids.json

embeddings.npy holds an L2-normalized float32 matrix (one row per corpus record,
in file order); ids.json holds the matching record ids so you can line them up.
"""

from __future__ import annotations
import json
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).parent
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(l) for l in (HERE / "corpus.jsonl").open()]
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode([r["text"] for r in rows], normalize_embeddings=True)
    np.save(HERE / "embeddings.npy", np.asarray(vectors, dtype="float32"))
    (HERE / "ids.json").write_text(json.dumps([r["id"] for r in rows]))
    print(f"embedded {len(rows)} records with {MODEL_NAME}; wrote embeddings.npy and ids.json")


if __name__ == "__main__":
    main()
