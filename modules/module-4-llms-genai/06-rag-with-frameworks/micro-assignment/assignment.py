"""Micro-assignment 4.6: measure a retrieval-quality delta (movie plots), on the framework.

New dataset (data/movies.jsonl). Both retrievers use the class 4.6 stack, not the
raw FAISS of class 4.5: the naive one is dense retrieval over a LangChain Chroma
store (all-MiniLM-L6-v2); the upgraded one retrieves a wider pool from the same
store and re-ranks it with a cross-encoder. See README.md. Models download on
first run (needs internet).

    pip install langchain-chroma langchain-huggingface chromadb sentence-transformers
"""

import json
import pathlib

DATA = pathlib.Path(__file__).resolve().parent / "data" / "movies.jsonl"
ROWS = [json.loads(l) for l in open(DATA)]

LABELED = [
    ("hunting artificial people in a rainy future city", "blade-runner-1982"),
    ("a boy travels to the land of the dead to find family", "coco-2017"),
    ("scientists bring dinosaurs back and it goes wrong", "jurassic-park-1993"),
    ("a con man robs three casinos in one night", "oceans-eleven-2001"),
    ("entering people's dreams to plant an idea", "inception-2010"),
    ("the fancy-car heist crew that steals gold", "italian-job-2003"),
]

# ---- 1. build_store(): a LangChain Chroma store over the plots ----
# Each plot is a Document with metadata {"id","title","year"}; embed with
# all-MiniLM-L6-v2 (HuggingFaceEmbeddings). Keep it in memory (no persist_directory).
# your code here

# ---- 2. make_naive(store, k): dense top-k ids via store.similarity_search ----
# your code here

# ---- 3. make_upgraded(store, reranker, k, pool): dense wide pool -> cross-encoder rerank ----
# your code here

# ---- 4. hit_at_k(ids, gold, k) and evaluate(retrieve_fn, k) ----
# your code here

# ---- 5. Print naive vs upgraded hit@3 and reason about when rerank helps ----
if __name__ == "__main__":
    # your code here
    pass
