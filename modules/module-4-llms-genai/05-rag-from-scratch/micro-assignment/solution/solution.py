"""Micro-assignment 4.5: reference solution (movie-plot RAG).

Same RAG loop as the class (retrieve, build a cited prompt, generate), on a NEW
dataset: real movie plots (data/movies.jsonl), not the IRS tax corpus. Encoding is
transformer-based (all-MiniLM-L6-v2) + FAISS, exactly as in class; generation goes
through the real llm.chat (with an extractive fallback if no provider is set up).
The embedding model downloads on first run, so this needs internet.

Run:  python solution.py
    pip install sentence-transformers faiss-cpu numpy python-dotenv
    # plus a provider, e.g.: pip install ollama   (or set GEMINI_API_KEY / GROQ_API_KEY)
"""

import json
import pathlib
import numpy as np
import llm

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "movies.jsonl"
ROWS = [json.loads(l) for l in open(DATA)]
PLOTS = [r["plot"] for r in ROWS]


# ---- 1. retriever: transformer embeddings + FAISS, with an optional year filter ----
def build_retriever():
    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = np.asarray(model.encode(PLOTS, normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model, index


def retrieve(model, index, query, k=2, year=None):
    qv = np.asarray(model.encode(query, normalize_embeddings=True), dtype="float32")[None, :]
    scores, idx = index.search(qv, len(ROWS))    # over-fetch, then filter
    hits = []
    for s, i in zip(scores[0], idx[0]):          # pair each row with its score
        r = ROWS[i]
        if year is not None and r["year"] != year:
            continue
        hits.append({**r, "score": float(s)})    # copy record, attach cosine score
        if len(hits) == k:
            break
    return hits


# ---- 2. build_prompt: numbered, sourced context + grounding rule ----
def build_prompt(question, hits):
    ctx = "\n".join(f"[{n}] {r['plot']}  (source: {r['title']} {r['year']})"
                    for n, r in enumerate(hits, 1))
    return (f"Context:\n{ctx}\n\nQuestion: {question}\n\n"
            'Answer using ONLY the context above. Cite the passages you use like [1]. '
            'If the context does not contain the answer, say so.')


# ---- 3. answer: retrieve, build prompt, generate (real model, extractive fallback) ----
def answer(model, index, question, k=2, year=None):
    hits = retrieve(model, index, question, k=k, year=year)
    prompt = build_prompt(question, hits)
    try:
        text = llm.chat([{"role": "user", "content": prompt}])       # REAL model call
    except Exception:
        text = f"(extractive fallback, no LLM) {hits[0]['plot']} [1]"
    return text, hits


if __name__ == "__main__":
    model, index = build_retriever()
    q = "In The Italian Job, how do the thieves get away?"

    # 4. Without a year filter, a REMAKE is ambiguous: the query matches both the
    #    1969 original and the 2003 remake, and you may get the wrong version.
    print("BEFORE (no year filter):")
    _, hits = answer(model, index, q, k=2)
    print("  retrieved:", [(r["id"], r["year"]) for r in hits])

    # 5. Filter to the version you mean; now the answer is grounded in that film.
    print("AFTER (year=2003):")
    text, hits = answer(model, index, q, k=2, year=2003)
    print("  retrieved:", [(r["id"], r["year"]) for r in hits])
    print("  answer:", text[:200])

    # Expected: BEFORE mixes both Italian Job entries (1969 and 2003); the year
    # filter keeps only 2003, so the answer is grounded in the remake. Retrieval,
    # not the model, decides which film you are answering about, exactly the lesson
    # from the class stale-document demo, here as a remake-version filter.
