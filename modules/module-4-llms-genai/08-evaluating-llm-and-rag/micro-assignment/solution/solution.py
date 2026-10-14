"""Micro-assignment 4.8 reference solution: evaluate the movie-plot assistant.

Same metrics as the class (faithfulness, answer relevance, context relevance via
LLM-as-judge, plus deterministic context recall), on a NEW dataset: real movie
plots (data/movies.jsonl), not the IRS corpus. The embedding model downloads on
first run; the judge and the answers are real llm.chat calls.

Run:  python solution.py
    pip install sentence-transformers faiss-cpu numpy python-dotenv
    # plus a provider, e.g.: pip install ollama
"""

import json
import pathlib
import re
import numpy as np
import llm

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "movies.jsonl"
ROWS = [json.loads(l) for l in open(DATA)]

# A small eval set: question, gold movie id (None = unanswerable), reference.
MOVIE_EVAL = [
    ("Which film is about hunting artificial humans in a rainy future city?", "blade-runner-1982", "Blade Runner (1982)."),
    ("A boy travels to the land of the dead to find his family.", "coco-2017", "Coco (2017)."),
    ("Scientists bring dinosaurs back for a park and it goes wrong.", "jurassic-park-1993", "Jurassic Park (1993)."),
    ("A crew robs three Las Vegas casinos in one night.", "oceans-eleven-2001", "Ocean's Eleven (2001)."),
    ("Thieves enter people's dreams to plant an idea.", "inception-2010", "Inception (2010)."),
    ("A gold heist using Mini Coopers, the modern remake.", "italian-job-2003", "The Italian Job (2003)."),
    ("A wrongly imprisoned banker slowly plans his escape.", "shawshank-1994", "The Shawshank Redemption (1994)."),
    ("A father fish searches the ocean for his lost son.", "finding-nemo-2003", "Finding Nemo (2003)."),
    ("Which film features a hobbit destroying a magic ring?", None, "Not found in these plots."),
]


# ---- 1. retriever + answer (the assistant under test) ----
def build_retriever():
    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = np.asarray(model.encode([r["plot"] for r in ROWS], normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model, index


def answer(model, index, question, k=3):
    qv = np.asarray(model.encode(question, normalize_embeddings=True), dtype="float32")[None, :]
    _, idx = index.search(qv, k)
    hits = [ROWS[i] for i in idx[0]]
    ctx = "\n".join(f"[{n}] {r['title']} ({r['year']}): {r['plot']}" for n, r in enumerate(hits, 1))
    prompt = (f"Context:\n{ctx}\n\nQuestion: {question}\n\n"
              "Name the film using ONLY the context. If it is not there, reply "
              'exactly: "I do not find that in these plots."')
    try:
        text = llm.chat([{"role": "user", "content": prompt}])
    except Exception:
        text = "I do not find that in these plots."
    return text, hits


# ---- 2. context recall: deterministic, no judge ----
def context_recall(gold_id, retrieved_ids):
    return float(gold_id in retrieved_ids)


# ---- 3. LLM-as-judge metrics (each returns a number in [0, 1]) ----
def _judge_number(instruction):
    reply = llm.chat([{"role": "user", "content": instruction +
                       "\n\nRespond with ONLY a number between 0 and 1."}])
    m = re.search(r"(\d*\.?\d+)", reply)
    return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0


def faithfulness(answer_text, contexts):
    ctx = "\n".join(contexts)
    return _judge_number(
        "Grade FAITHFULNESS: what fraction of the answer's claims are supported by "
        "the context? Return a decimal (1.0 = all supported; a correct 'not found' "
        f"is fully faithful).\n\nContext:\n{ctx}\n\nAnswer: {answer_text}")


def answer_relevance(question, answer_text):
    return _judge_number(
        "Grade ANSWER RELEVANCE: how directly does the answer address the question? "
        f"1.0 = fully on point, 0.0 = off-topic.\n\nQuestion: {question}\n\nAnswer: {answer_text}")


def context_relevance(question, contexts):
    listed = "\n".join(f"[{n}] {c}" for n, c in enumerate(contexts, 1))
    return _judge_number(
        "Grade CONTEXT RELEVANCE: what fraction of these passages are relevant to "
        f"the question? Return a decimal.\n\nQuestion: {question}\n\nPassages:\n{listed}")


# ---- 4. evaluate over the set ----
def evaluate(model, index, k=3):
    sums = {"faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0}
    rec_sum, rec_n = 0.0, 0
    for q, gold, _ref in MOVIE_EVAL:
        text, hits = answer(model, index, q, k=k)
        contexts = [f"{h['title']} ({h['year']}): {h['plot']}" for h in hits]
        sums["faithfulness"] += faithfulness(text, contexts)
        sums["answer_relevance"] += answer_relevance(q, text)
        sums["context_relevance"] += context_relevance(q, contexts)
        if gold is not None:
            rec_sum += context_recall(gold, [h["id"] for h in hits]); rec_n += 1
    n = len(MOVIE_EVAL)
    means = {m: s / n for m, s in sums.items()}
    means["context_recall"] = rec_sum / rec_n if rec_n else float("nan")
    return means


if __name__ == "__main__":
    model, index = build_retriever()
    means = evaluate(model, index)
    for m in ("faithfulness", "answer_relevance", "context_relevance", "context_recall"):
        print(f"  {m:18} {means[m]:.2f}")
    weakest = min(means, key=lambda m: means[m])
    print(f"\nWeakest: {weakest} ({means[weakest]:.2f}).")
    # 5. One fix, matched to the weak metric: if context_recall is lowest, the
    #    retriever missed the right plot, so raise k or add a re-ranker (class 4.6).
    #    If faithfulness is lowest, tighten the prompt to answer only from context.
    #    If answer_relevance is lowest, the prompt should force a direct film name.
