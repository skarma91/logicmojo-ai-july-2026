"""Class 4.8 build: evaluate the RAG assistant with hand-built metrics.

We reuse the class 4.5 (RAG from scratch) assistant over the real IRS corpus,
run it on a 30-item eval set (data/eval_set.jsonl), and score four metrics:

  * faithfulness      - is every claim in the answer supported by the context?
  * answer relevance  - does the answer actually address the question?
  * context relevance - of the passages retrieved, how many are on-topic?
  * context recall    - was the gold passage retrieved? (deterministic, no judge)

The first three are hand-built LLM-as-judge calls through llm.py, so you can see
exactly what each metric checks. RAGAS (the standard library) computes the same
ideas; a reference block at the bottom of this file shows how, and why we keep it
out of the core (its pinned dependencies often clash with current LangChain).

Run:
    pip install sentence-transformers faiss-cpu numpy python-dotenv
    # plus a provider for the judge and the answers, e.g.: pip install ollama
    python main.py

NOTE: the judge is a real model, so exact scores vary run to run. The point is
the method: measure each axis, then fix the weakest. Use a capable judge model
(a small local model is a weak grader); this is where a hosted or frontier model
in .env pays off.
"""

from __future__ import annotations
import json
import logging
import pathlib
import re

import llm

DATA = pathlib.Path(__file__).parent / "data"
log = logging.getLogger("course.eval")


# ===========================================================================
# The assistant under test: retrieve, then generate (reused from class 4.5)
# ===========================================================================
def load_corpus():
    return [json.loads(l) for l in (DATA / "corpus.jsonl").open()]


def build_retriever(corpus):
    """Embed and index the corpus, loading a cached embedding if present."""
    import numpy as np, faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_path, ids_path = DATA / "embeddings.npy", DATA / "ids.json"
    ids_now = [r["id"] for r in corpus]
    vectors = None
    if emb_path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids_now:
        vectors = np.load(emb_path).astype("float32")
        log.info("loaded %d cached embeddings", len(vectors))
    if vectors is None:
        vectors = np.array(model.encode([r["text"] for r in corpus], normalize_embeddings=True), dtype="float32")
        np.save(emb_path, vectors); ids_path.write_text(json.dumps(ids_now))
        log.info("embedded and cached %d passages", len(vectors))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return {"model": model, "index": index, "records": corpus}


def retrieve(retriever, query, k=3):
    import numpy as np
    qv = np.array(retriever["model"].encode(query, normalize_embeddings=True), dtype="float32")[None, :]
    scores, ids = retriever["index"].search(qv, k)
    return [retriever["records"][i] for i in ids[0]]


def answer(retriever, question, k=3):
    hits = retrieve(retriever, question, k=k)
    ctx = "\n".join(f"[{n}] {r['text']}" for n, r in enumerate(hits, 1))
    prompt = (f"Context:\n{ctx}\n\nQuestion: {question}\n\n"
              "Answer using ONLY the context above. If the context does not "
              'contain the answer, reply exactly: "I do not find that in the documents."')
    try:
        text = llm.chat([{"role": "user", "content": prompt}])
    except Exception:
        text = "I do not find that in the documents."   # no provider: safe refusal
    return text, hits


# ===========================================================================
# Judges: small LLM-as-judge calls that each return a number in [0, 1]
# ===========================================================================
def _judge_number(instruction: str) -> float:
    """Ask the model to return a single number in [0, 1] and parse it."""
    reply = llm.chat([{"role": "user", "content": instruction +
                       "\n\nRespond with ONLY a number between 0 and 1."}])
    m = re.search(r"(\d*\.?\d+)", reply)
    if not m:
        return 0.0
    return max(0.0, min(1.0, float(m.group(1))))


def faithfulness(answer_text: str, contexts: list[str]) -> float:
    """Fraction of the answer's factual claims that the context supports.

    A fuller version lists each claim and checks it one by one; here we let the
    judge do that decomposition and return the resulting fraction."""
    ctx = "\n".join(contexts)
    return _judge_number(
        "You are grading FAITHFULNESS. Split the answer into its factual claims "
        "and decide how many are supported by the context. Return supported/total "
        "as a decimal (1.0 = every claim supported). A correct 'I do not find "
        f"that' answer is fully faithful.\n\nContext:\n{ctx}\n\nAnswer: {answer_text}")


def answer_relevance(question: str, answer_text: str) -> float:
    """How directly the answer addresses the question (0 = off-topic, 1 = fully)."""
    return _judge_number(
        "You are grading ANSWER RELEVANCE: how directly the answer addresses the "
        "question, ignoring whether it is factually right. 1.0 = fully on point, "
        "0.0 = off-topic or evasive. A correct 'I do not find that' answer to an "
        f"unanswerable question is relevant.\n\nQuestion: {question}\n\nAnswer: {answer_text}")


def context_relevance(question: str, contexts: list[str]) -> float:
    """Fraction of retrieved passages that are on-topic for the question."""
    listed = "\n".join(f"[{n}] {c}" for n, c in enumerate(contexts, 1))
    return _judge_number(
        "You are grading CONTEXT RELEVANCE: of the retrieved passages below, what "
        "fraction are relevant to the question? Return relevant/total as a decimal."
        f"\n\nQuestion: {question}\n\nPassages:\n{listed}")


def context_recall(gold_id, retrieved_ids) -> float:
    """Deterministic: 1 if the gold passage was retrieved, else 0. No judge needed."""
    return float(gold_id in retrieved_ids)


# ===========================================================================
# Run the eval set and report the four metrics
# ===========================================================================
def evaluate(retriever, eval_set, k=3):
    sums = {"faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0}
    recall_sum, recall_n = 0.0, 0
    for item in eval_set:
        q = item["question"]
        text, hits = answer(retriever, q, k=k)
        contexts = [h["text"] for h in hits]
        ids = [h["id"] for h in hits]
        sums["faithfulness"] += faithfulness(text, contexts)
        sums["answer_relevance"] += answer_relevance(q, text)
        sums["context_relevance"] += context_relevance(q, contexts)
        if item["gold_id"] is not None:            # recall only where a gold exists
            recall_sum += context_recall(item["gold_id"], ids)
            recall_n += 1
        log.info("scored %s", item["id"])
    n = len(eval_set)
    means = {m: s / n for m, s in sums.items()}
    means["context_recall"] = recall_sum / recall_n if recall_n else float("nan")
    return means


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    corpus = load_corpus()
    eval_set = [json.loads(l) for l in (DATA / "eval_set.jsonl").open()]
    retriever = build_retriever(corpus)
    print(f"evaluating on {len(eval_set)} items over {len(corpus)} IRS passages\n")

    means = evaluate(retriever, eval_set)
    for metric in ("faithfulness", "answer_relevance", "context_relevance", "context_recall"):
        print(f"  {metric:18} {means[metric]:.2f}")

    weakest = min(means, key=lambda m: means[m])
    print(f"\nWeakest metric: {weakest} ({means[weakest]:.2f}). Fix that first: low "
          "recall points at the retriever; low faithfulness at the prompt or model.")


if __name__ == "__main__":
    main()


# ===========================================================================
# REFERENCE ONLY: the same metrics with RAGAS (not run here)
# ---------------------------------------------------------------------------
# RAGAS packages faithfulness, answer relevance, context precision, and context
# recall behind one call. Sketch:
#
#     pip install ragas datasets
#     from ragas import evaluate as ragas_evaluate
#     from ragas.metrics import faithfulness, answer_relevancy, \
#         context_precision, context_recall
#     from datasets import Dataset
#     ds = Dataset.from_dict({
#         "question":     [...],   # each eval question
#         "answer":       [...],   # the assistant's answer
#         "contexts":     [[...]], # the retrieved passages per question
#         "ground_truth": [...],   # the reference answer
#     })
#     result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy,
#                                          context_precision, context_recall])
#
# PITFALL: RAGAS pins particular LangChain and related versions that frequently
# clash with a current LangChain or LangGraph install. Run it in an isolated
# virtual environment, or expect to resolve a dependency conflict. That is why we
# hand-build the metrics above for teaching, and keep RAGAS as a reference.
# ===========================================================================
