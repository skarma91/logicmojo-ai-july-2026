"""Class 4.5 build: the Domain Knowledge Assistant, hand-built RAG.

A grounded question-answering assistant over real IRS tax publications (public
domain). It retrieves the relevant passages, builds a cited prompt, and answers
using only those passages, citing the publication, tax year, and page.

Run it with:

    pip install sentence-transformers faiss-cpu python-dotenv
    # then a provider SDK, e.g.:  pip install ollama   (and run `ollama serve`)
    python main.py

The corpus ships in data/corpus.jsonl (built by data/make_corpus.py; run
data/fetch_corpus.py to rebuild the full set from the original PDFs). Retrieval
reuses the ideas from class 4.4 (Embeddings and vector databases). Generation
goes through llm.chat (see llm.py), which selects the provider (gemini, groq, or
ollama) from the PROVIDER setting in a .env file at the project root; ollama is
the free local default. If no provider is reachable, it falls back to an
extractive answer (the top retrieved passage) so the pipeline still runs end to
end and you can see retrieval and citations working.

The pure functions (load_corpus, build_prompt, render_sources, format_citation)
need no model and are unit-testable on their own.
"""

from __future__ import annotations
import json
import logging
import pathlib
import textwrap

DATA = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"

# Our own logger for the RAG steps. Every model call is already logged inside
# llm.py; here we also log retrieval so the whole pipeline is observable.
log = logging.getLogger("course.rag")


# ===========================================================================
# Pure functions (no model needed): loading, citation and prompt building
# ===========================================================================
def load_corpus(path=DATA) -> list[dict]:
    """Load the real IRS passages with their metadata."""
    return [json.loads(line) for line in open(path)]


def format_citation(rec: dict) -> str:
    """A human-readable source label for one passage.

    Includes the page when we have one. The Docling corpus builder exports whole
    documents (page = 0), so we omit the page in that case.
    """
    page = rec.get("page")
    return f"{rec['pub']} ({rec['tax_year']}), p{page}" if page else f"{rec['pub']} ({rec['tax_year']})"


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """Assemble the augmented prompt: numbered, sourced context + a grounding rule.

    This is the "augment" step of RAG: we paste the retrieved passages into the
    prompt, number them [1], [2], ..., and tell the model to answer only from them.
    """
    lines = ["Context:"]
    # Number each passage and tag it with its source, so the model can cite [n].
    for n, r in enumerate(retrieved, 1):
        lines.append(f"[{n}] {r['text']}  (source: {format_citation(r)})")
    lines += [
        "",
        f"Question: {question}",
        "",
        "Answer using ONLY the context above. Cite the passages you use like [1]. "
        'If the context does not contain the answer, reply exactly: '
        '"I do not find that in the documents."',
    ]
    return "\n".join(lines)


def render_sources(retrieved: list[dict]) -> str:
    """Map the [n] citations back to real sources for the reader."""
    return "\n".join(f"[{n}] {format_citation(r)}  {r['source_url']}"
                     for n, r in enumerate(retrieved, 1))


# ===========================================================================
# Retrieval (needs the embedding model; built lazily so importing this module
# does not trigger a download). Reuses the class 4.4 retriever design.
# ===========================================================================
def build_retriever(corpus: list[dict]):
    """Embed and index the corpus. Returns a retriever dict.

    Embedding every passage is the slow step, so we cache it. If
    data/precompute_embeddings.py has already been run, data/embeddings.npy and
    data/ids.json exist; we load those and skip re-embedding. This is the same
    whether corpus.jsonl is the shipped sample or was rebuilt from the real PDFs
    by fetch_corpus.py, so the workflow is identical in both cases. If the cache
    is missing or does not match the current corpus, we embed now and write it,
    so the next run is instant.
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")   # needed to encode queries too
    data_dir = DATA.parent                            # the code/data folder
    emb_path, ids_path = data_dir / "embeddings.npy", data_dir / "ids.json"
    ids_now = [r["id"] for r in corpus]

    vectors = None
    # Use the cache only if it lines up with the corpus we are about to serve.
    if emb_path.exists() and ids_path.exists():
        if json.loads(ids_path.read_text()) == ids_now:
            vectors = np.load(emb_path).astype("float32")
            log.info("loaded %d cached embeddings from %s", len(vectors), emb_path.name)
        else:
            log.info("cache ids do not match corpus.jsonl; re-embedding")
    if vectors is None:
        # No usable cache: embed now, then save it so later runs are instant.
        vectors = np.array(
            model.encode([r["text"] for r in corpus], normalize_embeddings=True),
            dtype="float32",
        )
        np.save(emb_path, vectors)
        ids_path.write_text(json.dumps(ids_now))
        log.info("embedded %d passages and cached to %s", len(vectors), emb_path.name)

    # Inner-product index; on normalized vectors that equals cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])       # shape[1] = vector length
    index.add(vectors)
    # Keep the model, the index, and the original records together so retrieve()
    # can map a search hit back to its text and metadata.
    return {"model": model, "index": index, "records": corpus}


def retrieve(retriever, query: str, k: int = 3, tax_year: int | None = None) -> list[dict]:
    """Return the top-k records, optionally filtered to one tax year."""
    import numpy as np

    # Encode the query the same way, and reshape to a 1-row matrix ([None, :]).
    qv = np.array(retriever["model"].encode(query, normalize_embeddings=True), dtype="float32")[None, :]
    pool = len(retriever["records"])                     # fetch everything, then filter
    scores, ids = retriever["index"].search(qv, pool)   # ranked by similarity
    hits = []
    for s, i in zip(scores[0], ids[0]):                 # [0]: we searched one query
        rec = retriever["records"][i]                   # row index -> original record
        if tax_year is not None and rec["tax_year"] != tax_year:
            continue                                    # skip rows failing the filter
        hits.append({**rec, "score": float(s)})         # copy record, add its score
        if len(hits) == k:                              # stop at k kept results
            break
    # Log the retrieval so the pipeline is observable (which chunks, which filter).
    log.info("retrieved %d passages (tax_year=%s) ids=%s",
             len(hits), tax_year, [h["id"] for h in hits])
    return hits


# ===========================================================================
# Generation: real LLM if available, else an extractive fallback
# ===========================================================================
def generate(prompt: str) -> str:
    # This is the "generate" step: hand the augmented prompt to the model.
    import llm  # the shared provider helper (gemini, groq, or ollama via .env)
    try:
        # Send the whole prompt as one user message and return the model's reply.
        return llm.chat([{"role": "user", "content": prompt}])
    except Exception:
        # No provider reachable (no key, offline, server down). So we can still show
        # retrieval working, pull passage [1] back out of the prompt and return it
        # verbatim. This is "extractive": it quotes a passage rather than writing one.
        first = prompt.split("[1] ", 1)[-1].split("  (source:", 1)[0]
        return f"(extractive fallback, no LLM available) {first} [1]"


def answer(retriever, question: str, k: int = 3, tax_year: int | None = None) -> dict:
    retrieved = retrieve(retriever, question, k=k, tax_year=tax_year)
    prompt = build_prompt(question, retrieved)
    return {"answer": generate(prompt), "sources": render_sources(retrieved), "retrieved": retrieved}


# ===========================================================================
# Demo: the stale-document failure, then the fix
# ===========================================================================
def main():
    # Configure logging once for the program so llm.py's per-call logs and our
    # retrieval logs both appear at INFO.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    corpus = load_corpus()
    print(f"loaded {len(corpus)} passages from {len({r['pub'] for r in corpus})} IRS publications\n")
    retriever = build_retriever(corpus)
    q = "What is the standard deduction for a single filer?"

    print("=" * 72)
    print("BEFORE (no tax-year filter): last year's figure can be retrieved")
    before = answer(retriever, q, k=2)
    for r in before["retrieved"]:
        print(f"  retrieved {r['id']}  tax_year={r['tax_year']}  score={r['score']:.3f}")
    print(textwrap.indent(before["answer"], "  "))
    print("  NOTE: the 2024 passage says 14,600; that is last year's amount.")

    print("=" * 72)
    print("AFTER (filter to tax_year=2025): only the current figure remains")
    after = answer(retriever, q, k=2, tax_year=2025)
    for r in after["retrieved"]:
        print(f"  retrieved {r['id']}  tax_year={r['tax_year']}  score={r['score']:.3f}")
    print(textwrap.indent(after["answer"], "  "))
    print("  Sources:\n" + textwrap.indent(after["sources"], "    "))
    print("\nRetrieval quality decides answer quality: the metadata filter is what")
    print("keeps the model from citing an out-of-date number.")


if __name__ == "__main__":
    main()
