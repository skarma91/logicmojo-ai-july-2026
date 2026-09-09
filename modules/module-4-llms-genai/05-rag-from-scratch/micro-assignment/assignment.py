"""Micro-assignment 4.5: a RAG loop over movie plots.

New dataset (data/movies.jsonl), transformer embeddings (all-MiniLM-L6-v2) + FAISS,
real generation via llm.chat. See README.md for the five problems. The embedding
model downloads on first run (needs internet); generation needs a provider (Ollama
local, or a Groq/Gemini key in .env), otherwise the code falls back to extractive.

    pip install sentence-transformers faiss-cpu numpy python-dotenv
"""

import json
import pathlib
import numpy as np
import llm

# ---- Given: the movie corpus ----
DATA = pathlib.Path(__file__).resolve().parent / "data" / "movies.jsonl"
ROWS = [json.loads(l) for l in open(DATA)]
PLOTS = [r["plot"] for r in ROWS]
# Each record has: id, title, year, genre, director, source_url, plot.

# ---- 1. build_retriever() and retrieve(model, index, query, k, year=None) ----
# Encode PLOTS with all-MiniLM-L6-v2, index with faiss.IndexFlatIP; retrieve
# over-fetches then keeps the given year (if any).
# your code here

# ---- 2. build_prompt(question, hits): numbered, sourced context + grounding rule ----
# your code here

# ---- 3. answer(model, index, question, k, year=None): retrieve, prompt, llm.chat ----
# (fall back to an extractive answer if no provider is available)
# your code here

# ---- 4 and 5. Show the remake-version problem, then fix it with a year filter ----
# Ask "In The Italian Job, how do the thieves get away?" with k=2:
#   - no filter: the 1969 and 2003 versions are both candidates (ambiguous)
#   - year=2003: only the remake, so the answer is grounded in the version you mean
if __name__ == "__main__":
    # your code here
    pass
