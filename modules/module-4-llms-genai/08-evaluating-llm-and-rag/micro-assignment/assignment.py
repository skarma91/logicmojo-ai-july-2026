"""Micro-assignment 4.8: evaluate the movie-plot assistant.

Same metrics as class (faithfulness, answer relevance, context relevance via
LLM-as-judge, plus deterministic context recall), on a NEW dataset: real movie
plots (data/movies.jsonl). See README.md. The embedding model downloads on first
run; the judge and answers are real llm.chat calls, so set up a provider.

    pip install sentence-transformers faiss-cpu numpy python-dotenv ollama
"""

import json
import pathlib
import re
import numpy as np
import llm

DATA = pathlib.Path(__file__).resolve().parent / "data" / "movies.jsonl"
ROWS = [json.loads(l) for l in open(DATA)]

# Given: a small eval set (question, gold movie id or None, reference answer).
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

# ---- 1. build_retriever() and answer(model, index, question, k): retrieve + generate ----
# Encode plots with all-MiniLM-L6-v2 + faiss.IndexFlatIP; answer only from context,
# else reply "I do not find that in these plots."
# your code here

# ---- 2. context_recall(gold_id, retrieved_ids): deterministic, 1 if gold retrieved ----
# your code here

# ---- 3. faithfulness / answer_relevance / context_relevance: LLM-as-judge, each 0..1 ----
# your code here

# ---- 4. evaluate(model, index): mean of each metric over MOVIE_EVAL (recall where gold exists) ----
# your code here

# ---- 5. Print the four means, name the weakest, and say one fix (reasoning) ----
if __name__ == "__main__":
    # your code here
    pass
