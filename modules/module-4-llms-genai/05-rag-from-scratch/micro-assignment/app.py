"""Micro-assignment 4.5, problem 6: a Streamlit UI over your movie RAG loop.

Import the functions you wrote in assignment.py and draw:
  1. a text box for the question,
  2. the grounded answer with its sources, and
  3. an expander that lists the retrieved plots with their titles, years, and
     similarity scores (this is what makes the remake ambiguity visible).

No new retrieval logic goes here; reuse build_retriever / retrieve / answer.

Run it:

    pip install streamlit sentence-transformers faiss-cpu numpy python-dotenv
    streamlit run app.py

A complete reference build is in solution/app.py.
"""

import streamlit as st

# import assignment            # your build_retriever / retrieve / answer live here

# ---- 1. Build and cache the index once (see @st.cache_resource) ----
# your code here

# ---- 2. Title, a text_input for the question, and a k slider / year filter ----
# your code here

# ---- 3. On a button click: call answer(), show the reply and its sources ----
# your code here

# ---- 4. An expander that shows each retrieved plot, its year, and its score ----
# your code here
