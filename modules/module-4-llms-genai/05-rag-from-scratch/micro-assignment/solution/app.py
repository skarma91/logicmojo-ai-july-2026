"""Micro-assignment 4.5, reference UI: a Streamlit face over the movie RAG loop.

Problem 6. No new retrieval logic: this imports the functions from solution.py
(build_retriever, retrieve, build_prompt, answer) and draws a question box, the
grounded answer with sources, and an expander showing the retrieved plots and
their similarity scores. That panel is where the remake ambiguity becomes
visible: ask about "The Italian Job" with no year filter and watch both the 1969
and 2003 entries show up.

Run it:

    pip install streamlit sentence-transformers faiss-cpu numpy python-dotenv
    # plus a provider, e.g.:  pip install ollama   (and run `ollama serve`)
    streamlit run app.py
"""

from __future__ import annotations
import logging

import streamlit as st

# Reuse the solution build. Importing solution.py defines its functions; the demo
# under __main__ does not run on import.
import solution

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# Build the index once and cache it (embedding downloads a small model the first
# time and takes a few seconds; we do not want that on every keystroke).
@st.cache_resource(show_spinner="Encoding movie plots and building the index...")
def get_index():
    return solution.build_retriever()


model, index = get_index()

st.title("Movie Plot Assistant")
st.caption(f"Grounded answers over {len(solution.ROWS)} real movie plots, with sources.")

question = st.text_input(
    "Your question",
    value="In The Italian Job, how do the thieves get away?",
)

col1, col2 = st.columns(2)
with col1:
    k = st.slider("Plots to retrieve (k)", min_value=1, max_value=5, value=2)
with col2:
    # The year filter that resolves the remake ambiguity (1969 vs 2003).
    years = ["All years"] + sorted({r["year"] for r in solution.ROWS})
    year_choice = st.selectbox("Release year filter", years)
year = None if year_choice == "All years" else int(year_choice)

if st.button("Ask", type="primary") and question.strip():
    text, hits = solution.answer(model, index, question, k=k, year=year)

    st.subheader("Answer")
    st.write(text)

    st.subheader("Sources")
    st.text("\n".join(f"[{n}] {r['title']} ({r['year']})" for n, r in enumerate(hits, 1)))

    with st.expander("Retrieved plots and scores"):
        st.caption("If the wrong film answered, the reason is here: check which "
                   "titles and years were retrieved, and their scores.")
        for n, r in enumerate(hits, 1):
            st.markdown(f"**[{n}] {r['title']} ({r['year']})**  ·  score `{r['score']:.3f}`")
            st.write(r["plot"][:300] + ("..." if len(r["plot"]) > 300 else ""))
            st.divider()

    if year is None and "italian job" in question.lower():
        st.info("Ask again with the year set to 2003, then reopen the panel: only "
                "the remake remains, so the answer is grounded in the film you mean.")
