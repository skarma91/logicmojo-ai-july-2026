"""Class 4.5 build, the UI: a small Streamlit face over the RAG loop.

This adds NO new retrieval or generation logic. It imports the functions you
already wrote in main.py (build_retriever, retrieve, build_prompt, generate,
format_citation) and draws three things:

  1. a box to type a question,
  2. the grounded answer with its sources, and
  3. an expander that shows the retrieved chunks AND their similarity scores.

That third panel is the whole point. In main.py the "bad-chunk failure" was
something you read about; here it is on screen. When an answer looks wrong, open
the panel and see which passage (and which tax year) the retriever actually
pulled in. Retrieval quality decides answer quality, and now you can watch it.

Run it:

    pip install streamlit sentence-transformers faiss-cpu python-dotenv
    # plus a provider, e.g.:  pip install ollama   (and run `ollama serve`)
    streamlit run app.py

The provider (gemini, groq, or ollama) is read from the .env file at the project
root, exactly as in main.py. With no provider set up, generate() falls back to an
extractive answer, so the UI still works end to end.
"""

from __future__ import annotations
import logging

import streamlit as st

# Reuse the class build. Importing main.py runs only its top-level code (defining
# functions and the DATA path); main() is guarded by __main__, so nothing runs.
import main

# Send library logs to the console so you can watch retrieval and each model call
# in the terminal while you click around in the browser.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Build the retriever once and keep it. Embedding the corpus takes a few seconds
# and downloads a small model the first time, so we do NOT want to repeat it on
# every keystroke. @st.cache_resource stores the returned object across reruns
# (Streamlit reruns the whole script each time the user interacts).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading corpus and building the index...")
def get_retriever():
    corpus = main.load_corpus()
    return main.build_retriever(corpus), corpus


retriever, corpus = get_retriever()

# ---- Page header -----------------------------------------------------------
st.title("Domain Knowledge Assistant")
st.caption(
    f"Grounded question answering over {len(corpus)} passages from "
    f"{len({r['pub'] for r in corpus})} IRS publications. Answers cite their sources."
)

# ---- Controls: the question, and the two knobs from main.py ----------------
question = st.text_input(
    "Your question",
    value="What is the standard deduction for a single filer?",
)

col1, col2 = st.columns(2)
with col1:
    # Same top-k trade-off taught in class 4.4 (Embeddings and vector databases).
    k = st.slider("Passages to retrieve (k)", min_value=1, max_value=5, value=3)
with col2:
    # The metadata filter that fixes the stale-document failure. "All years" = no
    # filter (the failure case); pick 2025 to keep only the current figure.
    year_choice = st.selectbox("Tax year filter", ["All years", 2025, 2024])
tax_year = None if year_choice == "All years" else int(year_choice)

# ---- Run the loop when asked ----------------------------------------------
if st.button("Ask", type="primary") and question.strip():
    # One call into the class build: retrieve -> build_prompt -> generate.
    result = main.answer(retriever, question, k=k, tax_year=tax_year)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    # render_sources already maps [n] back to a citation and URL, one per line.
    st.text(result["sources"])

    # ---- The debug panel: what retrieval actually returned -----------------
    # Collapsed by default so the answer stays clean, but one click reveals the
    # evidence: each retrieved passage, its tax year, and its similarity score.
    with st.expander("Retrieved chunks and scores"):
        st.caption(
            "If the answer looks wrong, the cause is usually here: an off-topic "
            "or out-of-date passage that scored high enough to be retrieved."
        )
        for n, r in enumerate(result["retrieved"], 1):
            # score is the cosine similarity attached by retrieve() in main.py.
            st.markdown(
                f"**[{n}] {main.format_citation(r)}**  \n"
                f"score `{r['score']:.3f}`  ·  tax_year `{r['tax_year']}`"
            )
            # Show a short preview of the passage text, not the whole thing.
            st.write(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
            st.divider()

    # Teaching nudge: try the same question with "All years" vs 2025 and watch
    # the panel. With no filter, last year's passage can outrank this year's.
    if tax_year is None:
        st.info(
            "Tip: ask the standard-deduction question with the filter on 'All "
            "years', then switch to 2025. Open the panel to see the stale 2024 "
            "passage drop out. That is retrieval quality deciding the answer."
        )
