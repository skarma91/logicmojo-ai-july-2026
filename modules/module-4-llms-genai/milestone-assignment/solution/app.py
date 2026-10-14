"""Module 4 milestone (reference): a thin Streamlit UI over the same assistant.

Reuses assistant.py, it does not re-implement anything. Shows the grounded answer
and the sources it cited, so the RAG plus fine-tuning result is tangible.

Run:
    pip install streamlit
    streamlit run app.py
"""

from __future__ import annotations
import streamlit as st

import llm
from assistant import build_store, load_corpus, answer, DISCLAIMER

st.set_page_config(page_title="Health Information Assistant", page_icon=None)
st.title("Health Information Assistant")
st.caption("RAG supplies the facts and citations; your model supplies the answer style.")
st.info(DISCLAIMER)


@st.cache_resource
def _store():
    return build_store(load_corpus())


@st.cache_resource
def _model(model_name: str):
    # Empty string means "use the .env default"; otherwise override MODEL_NAME.
    return llm.get_chat_model(model=model_name or None)


with st.sidebar:
    st.subheader("Generator")
    model_name = st.text_input("MODEL_NAME override", value="",
                               help="Leave blank for the .env default. Use your tuned model, e.g. dka-assistant.")
    k = st.slider("Passages to retrieve (k)", 1, 6, 3)
    rerank = st.checkbox("Cross-encoder rerank", value=True)

question = st.text_input("Ask a question", value="What is a normal blood pressure reading?")
if st.button("Answer") and question.strip():
    with st.spinner("Retrieving and generating..."):
        text, sources = answer(_store(), question, model=_model(model_name), k=k, rerank=rerank)
    st.markdown("### Answer")
    st.write(text)
    st.markdown("### Sources")
    for s in sources:
        st.markdown(f"- {s}")
