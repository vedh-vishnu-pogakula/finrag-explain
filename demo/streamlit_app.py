"""
Streamlit demo shell -- not wired to the real pipeline yet (nothing to wire to until
retrieval/attribution/grounding/faithfulness modules exist). This exists now so the
target UX is decided early: query -> retrieved chunks -> attribution weights ->
highlighted evidence -> answer -> faithfulness verdict (brief Section 15 deliverable).

Per CLAUDE.md / brief Section 14: cache model + index loading with st.cache_resource so
the demo doesn't reload the embedding model on every query -- that's already wired below
even though load_pipeline() is a stub, so you don't forget it later when it matters.

Run with: streamlit run demo/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Explainable Financial RAG", layout="wide")


@st.cache_resource
def load_pipeline():
    # TODO: load embedding model + FAISS index once here, not per-query.
    # from src.retrieval.retriever import Retriever
    # return Retriever.from_config("configs/config.yaml")
    return None


def main():
    st.title("Explainable Financial RAG")
    st.caption("Retrieval attribution + faithfulness-verified explanations for financial QA")

    pipeline = load_pipeline()
    if pipeline is None:
        st.info(
            "Pipeline not wired up yet -- this is a placeholder shell. "
            "Come back once src/retrieval/ and src/generation/ exist."
        )

    query = st.text_input("Ask a financial question (FinQA / TAT-QA style):")
    if query:
        st.write("Retrieved chunks, attribution weights, highlighted evidence, answer, "
                  "and faithfulness verdict will render here once the pipeline is wired up.")


if __name__ == "__main__":
    main()
