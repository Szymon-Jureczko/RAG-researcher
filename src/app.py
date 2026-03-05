"""Streamlit UI for the Research Paper RAG system."""

import logging

import streamlit as st

from src.rag_chain import create_rag_pipeline

logging.basicConfig(level=logging.INFO)


def main() -> None:
    st.set_page_config(page_title="Research Paper RAG", page_icon=":books:", layout="wide")
    st.title("Research Paper RAG")
    st.caption("Hybrid search over your indexed research papers")

    question = st.text_input("Ask a research question")
    if question:
        with st.spinner("Generating answer..."):
            try:
                chain = create_rag_pipeline()
                answer = chain.invoke(question)
                st.markdown("### Answer")
                st.write(answer)
            except FileNotFoundError:
                st.error("No FAISS index found. Run the ingestion pipeline first.")


if __name__ == "__main__":
    main()
