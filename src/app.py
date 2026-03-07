"""Streamlit UI for the Research Paper RAG system."""

import logging

import streamlit as st

from src.config import load_config
from src.rag_chain import create_rag_pipeline

logging.basicConfig(level=logging.INFO)
CONFIG_PATH = "config.yaml"


def main() -> None:
    st.set_page_config(page_title="Research Paper RAG", page_icon=":books:", layout="wide")
    st.title("Research Paper RAG")

    cfg = load_config(CONFIG_PATH)

    with st.sidebar:
        st.header("Configuration")
        domains = list(cfg.get("domains", {}).keys()) or ["all"]
        domain = st.selectbox("Domain", options=["all"] + domains, index=0)
        model = st.radio(
            "Model tier",
            options=["standard (gpt-4o-mini)", "research (gpt-4o)"],
            index=0,
        )

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
