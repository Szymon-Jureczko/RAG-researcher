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
        with st.spinner("Retrieving and generating answer..."):
            try:
                chain = create_rag_pipeline(return_sources=True)
                result = chain(question) if callable(chain) else {"answer": chain.invoke(question), "sources": []}
                st.markdown("### Answer")
                st.write(result["answer"])
                if result.get("sources"):
                    with st.expander("Retrieved sources"):
                        for i, doc in enumerate(result["sources"], 1):
                            meta = doc.metadata
                            title = meta.get("title") or meta.get("source", "Unknown")
                            st.markdown(f"**{i}. {title}**")
                            st.caption(f"domain: {meta.get('domain', '-')}  |  source: {meta.get('source', '-')}")
                            st.text(doc.page_content[:400] + "...")
            except FileNotFoundError:
                st.error("No FAISS index found. Run the ingestion pipeline first.")


if __name__ == "__main__":
    main()
