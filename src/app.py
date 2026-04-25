"""Streamlit UI for the Research Paper RAG system."""

import logging

import streamlit as st

from src.config import load_config
from pathlib import Path

from src.pipeline import run_pipeline
from src.crawlers import fetch_local_pdfs
from src.rag_chain import QueryMode, create_rag_pipeline

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
        mode_choice = st.radio(
            "Query mode",
            options=["Standard (gpt-4o-mini)", "Deep Research (gpt-4o / Claude)"],
            index=0,
        )
        active_mode = QueryMode.RESEARCH if mode_choice.startswith("Deep") else QueryMode.STANDARD

        st.divider()
        st.subheader("Ingest PDFs")
        pdf_dir = cfg.get("sources", {}).get("local", {}).get("pdf_dir", "data/papers/")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True
        )
        if st.button("Run ingestion") and uploaded_files:
            save_dir = Path(pdf_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for uf in uploaded_files:
                (save_dir / uf.name).write_bytes(uf.getvalue())
            with st.spinner("Indexing..."):
                docs = fetch_local_pdfs(pdf_dir)
                run_pipeline(docs, CONFIG_PATH)
            st.success(f"Ingested {len(uploaded_files)} PDF(s)")

    question = st.text_input("Ask a research question")
    if question:
        with st.spinner("Retrieving and generating answer..."):
            try:
                chain = create_rag_pipeline(mode=active_mode, return_sources=True)
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
                st.error("No FAISS index found. Upload PDFs and run ingestion first.")


if __name__ == "__main__":
    main()
