"""Streamlit UI for the Research Paper RAG system.

Provides a web interface for:
- Choosing a query mode: Standard (gpt-4o-mini) or Deep Research (gpt-4o)
- Entering free-text queries about any topic
- Ingesting PDFs via direct upload or an arXiv search query (any subject)
- Displaying RAG-generated answers with active model badge
- Browsing titles of all papers currently stored in the FAISS index
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from langchain_core.documents import Document

from src.config import load_config
from src.crawlers import (
    fetch_arxiv_papers,
    fetch_local_pdfs,
    fetch_pubmed_papers,
    fetch_semantic_scholar_papers,
    fetch_wikipedia_articles,
    filter_relevant_docs,
)
from src.pipeline import run_pipeline
from src.rag_chain import QueryMode, create_rag_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"

_MODE_LABELS: dict[QueryMode, str] = {
    QueryMode.STANDARD: "Standard  (gpt-4o-mini)",
    QueryMode.RESEARCH: "Deep Research  (gpt-4o / Claude)",
}

_MODE_HELP: dict[QueryMode, str] = {
    QueryMode.STANDARD: (
        "Fast and cost-effective. Best for factual lookups, "
        "sample-size questions, and single-paper queries."
    ),
    QueryMode.RESEARCH: (
        "Uses gpt-4o (or Claude 3.5 Sonnet). Reserved for cross-paper synthesis, "
        "conflicting-result analysis, and hypothesis generation."
    ),
}


@st.cache_data(ttl=300, show_spinner=False)
def _get_ingested_papers(config_path: str) -> list[dict]:
    """Extract unique paper metadata from the persisted FAISS docstore.

    Reads only the serialised docstore (index.pkl) without loading the
    embedding model, deduplicates chunks by source path/ID, and returns
    one row per original paper.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        List of dicts with keys: title, domain, doi, date, source.
        Returns an empty list if no index exists yet.
    """
    config = load_config(config_path)
    index_path: str = config.get("vector_store", {}).get(
        "faiss_index_path", "data/faiss_index"
    )
    pkl_path = Path(index_path) / "index.pkl"

    if not pkl_path.exists():
        return []

    try:
        with open(pkl_path, "rb") as fh:
            docstore, _ = pickle.load(fh)  # tuple: (InMemoryDocstore, index_to_id)
    except Exception as exc:
        logger.warning("Could not read FAISS docstore: %s", exc)
        return []

    seen: set[str] = set()
    papers: list[dict] = []
    for doc in docstore._dict.values():
        meta = doc.metadata
        source: str = meta.get("source", "")
        # arxiv_id is unique per paper; title is the fallback for arXiv chunks
        # that lack it; source (file path) works for local PDFs.
        # Avoid using the literal string "arxiv" as a key — it is shared by
        # every arXiv chunk and would collapse all papers into one entry.
        dedup_key = (
            meta.get("arxiv_id")
            or meta.get("title")
            or (source if source != "arxiv" else None)
            or ""
        )
        if not dedup_key or dedup_key in seen:
            continue
        seen.add(dedup_key)
        papers.append(
            {
                "title": meta.get("title") or source or "Unknown",
                "domain": meta.get("domain", "—"),
                "doi": meta.get("doi", ""),
                "date": meta.get("date", ""),
                "source": source,
            }
        )

    papers.sort(key=lambda p: p["title"].lower())
    return papers


def _run_ingestion(
    query: str,
    uploaded_files: list,
    enrich_metadata: bool,
    relevance_threshold: float,
    use_arxiv: bool,
    use_pubmed: bool,
    use_semantic_scholar: bool,
    use_wikipedia: bool,
) -> None:
    """Fetch from all selected sources, filter for relevance, chunk, and index.

    Args:
        query: Search string sent to every enabled online source.
        uploaded_files: Streamlit UploadedFile objects saved to the local PDF dir.
        enrich_metadata: Run gpt-4o-mini metadata extraction before indexing.
        relevance_threshold: Cosine similarity cut-off applied to all online results.
        use_arxiv: Fetch from arXiv.
        use_pubmed: Fetch from PubMed.
        use_semantic_scholar: Fetch from Semantic Scholar.
        use_wikipedia: Fetch background articles from Wikipedia.
    """
    cfg = load_config(CONFIG_PATH)
    sources_cfg: dict = cfg.get("sources", {})
    pdf_dir: str = sources_cfg.get("local", {}).get("pdf_dir", "data/papers/")
    embedding_model: str = cfg.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    all_docs: list[Document] = []

    with st.status("Ingesting papers...", expanded=True) as status:

        # ── Uploaded PDFs ────────────────────────────────────────────────────
        if uploaded_files:
            save_dir = Path(pdf_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for uploaded_file in uploaded_files:
                dest = save_dir / uploaded_file.name
                dest.write_bytes(uploaded_file.getvalue())
            st.write(f"Saved **{len(uploaded_files)}** PDF(s) to `{pdf_dir}`.")

        try:
            local_docs = fetch_local_pdfs(pdf_dir)
            if local_docs:
                st.write(f"Loaded **{len(local_docs)}** local PDF(s).")
                all_docs.extend(local_docs)
        except FileNotFoundError:
            pass

        def _fetch_and_filter(label: str, fetcher, max_results: int) -> None:
            """Run one online fetcher, apply relevance filter, extend all_docs."""
            if not query.strip():
                return
            st.write(f"Fetching from **{label}**...")
            try:
                docs = fetcher(query, max_results=max_results)
                docs, n_dropped = filter_relevant_docs(
                    docs, query, embedding_model, threshold=relevance_threshold
                )
                kept_msg = (
                    f"{label}: kept **{len(docs)}** relevant result(s)"
                    + (f", dropped {n_dropped} off-topic." if n_dropped else ".")
                )
                if len(docs) == 0 and n_dropped > 0:
                    kept_msg += " ⬇ Try lowering the relevance threshold."
                st.write(kept_msg)
                all_docs.extend(docs)
            except Exception as exc:
                st.warning(f"{label} fetch failed: {exc}")

        # ── Online sources ─────────────────────────────────────────────────
        if use_arxiv:
            _fetch_and_filter(
                "arXiv", fetch_arxiv_papers,
                sources_cfg.get("arxiv", {}).get("max_results", 100),
            )
        if use_pubmed:
            _fetch_and_filter(
                "PubMed", fetch_pubmed_papers,
                sources_cfg.get("pubmed", {}).get("max_results", 50),
            )
        if use_semantic_scholar:
            _fetch_and_filter(
                "Semantic Scholar", fetch_semantic_scholar_papers,
                sources_cfg.get("semantic_scholar", {}).get("max_results", 50),
            )
        if use_wikipedia:
            _fetch_and_filter(
                "Wikipedia", fetch_wikipedia_articles,
                sources_cfg.get("wikipedia", {}).get("max_results", 3),
            )

        if not all_docs:
            st.warning("No documents found. Select at least one source or upload PDFs.")
            status.update(label="Ingestion failed", state="error")
            return

        label_txt = f"Processing **{len(all_docs)}** document(s)..."
        if enrich_metadata:
            label_txt += " Extracting metadata with gpt-4o-mini..."
        st.write(label_txt)

        run_pipeline(all_docs, CONFIG_PATH, enrich_metadata=enrich_metadata)
        status.update(label=f"Ingested {len(all_docs)} documents", state="complete")
        _get_ingested_papers.clear()


def _query_rag(question: str, mode: QueryMode) -> str:
    """Run a question through the RAG chain across all indexed content.

    Args:
        question: User's natural-language question.
        mode: QueryMode.STANDARD (gpt-4o-mini) or QueryMode.RESEARCH (gpt-4o).

    Returns:
        The LLM-generated answer string.

    Raises:
        FileNotFoundError: If no FAISS index has been built yet.
    """
    chain: Any = create_rag_pipeline(
        query_domain=None, mode=mode, config_path=CONFIG_PATH
    )
    return chain.invoke(question)


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Research Paper RAG",
        page_icon=":books:",
        layout="wide",
    )
    st.title("Research Paper RAG")
    st.caption("Hybrid search over your indexed documents — upload anything, ask anything")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        # Query mode selector
        st.subheader("Query mode")
        mode_choice: str = st.radio(
            "Select LLM tier",
            options=list(_MODE_LABELS.values()),
            index=0,
            help=(
                "Standard: gpt-4o-mini for factual queries (~95% of requests). "
                "Deep Research: gpt-4o / Claude for synthesis and hypothesis generation."
            ),
        )
        active_mode: QueryMode = (
            QueryMode.RESEARCH
            if mode_choice == _MODE_LABELS[QueryMode.RESEARCH]
            else QueryMode.STANDARD
        )
        st.caption(_MODE_HELP[active_mode])

        st.divider()

        # Ingestion controls
        st.subheader("Ingestion")
        search_query: str = st.text_input(
            "Search query",
            placeholder="e.g. medieval Islamic architecture, CRISPR, options pricing",
            help="Sent to every enabled source below.",
        )

        st.caption("Sources")
        col_a, col_b = st.columns(2)
        with col_a:
            use_arxiv: bool = st.checkbox("arXiv", value=True,
                help="Sciences, maths, CS, economics preprints.")
            use_pubmed: bool = st.checkbox("PubMed", value=False,
                help="Biomedical and life sciences abstracts.")
        with col_b:
            use_semantic_scholar: bool = st.checkbox("Semantic Scholar", value=False,
                help="Broadest coverage: CS, medicine, physics, law, economics.")
            use_wikipedia: bool = st.checkbox("Wikipedia", value=False,
                help="Background articles (3 per query). Good for non-technical context.")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            help="Upload your own PDF documents. They will be saved and indexed.",
        )
        enrich_meta: bool = st.checkbox(
            "Enrich metadata (gpt-4o-mini)",
            value=False,
            help=(
                "Run gpt-4o-mini over each fetched document to extract title, "
                "authors, year, keywords, and DOI. Improves filter precision "
                "but adds one API call per document."
            ),
        )
        relevance_threshold: float = st.slider(
            "Relevance filter threshold",
            min_value=0.10,
            max_value=0.55,
            value=0.30,
            step=0.05,
            help=(
                "Cosine similarity cut-off applied to all online results. "
                "0.30 (default) keeps most relevant results. "
                "Raise to 0.45–0.50 only for very focused queries. "
                "Values above 0.50 will drop most results."
            ),
        )
        if st.button("Run ingestion", type="primary"):
            _run_ingestion(
                query=search_query,
                uploaded_files=uploaded_files,
                enrich_metadata=enrich_meta,
                relevance_threshold=relevance_threshold,
                use_arxiv=use_arxiv,
                use_pubmed=use_pubmed,
                use_semantic_scholar=use_semantic_scholar,
                use_wikipedia=use_wikipedia,
            )

        st.divider()

        # ── Ingested Papers browser ───────────────────────────────────────
        st.subheader("Ingested Papers")
        with st.expander("View Ingested Papers"):
            col_refresh, col_count = st.columns([1, 2])
            with col_refresh:
                if st.button("Refresh", key="refresh_papers"):
                    _get_ingested_papers.clear()
            papers = _get_ingested_papers(CONFIG_PATH)
            if not papers:
                st.info("No papers indexed yet. Run ingestion first.")
            else:
                with col_count:
                    st.caption(f"{len(papers)} paper(s) in index")
                display_papers = papers[:100]
                for i, paper in enumerate(display_papers, 1):
                    line = f"{i}. **{paper['title']}**"
                    tags: list[str] = []
                    if paper["domain"] != "—":
                        tags.append(f"`{paper['domain']}`")
                    if paper.get("date"):
                        tags.append(paper["date"])
                    if paper.get("doi"):
                        tags.append(f"DOI: {paper['doi']}")
                    if tags:
                        line += "  \n" + " | ".join(tags)
                    st.markdown(line)
                if len(papers) > 100:
                    st.caption(f"… and {len(papers) - 100} more not shown.")

    # ── Main area ─────────────────────────────────────────────────────────────
    # Model badge
    if active_mode == QueryMode.RESEARCH:
        st.info(
            "**Deep Research mode** — answers generated by gpt-4o / Claude 3.5 Sonnet. "
            "Re-ranking still uses gpt-4o-mini.",
            icon="🔬",
        )
    else:
        st.success(
            "**Standard mode** — answers and re-ranking both use gpt-4o-mini.",
            icon="⚡",
        )

    question: str = st.text_input(
        "Ask a research question",
        placeholder=(
            "Standard: 'What was the sample size in the Smith et al. study?'  |  "
            "Deep Research: 'What are common themes across these diffusion model papers?'"
        ),
    )

    if question:
        with st.spinner("Retrieving, re-ranking, and generating answer..."):
            try:
                answer: str = _query_rag(question, active_mode)
                st.markdown("### Answer")
                st.write(answer)
            except FileNotFoundError:
                st.error(
                    "No FAISS index found. Upload PDFs or run an arXiv search first."
                )
            except EnvironmentError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
