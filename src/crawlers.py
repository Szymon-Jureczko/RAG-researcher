"""Crawlers: fetch raw documents from arXiv and local PDF directories.

Chunking, embedding, and indexing live in pipeline.py.
"""

import logging
from pathlib import Path
from typing import Optional

import arxiv as arxiv_lib
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def fetch_arxiv_papers(
    domain_query: str,
    max_results: int = 100,
    domain_key: Optional[str] = None,
) -> list[Document]:
    """Fetch papers from arXiv for a given domain query string.

    Args:
        domain_query: arXiv category or search string (e.g. "cs.LG machine learning").
        max_results: Maximum number of papers to retrieve.
        domain_key: Optional short label stored as metadata.

    Returns:
        Document objects tagged with source/domain metadata.

    Raises:
        ValueError: If domain_query is empty.
    """
    if not domain_query.strip():
        raise ValueError("domain_query must not be empty.")

    logger.info("Fetching %d papers from arXiv | query: %s", max_results, domain_query)
    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(query=domain_query, max_results=max_results)
        docs = []
        for result in client.results(search):
            docs.append(
                Document(
                    page_content=result.summary,
                    metadata={
                        "source": "arxiv",
                        "domain": domain_key or domain_query,
                        "title": result.title,
                        "authors": ", ".join(str(a) for a in result.authors),
                        "published": str(result.published),
                        "arxiv_id": result.entry_id,
                    },
                )
            )
        logger.info("Fetched %d documents from arXiv.", len(docs))
        return docs
    except Exception as exc:
        logger.error("arXiv fetch failed for query '%s': %s", domain_query, exc)
        raise


def fetch_local_pdfs(pdf_dir: str, domain: Optional[str] = None) -> list[Document]:
    """Load raw PDF documents from a local directory using DirectoryLoader."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    logger.info("Loading PDFs from: %s", pdf_dir)
    loader = DirectoryLoader(str(pdf_path), glob="**/*.pdf", show_progress=True)
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", "local")
        if domain:
            doc.metadata.setdefault("domain", domain)
    return docs
