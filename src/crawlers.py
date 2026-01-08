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
    """Fetch papers from arXiv for a given query string."""
    logger.info("Fetching %d papers from arXiv | query: %s", max_results, domain_query)
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
                    "published": str(result.published),
                },
            )
        )
    logger.info("Fetched %d documents from arXiv.", len(docs))
    return docs


def fetch_local_pdfs(pdf_dir: str, domain: Optional[str] = None) -> list[Document]:
    """Load raw PDF documents from a local directory."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    loader = DirectoryLoader(str(pdf_path), glob="**/*.pdf")
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", "local")
        if domain:
            doc.metadata.setdefault("domain", domain)
    return docs
