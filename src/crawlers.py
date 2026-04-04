"""Crawlers module: responsible ONLY for fetching and raw parsing of research papers.

Handles arXiv, PubMed, Semantic Scholar, Wikipedia, and local PDF directories.
All chunking, embedding, and indexing logic belongs in pipeline.py.
"""

import logging
from pathlib import Path
from typing import Optional

import arxiv as arxiv_lib
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def filter_relevant_docs(
    docs: list[Document],
    query: str,
    embedding_model: str,
    threshold: float = 0.35,
) -> tuple[list[Document], int]:
    """Drop documents whose content is semantically dissimilar to the query.

    Embeds the query and each document's representative text (title + first 400
    characters of content), then computes cosine similarity.  Documents below
    ``threshold`` are discarded before indexing so irrelevant arXiv keyword
    matches never enter the FAISS index.

    Uses the same HuggingFace model already configured in config.yaml so no
    extra API calls are needed.

    Args:
        docs: Candidate documents returned by ``fetch_arxiv_papers``.
        query: The original user search string — used as the relevance signal.
        embedding_model: HuggingFace model name (e.g
            ``"sentence-transformers/all-MiniLM-L6-v2"``).
        threshold: Cosine similarity cut-off in [0, 1]. Default 0.35 removes
            clear mismatches while retaining loosely related work.  Raise to
            0.45–0.50 for stricter filtering.

    Returns:
        Tuple of (filtered_docs, n_dropped) so callers can report to the user.
    """
    if not docs:
        return docs, 0

    embedder = HuggingFaceEmbeddings(model_name=embedding_model)

    # Representative text: title (if present) + start of abstract
    def _doc_text(doc: Document) -> str:
        title = doc.metadata.get("title", "")
        body = doc.page_content[:400]
        return f"{title}. {body}" if title else body

    query_vec = np.array(embedder.embed_query(query))
    doc_texts = [_doc_text(d) for d in docs]
    doc_vecs = np.array(embedder.embed_documents(doc_texts))

    # Cosine similarity: dot product of unit vectors
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    similarities: np.ndarray = doc_norms @ query_norm

    filtered = [doc for doc, sim in zip(docs, similarities) if sim >= threshold]
    n_dropped = len(docs) - len(filtered)
    logger.info(
        "Relevance filter | query='%s' | kept=%d | dropped=%d | threshold=%.2f",
        query, len(filtered), n_dropped, threshold,
    )
    return filtered, n_dropped


def fetch_arxiv_papers(
    domain_query: str,
    max_results: int = 100,
    domain_key: Optional[str] = None,
) -> list[Document]:
    """Fetch papers from arXiv for a given domain query string.

    Args:
        domain_query: arXiv category or search query (e.g. "cs.LG machine learning").
        max_results: Maximum number of papers to retrieve.
        domain_key: Optional short domain label to store in document metadata.

    Returns:
        List of Document objects, each tagged with `source` and `domain` metadata.

    Raises:
        ValueError: If domain_query is an empty string.
        Exception: If the arXiv API call fails unexpectedly.
    """
    if not domain_query.strip():
        raise ValueError("domain_query must not be empty.")

    logger.info("Fetching %d papers from arXiv | query: %s", max_results, domain_query)
    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(query=domain_query, max_results=max_results)
        docs = []
        for result in client.results(search):
            doc = Document(
                page_content=result.summary,
                metadata={
                    "source": "arxiv",
                    "domain": domain_key or domain_query,
                    "title": result.title,
                    "authors": ", ".join(str(a) for a in result.authors),
                    "published": str(result.published),
                    "arxiv_id": result.entry_id,
                    "doi": result.doi or "",
                },
            )
            docs.append(doc)
        logger.info("Fetched %d documents from arXiv.", len(docs))
        return docs
    except Exception as exc:
        logger.error("arXiv fetch failed for query '%s': %s", domain_query, exc)
        raise


def fetch_local_pdfs(
    pdf_dir: str,
    domain: Optional[str] = None,
) -> list[Document]:
    """Load raw PDF documents from a local directory using DirectoryLoader.

    Args:
        pdf_dir: Path to the directory containing PDF files (searched recursively).
        domain: Optional domain label to attach to each document's metadata.

    Returns:
        List of Document objects with `source` and optional `domain` metadata.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        Exception: If DirectoryLoader encounters an error during loading.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    logger.info("Loading PDFs from: %s", pdf_dir)
    try:
        loader = DirectoryLoader(str(pdf_path), glob="**/*.pdf", show_progress=True)
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", "local")
            if domain:
                doc.metadata.setdefault("domain", domain)
        logger.info("Loaded %d documents from local directory.", len(docs))
        return docs
    except Exception as exc:
        logger.error("Local PDF load failed for '%s': %s", pdf_dir, exc)
        raise


# ── PubMed ────────────────────────────────────────────────────────────────────

def fetch_pubmed_papers(
    query: str,
    max_results: int = 50,
) -> list[Document]:
    """Fetch biomedical abstracts from PubMed via LangChain's PubMedLoader.

    Covers medicine, life sciences, pharmacology, and related fields.
    No API key required. Results are the full abstract text with PMID metadata.

    Args:
        query: PubMed search string (same syntax as pubmed.ncbi.nlm.nih.gov).
        max_results: Maximum number of papers to retrieve (PubMed cap: 10 000).

    Returns:
        List of Document objects tagged with ``source: "pubmed"``.

    Raises:
        ImportError: If the ``xmltodict`` package is not installed.
        Exception: If the PubMed API call fails.
    """
    try:
        from langchain_community.document_loaders import PubMedLoader
    except ImportError as exc:
        raise ImportError(
            "PubMed support requires xmltodict: pip install xmltodict"
        ) from exc

    logger.info("Fetching %d papers from PubMed | query: %s", max_results, query)
    try:
        loader = PubMedLoader(query, load_max_docs=max_results)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = "pubmed"
        logger.info("Fetched %d documents from PubMed.", len(docs))
        return docs
    except Exception as exc:
        logger.error("PubMed fetch failed for query '%s': %s", query, exc)
        raise


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def fetch_semantic_scholar_papers(
    query: str,
    max_results: int = 50,
) -> list[Document]:
    """Fetch paper abstracts from the Semantic Scholar public API.

    Covers computer science, medicine, physics, economics, law, and more —
    the broadest academic source available.  No API key is required for up to
    100 requests per 5 minutes.  Set SEMANTIC_SCHOLAR_API_KEY in .env to raise
    the limit to 1 request/second.

    Automatically retries on HTTP 429 (rate-limit) responses with exponential
    backoff (up to 3 attempts) so transient throttling does not fail silently.

    Args:
        query: Free-text search string.
        max_results: Number of results to request (API hard cap: 100 per call).

    Returns:
        List of Document objects with title, authors, year, DOI metadata,
        tagged with ``source: "semantic_scholar"``.

    Raises:
        RuntimeError: If all retry attempts are exhausted due to rate-limiting.
        requests.HTTPError: If the API returns a non-429 error status.
    """
    import os
    import time
    import requests as req

    logger.info(
        "Fetching %d papers from Semantic Scholar | query: %s", max_results, query
    )

    headers: dict[str, str] = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,abstract,authors,year,externalIds,publicationDate",
    }

    _max_retries = 3
    _backoff = 5  # seconds; doubles on each retry

    response: req.Response | None = None
    for attempt in range(_max_retries):
        try:
            response = req.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                headers=headers,
                timeout=30,
            )
        except req.RequestException as exc:
            logger.error("Semantic Scholar network error (attempt %d): %s", attempt + 1, exc)
            raise

        if response.status_code == 429:
            wait = _backoff * (2 ** attempt)
            logger.warning(
                "Semantic Scholar rate-limited (429). Waiting %ds before retry %d/%d.",
                wait, attempt + 1, _max_retries,
            )
            time.sleep(wait)
            continue

        try:
            response.raise_for_status()
        except req.HTTPError as exc:
            logger.error("Semantic Scholar API error: %s", exc)
            raise
        break
    else:
        raise RuntimeError(
            "Semantic Scholar rate-limit (429) persisted after "
            f"{_max_retries} retries. Add a SEMANTIC_SCHOLAR_API_KEY to .env "
            "or reduce request frequency."
        )

    docs: list[Document] = []
    for paper in response.json().get("data", []):
        abstract: str = paper.get("abstract") or ""
        if not abstract:
            continue  # papers without abstracts are not useful for RAG
        external_ids: dict = paper.get("externalIds") or {}
        doc = Document(
            page_content=abstract,
            metadata={
                "source": "semantic_scholar",
                "title": paper.get("title", ""),
                "authors": ", ".join(
                    a.get("name", "") for a in (paper.get("authors") or [])
                ),
                "year": str(paper.get("year") or ""),
                "doi": external_ids.get("DOI", ""),
                # Keep arxiv_id so the dedup logic in the Ingested Papers browser
                # can identify cross-listed papers correctly.
                "arxiv_id": external_ids.get("ArXiv", ""),
                "published": paper.get("publicationDate", ""),
            },
        )
        docs.append(doc)

    logger.info("Fetched %d documents from Semantic Scholar.", len(docs))
    return docs


# ── Wikipedia ─────────────────────────────────────────────────────────────────

# Wikipedia's API policy (T400119) requires a descriptive User-Agent header.
# The third-party `wikipedia` library (v1.4.0) omits this, causing 403 / empty
# responses.  We call the MediaWiki API directly so we can set the header.
_WIKIPEDIA_USER_AGENT = "research-rag-app/1.0 (https://github.com/user/research-rag; educational)"
_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_articles(
    query: str,
    max_results: int = 3,
) -> list[Document]:
    """Fetch Wikipedia articles for general background context.

    Calls the MediaWiki Action API directly with a valid User-Agent so the
    request is not rejected by Wikipedia's bot-policy enforcement (T400119).
    Does not depend on the third-party ``wikipedia`` library.

    Args:
        query: Search string used to find the most relevant Wikipedia pages.
        max_results: Number of articles to load (each can be ~10 000 words).

    Returns:
        List of Document objects tagged with ``source: "wikipedia"``.

    Raises:
        requests.HTTPError: If the Wikipedia API returns a non-200 status.
    """
    import requests as req

    session = req.Session()
    session.headers.update({"User-Agent": _WIKIPEDIA_USER_AGENT})

    logger.info("Fetching %d Wikipedia articles | query: %s", max_results, query)

    # Step 1: search for relevant page titles
    search_resp = session.get(
        _WIKIPEDIA_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        },
        timeout=15,
    )
    search_resp.raise_for_status()
    search_data = search_resp.json()
    titles = [item["title"] for item in search_data.get("query", {}).get("search", [])]

    if not titles:
        logger.info("Wikipedia returned no search results for '%s'.", query)
        return []

    # Step 2: fetch the full extract (plain-text content) for each title
    docs: list[Document] = []
    for title in titles:
        try:
            page_resp = session.get(
                _WIKIPEDIA_API,
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "extracts|info",
                    "exintro": False,     # full article, not just intro
                    "explaintext": True,  # plain text, no wikitext markup
                    "inprop": "url",
                    "format": "json",
                },
                timeout=20,
            )
            page_resp.raise_for_status()
            pages = page_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text: str = page.get("extract", "").strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "wikipedia",
                            "title": page.get("title", title),
                            "url": page.get("fullurl", ""),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("Wikipedia: could not fetch page '%s': %s", title, exc)

    logger.info("Fetched %d Wikipedia articles.", len(docs))
    return docs