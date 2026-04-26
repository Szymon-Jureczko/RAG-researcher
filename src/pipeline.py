"""Pipeline module: responsible ONLY for chunking, embedding, and indexing documents.

Takes raw Document objects from crawlers.py and produces a persisted FAISS index.
PDF-to-markdown conversion via PyMuPDF4LLM is handled here before chunking.
"""

import logging
from pathlib import Path
from typing import Optional

import pymupdf4llm
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.config import get_env_var, load_config

logger = logging.getLogger(__name__)


# ── Metadata extraction ───────────────────────────────────────────────────────

class PaperMetadata(BaseModel, frozen=True):
    """Structured bibliographic metadata extracted from a research paper.

    Used as the structured output schema for gpt-4o-mini during ingestion.
    frozen=True enforces immutability after extraction.
    """

    title: str = Field(description="Full paper title")
    authors: list[str] = Field(description="Author names, e.g. ['Smith J', 'Lee A']")
    year: str = Field(description='4-digit publication year, e.g. "2024"')
    keywords: list[str] = Field(description="Up to 8 subject keywords")
    doi: str = Field(default="", description="DOI string if present, else empty string")


_METADATA_EXTRACTION_PROMPT = """\
Extract structured bibliographic metadata from the research paper text below.
If a field cannot be determined from the text, use an empty string or empty list.

Text:
{text}"""


def parse_pdf_to_document(pdf_path: str, metadata: Optional[dict] = None) -> Document:
    """Convert a single PDF to a LangChain Document via PyMuPDF4LLM markdown extraction.

    Args:
        pdf_path: Path to the PDF file on disk.
        metadata: Optional metadata dict to attach (e.g. domain, doi, date, source).

    Returns:
        A Document whose page_content is the markdown-extracted text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: If PyMuPDF4LLM fails to parse the file.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.debug("Parsing PDF to markdown: %s", pdf_path)
    try:
        md_text: str = pymupdf4llm.to_markdown(pdf_path)
        doc_metadata = metadata or {}
        doc_metadata.setdefault("source", pdf_path)
        return Document(page_content=md_text, metadata=doc_metadata)
    except Exception as exc:
        logger.error("Failed to parse PDF '%s': %s", pdf_path, exc)
        raise


def parse_pdf_directory(
    pdf_dir: str,
    domain: Optional[str] = None,
) -> list[Document]:
    """Batch-convert all PDFs in a directory to Documents via PyMuPDF4LLM.

    Processes up to 10,000+ PDFs by iterating recursively over the directory.

    Args:
        pdf_dir: Path to the directory containing PDF files.
        domain: Optional domain label to embed in each document's metadata.

    Returns:
        List of Document objects parsed from all discovered PDF files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(pdf_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = list(dir_path.rglob("*.pdf"))
    logger.info("Found %d PDF files in '%s'. Parsing...", len(pdf_files), pdf_dir)

    docs: list[Document] = []
    for pdf_path in pdf_files:
        metadata: dict = {"source": str(pdf_path)}
        if domain:
            metadata["domain"] = domain
        try:
            doc = parse_pdf_to_document(str(pdf_path), metadata=metadata)
            docs.append(doc)
        except Exception as exc:
            logger.warning("Skipping '%s' due to parse error: %s", pdf_path, exc)

    logger.info("Successfully parsed %d / %d PDFs.", len(docs), len(pdf_files))
    return docs


def extract_document_metadata(doc: Document, llm: BaseChatModel) -> Document:
    """Use gpt-4o-mini to extract structured bibliographic metadata from a document.

    Sends the first 2500 characters (title + abstract region) to the model via
    structured output, then merges the result into the document's metadata dict.

    Args:
        doc: A Document whose page_content begins with the paper's title/abstract.
        llm: An LLM that supports structured output (e.g. gpt-4o-mini).

    Returns:
        A new Document with enriched metadata (title, authors, year, keywords, doi).
        Falls back to the unmodified original Document if extraction fails.
    """
    text_excerpt = doc.page_content[:2500]
    structured_llm = llm.with_structured_output(PaperMetadata)
    prompt = ChatPromptTemplate.from_template(_METADATA_EXTRACTION_PROMPT)
    chain = prompt | structured_llm
    try:
        paper_meta: PaperMetadata = chain.invoke({"text": text_excerpt})
        enriched = {**doc.metadata, **paper_meta.model_dump()}
        return Document(page_content=doc.page_content, metadata=enriched)
    except Exception as exc:
        logger.warning("Metadata extraction skipped for one document: %s", exc)
        return doc


def enrich_documents_metadata(
    docs: list[Document],
    config_path: str = "config.yaml",
) -> list[Document]:
    """Batch-enrich all documents with LLM-extracted bibliographic metadata.

    Uses gpt-4o-mini (metadata tier from config.yaml) to extract title, authors,
    year, keywords, and DOI. Processes each document individually and skips
    failures without halting the batch. Logs progress every 50 documents.

    Args:
        docs: Raw Document objects from crawlers to annotate.
        config_path: Path to the YAML configuration file.

    Returns:
        List of Documents with enriched metadata fields.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        KeyError: If the "metadata" tier is missing from config.yaml's `llm` section.
    """
    config = load_config(config_path)
    llm_cfg: dict = config.get("llm", {}).get("metadata", {})
    api_key = get_env_var("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model=llm_cfg.get("model_name", "gpt-4o-mini"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 512),
        api_key=api_key,
    )

    enriched_docs: list[Document] = []
    for i, doc in enumerate(docs):
        enriched_docs.append(extract_document_metadata(doc, llm))
        if (i + 1) % 50 == 0:
            logger.info("Metadata enrichment: %d / %d documents.", i + 1, len(docs))

    logger.info("Metadata enrichment complete: %d documents processed.", len(docs))
    return enriched_docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into semantic chunks, preserving all original metadata.

    Args:
        docs: List of raw Document objects to split.
        chunk_size: Maximum character count per chunk (roughly 1 token ≈ 4 chars).
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of chunked Document objects. Metadata (domain, doi, date, source) is
        propagated to every chunk automatically by LangChain's splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d documents into %d chunks.", len(docs), len(chunks))
    return chunks


def build_faiss_index(
    chunks: list[Document],
    embedding_model: str,
    index_save_path: str,
) -> FAISS:
    """Embed document chunks and build a persisted FAISS vector index.

    Args:
        chunks: List of chunked Document objects to embed and index.
        embedding_model: HuggingFace model identifier for sentence embeddings.
        index_save_path: Directory path where the FAISS index will be saved.

    Returns:
        A populated FAISS vectorstore instance.

    Raises:
        ValueError: If the chunks list is empty.
        Exception: If embedding generation or FAISS indexing fails.
    """
    if not chunks:
        raise ValueError("Cannot build FAISS index: chunk list is empty.")

    logger.info(
        "Building FAISS index | model: %s | chunks: %d", embedding_model, len(chunks)
    )
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        Path(index_save_path).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(index_save_path)
        logger.info("FAISS index saved to '%s'.", index_save_path)
        return vectorstore
    except Exception as exc:
        logger.error("FAISS index build failed: %s", exc)
        raise


def load_faiss_index(index_path: str, embedding_model: str) -> FAISS:
    """Load a previously saved FAISS index from disk.

    Args:
        index_path: Directory path containing the saved FAISS index files.
        embedding_model: HuggingFace model identifier (must match the one used at build time).

    Returns:
        A FAISS vectorstore loaded with allow_dangerous_deserialization enabled.

    Raises:
        FileNotFoundError: If the index directory does not exist.
        Exception: If the index cannot be deserialized.
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

    logger.info("Loading FAISS index from '%s'.", index_path)
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as exc:
        logger.error("Failed to load FAISS index from '%s': %s", index_path, exc)
        raise


def run_pipeline(
    docs: list[Document],
    config_path: str = "config.yaml",
    enrich_metadata: bool = False,
) -> FAISS:
    """Execute the full ingestion pipeline: (enrich) → chunk → embed → index.

    Reads all parameters from config.yaml. Intended to be called by crawlers
    after document fetching, or by Airflow DAGs for scheduled ingestion.

    Args:
        docs: Raw Document objects produced by crawlers.py.
        config_path: Path to the YAML configuration file.
        enrich_metadata: If True, run gpt-4o-mini over each document to extract
            structured metadata (title, authors, year, keywords, doi) before
            chunking. Adds ~1 API call per document; disable for large test runs.

    Returns:
        A populated and persisted FAISS vectorstore.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If docs list is empty.
    """
    if not docs:
        raise ValueError("Document list is empty. Nothing to process.")

    if enrich_metadata:
        logger.info("Enriching document metadata with gpt-4o-mini (%d docs)...", len(docs))
        docs = enrich_documents_metadata(docs, config_path)

    config = load_config(config_path)
    chunk_size: int = config.get("chunking", {}).get("chunk_size", 1000)
    chunk_overlap: int = config.get("chunking", {}).get("chunk_overlap", 200)
    embedding_model: str = config.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    index_path: str = config.get("vector_store", {}).get(
        "faiss_index_path", "data/faiss_index"
    )

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return build_faiss_index(chunks, embedding_model, index_path)
