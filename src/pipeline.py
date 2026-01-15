"""Pipeline: PDF -> markdown via PyMuPDF4LLM. Chunking and indexing TBD."""

import logging
from pathlib import Path
from typing import Optional

import pymupdf4llm
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def parse_pdf_to_document(pdf_path: str, metadata: Optional[dict] = None) -> Document:
    """Convert a PDF to a LangChain Document via PyMuPDF4LLM markdown."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    md_text: str = pymupdf4llm.to_markdown(pdf_path)
    doc_metadata = metadata or {}
    doc_metadata.setdefault("source", pdf_path)
    return Document(page_content=md_text, metadata=doc_metadata)


def parse_pdf_directory(pdf_dir: str, domain: Optional[str] = None) -> list[Document]:
    """Convert every PDF under pdf_dir into Documents."""
    dir_path = Path(pdf_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    docs: list[Document] = []
    for pdf_path in dir_path.rglob("*.pdf"):
        meta: dict = {"source": str(pdf_path)}
        if domain:
            meta["domain"] = domain
        try:
            docs.append(parse_pdf_to_document(str(pdf_path), metadata=meta))
        except Exception as exc:
            logger.warning("Skipping '%s': %s", pdf_path, exc)
    return docs


from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into semantic chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d documents into %d chunks.", len(docs), len(chunks))
    return chunks
