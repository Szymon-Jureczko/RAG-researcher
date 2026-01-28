"""Pipeline: chunk, embed, and persist documents to FAISS."""

import logging
from pathlib import Path
from typing import Optional

import pymupdf4llm
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_config

logger = logging.getLogger(__name__)


def parse_pdf_to_document(pdf_path: str, metadata: Optional[dict] = None) -> Document:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    md_text: str = pymupdf4llm.to_markdown(pdf_path)
    doc_metadata = metadata or {}
    doc_metadata.setdefault("source", pdf_path)
    return Document(page_content=md_text, metadata=doc_metadata)


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d -> %d chunks.", len(docs), len(chunks))
    return chunks


def build_faiss_index(
    chunks: list[Document],
    embedding_model: str,
    index_save_path: str,
) -> FAISS:
    """Embed chunks and persist a FAISS index to disk."""
    if not chunks:
        raise ValueError("Cannot build FAISS index: chunk list is empty.")
    logger.info("Building FAISS index | %s | %d chunks", embedding_model, len(chunks))
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(index_save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_save_path)
    logger.info("FAISS index saved to '%s'.", index_save_path)
    return vectorstore


def run_pipeline(docs: list[Document], config_path: str = "config.yaml") -> FAISS:
    """Chunk -> embed -> index."""
    if not docs:
        raise ValueError("Document list is empty. Nothing to process.")
    config = load_config(config_path)
    chunk_size = config.get("chunking", {}).get("chunk_size", 1000)
    chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 200)
    embedding_model = config.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    index_path = config.get("vector_store", {}).get(
        "faiss_index_path", "data/faiss_index"
    )
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return build_faiss_index(chunks, embedding_model, index_path)
