"""RAG chain: LLM retrieval and generation logic."""

import logging
from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import get_env_var, load_config

logger = logging.getLogger(__name__)


_PROMPT = """You are a research assistant. Answer the question concisely using only the context below.
If the context is insufficient, say so explicitly. Do not speculate.

Context:
{context}

Question: {question}

Answer:"""


def build_llm(config: dict, tier: str = "standard") -> BaseChatModel:
    llm_cfg: dict = config.get("llm", {}).get(tier, {})
    model_name = llm_cfg.get("model_name", "gpt-4o-mini")
    temperature = llm_cfg.get("temperature", 0.0)
    max_tokens = llm_cfg.get("max_tokens", 1024)
    api_key = get_env_var("OPENAI_API_KEY")
    logger.info("LLM | tier=%s | model=%s", tier, model_name)
    return ChatOpenAI(
        model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key
    )


def load_faiss_index(index_path: str, embedding_model: str) -> FAISS:
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def create_rag_pipeline(config_path: str = "config.yaml", return_sources: bool = False) -> Any:
    """Build a basic LCEL RAG chain from the persisted FAISS index."""
    config = load_config(config_path)
    embedding_model = config.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    index_path = config.get("vector_store", {}).get(
        "faiss_index_path", "data/faiss_index"
    )
    k = config.get("retrieval", {}).get("k", 5)

    vectorstore = load_faiss_index(index_path, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = build_llm(config, tier="standard")
    prompt = ChatPromptTemplate.from_template(_PROMPT)

    answer_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    if not return_sources:
        return answer_chain

    def _with_sources(question: str) -> dict:
        docs = retriever.invoke(question)
        return {"answer": answer_chain.invoke(question), "sources": docs}
    return _with_sources
