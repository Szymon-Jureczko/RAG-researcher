"""RAG chain: LLM retrieval and generation logic.

Initial scaffold: just the OpenAI LLM factory and FAISS index loader.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import get_env_var, load_config

logger = logging.getLogger(__name__)


def build_llm(config: dict, tier: str = "standard") -> BaseChatModel:
    """Instantiate an OpenAI LLM for the requested tier."""
    llm_cfg: dict = config.get("llm", {}).get(tier, {})
    model_name: str = llm_cfg.get("model_name", "gpt-4o-mini")
    temperature: float = llm_cfg.get("temperature", 0.0)
    max_tokens: int = llm_cfg.get("max_tokens", 1024)
    api_key = get_env_var("OPENAI_API_KEY")
    logger.info("LLM | tier=%s | model=%s", tier, model_name)
    return ChatOpenAI(
        model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key
    )


def load_faiss_index(index_path: str, embedding_model: str) -> FAISS:
    """Load a persisted FAISS index from disk."""
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
