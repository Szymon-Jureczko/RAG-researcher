"""RAG chain module: responsible ONLY for LLM retrieval and generation logic.

Implements a hybrid LLM strategy:
- gpt-4o-mini (standard tier): re-ranking, factual queries, metadata lookups (~95%)
- gpt-4o / Claude 3.5 Sonnet (research tier): synthesis and hypothesis generation

Pipeline per QueryMode:
  STANDARD  → retrieve(k=20) | rerank(gpt-4o-mini → k=5)  | answer with gpt-4o-mini
  RESEARCH  → retrieve(k=20) | rerank(gpt-4o-mini → k=10) | synthesise with gpt-4o
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import get_env_var, load_config

logger = logging.getLogger(__name__)


# ── Query routing ─────────────────────────────────────────────────────────────

class QueryMode(str, Enum):
    """Controls which LLM tier handles answer generation.

    Attributes:
        STANDARD: Uses gpt-4o-mini. Best for factual lookups, sample-size
            questions, and single-paper queries (~95% of traffic).
        RESEARCH: Uses gpt-4o (or Claude 3.5 Sonnet). Reserved for cross-paper
            synthesis, conflicting-result analysis, and hypothesis generation.
    """

    STANDARD = "standard"
    RESEARCH = "research"


# ── Prompt templates ──────────────────────────────────────────────────────────

_STANDARD_PROMPT = """\
You are a research assistant. Answer the question concisely using only the context below.
If the context is insufficient, say so explicitly. Do not speculate beyond what is stated.

Domain: {domain}

Context:
{context}

Question: {question}

Answer:"""

_SYNTHESIS_PROMPT = """\
You are an expert research analyst with deep knowledge of scientific methodology.
Your primary goal is to directly and thoroughly answer the user's question using the provided research papers.

Instructions:
- Read the question carefully and answer it directly first.
- Support your answer with specific evidence from the papers (cite authors/titles where possible).
- Only discuss themes, conflicts, or methodology if they are directly relevant to answering the question.
- Do NOT produce a generic paper summary or synthesis unless the question explicitly asks for one.
- If the papers do not contain enough information to answer the question, say so clearly.

Domain: {domain}

Research Papers:
{context}

Question: {question}

Answer:"""

_RERANK_PROMPT = """\
You are a relevance filter for scientific literature retrieval.
Given a research question and numbered text chunks, identify the {k} most relevant chunks.

Rules:
- Return ONLY a comma-separated list of 0-based indices, most relevant first
- No explanations, no labels, no other text
- Example output for k=5: 3,0,12,7,1

Question: {question}

Chunks:
{chunks}

Top {k} indices:"""


# ── LLM factory ───────────────────────────────────────────────────────────────

def build_llm(config: dict, tier: str = "standard") -> BaseChatModel:
    """Instantiate an LLM for the specified tier using values from config.yaml.

    Supports OpenAI models (gpt-4o-mini, gpt-4o) and Anthropic (claude-*).
    API keys are always read from environment variables.

    Args:
        config: Parsed config.yaml dictionary.
        tier: LLM tier key — "standard", "research", or "metadata".

    Returns:
        A configured BaseChatModel instance.

    Raises:
        KeyError: If the tier is absent from the `llm` section of config.yaml.
        EnvironmentError: If the required API key is not set in the environment.
        ImportError: If langchain-anthropic is not installed and a Claude model is chosen.
    """
    llm_cfg: dict = config.get("llm", {}).get(tier, {})
    if not llm_cfg:
        raise KeyError(f"LLM tier '{tier}' not found in config.yaml under 'llm'.")

    model_name: str = llm_cfg.get("model_name", "gpt-4o-mini")
    temperature: float = llm_cfg.get("temperature", 0.0)
    max_tokens: int = llm_cfg.get("max_tokens", 1024)

    if model_name.startswith("claude"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Install langchain-anthropic to use Claude models: "
                "pip install langchain-anthropic"
            ) from exc
        api_key = get_env_var("ANTHROPIC_API_KEY")
        logger.info("LLM | tier=%s | model=%s (Anthropic)", tier, model_name)
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    api_key = get_env_var("OPENAI_API_KEY")
    logger.info("LLM | tier=%s | model=%s (OpenAI)", tier, model_name)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


# ── Re-ranking ────────────────────────────────────────────────────────────────

def rerank_documents(
    docs: list[Document],
    question: str,
    llm: BaseChatModel,
    k: int = 5,
) -> list[Document]:
    """Re-rank candidate documents by relevance using a single gpt-4o-mini call.

    Presents all candidates in one prompt, asking the model to return the indices
    of the top-k most relevant chunks. This avoids the cost of per-chunk API calls.
    Falls back to the original ordering if the LLM response cannot be parsed.

    Args:
        docs: Candidate documents from the ensemble retriever (typically k=20).
        question: The user's original question used as the relevance signal.
        llm: A fast, cheap model for scoring (always gpt-4o-mini in practice).
        k: Number of documents to retain after re-ranking.

    Returns:
        Re-ordered list of up to k Documents, most relevant first.
    """
    if len(docs) <= k:
        return docs

    # Truncate chunks to limit re-rank prompt size without losing key content
    chunks_text = "\n\n".join(
        f"[{i}] {doc.page_content[:600]}" for i, doc in enumerate(docs)
    )

    prompt = ChatPromptTemplate.from_template(_RERANK_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        response: str = chain.invoke(
            {"question": question, "chunks": chunks_text, "k": k}
        )
    except Exception as exc:
        logger.warning("Re-ranking LLM call failed; using original order: %s", exc)
        return docs[:k]

    indices: list[int] = []
    for part in response.strip().split(","):
        try:
            idx = int(part.strip())
            if 0 <= idx < len(docs) and idx not in indices:
                indices.append(idx)
        except ValueError:
            continue

    if not indices:
        logger.warning("Re-ranker returned unparseable output; falling back to top-%d.", k)
        return docs[:k]

    # Pad to k with any un-ranked docs if the model returned fewer indices than k
    seen = set(indices)
    for i in range(len(docs)):
        if len(indices) >= k:
            break
        if i not in seen:
            indices.append(i)

    logger.info("Re-ranked %d → %d chunks.", len(docs), k)
    return [docs[i] for i in indices[:k]]


# ── Retriever ─────────────────────────────────────────────────────────────────

def load_faiss_index(index_path: str, embedding_model: str) -> FAISS:
    """Load a persisted FAISS index from disk.

    Args:
        index_path: Directory path containing saved FAISS index files.
        embedding_model: HuggingFace model name used at build time.

    Returns:
        A FAISS vectorstore instance ready for similarity search.

    Raises:
        FileNotFoundError: If the index directory does not exist.
        Exception: If deserialization of the index fails.
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as exc:
        logger.error("Failed to load FAISS index from '%s': %s", index_path, exc)
        raise


def build_ensemble_retriever(
    vectorstore: FAISS,
    corpus_docs: list[Document],
    k: int = 20,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    domain: Optional[str] = None,
) -> EnsembleRetriever:
    """Build a hybrid BM25 + FAISS retriever with optional domain filtering.

    k is kept high here (default 20) so the re-ranker has a wide candidate pool
    to select from before the final answer LLM receives only the best chunks.

    Args:
        vectorstore: A populated FAISS vectorstore instance.
        corpus_docs: Documents used to build the BM25 keyword index.
        k: Number of candidates to retrieve from each sub-retriever.
        bm25_weight: Reciprocal-rank fusion weight for BM25 (0.0–1.0).
        vector_weight: Reciprocal-rank fusion weight for FAISS (0.0–1.0).
        domain: Optional domain key for metadata-based filtering.

    Returns:
        An EnsembleRetriever that merges BM25 and vector results via weighted RRF.

    Raises:
        ValueError: If corpus_docs is empty or weights do not sum to 1.0.
    """
    if not corpus_docs:
        raise ValueError("corpus_docs must not be empty to build a BM25 index.")
    if abs((bm25_weight + vector_weight) - 1.0) > 1e-6:
        raise ValueError(
            f"bm25_weight + vector_weight must equal 1.0 "
            f"(got {bm25_weight} + {vector_weight})."
        )

    if domain:
        filtered_corpus = [d for d in corpus_docs if d.metadata.get("domain") == domain]
        if not filtered_corpus:
            logger.warning("No docs for domain '%s'; falling back to full corpus.", domain)
            filtered_corpus = corpus_docs
        vector_search_kwargs: dict[str, Any] = {"k": k, "filter": {"domain": domain}}
    else:
        filtered_corpus = corpus_docs
        vector_search_kwargs = {"k": k}

    bm25_retriever = BM25Retriever.from_documents(filtered_corpus)
    bm25_retriever.k = k
    vector_retriever = vectorstore.as_retriever(search_kwargs=vector_search_kwargs)

    logger.info(
        "EnsembleRetriever | domain=%s | k=%d | bm25=%.2f | vec=%.2f",
        domain, k, bm25_weight, vector_weight,
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )


# ── Chain builders ────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Concatenate document page_content into a single context string.

    Args:
        docs: Retrieved Document objects.

    Returns:
        All page_content values joined by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    retriever: BaseRetriever,
    answer_llm: BaseChatModel,
    reranker_llm: BaseChatModel,
    mode: QueryMode = QueryMode.STANDARD,
    domain: Optional[str] = None,
    final_k: int = 5,
) -> Any:
    """Build a LCEL RAG chain with re-ranking and mode-based LLM routing.

    Chain structure (LCEL):
        question → {context: retrieve+rerank+format, question, domain}
                 → prompt → answer_llm → str

    The reranker_llm (always gpt-4o-mini) filters the initial 20 candidates
    down to final_k before the answer_llm generates the response.

    Args:
        retriever: Hybrid ensemble retriever returning initial candidates.
        answer_llm: LLM for final generation — gpt-4o-mini (STANDARD) or gpt-4o (RESEARCH).
        reranker_llm: Fast LLM for re-ranking, always gpt-4o-mini regardless of mode.
        mode: QueryMode controlling the prompt template.
        domain: Domain label injected into the prompt for traceability.
        final_k: Number of chunks to retain after re-ranking.

    Returns:
        A runnable LCEL chain: str → str.
    """
    template = _SYNTHESIS_PROMPT if mode == QueryMode.RESEARCH else _STANDARD_PROMPT
    prompt = ChatPromptTemplate.from_template(template)

    def retrieve_rerank_format(question: str) -> str:
        """Retrieve, re-rank, and format document context in one step."""
        docs = retriever.invoke(question)
        reranked = rerank_documents(docs, question, reranker_llm, k=final_k)
        return _format_docs(reranked)

    chain = (
        {
            "context": RunnableLambda(retrieve_rerank_format),
            "question": RunnablePassthrough(),
            "domain": lambda _: domain or "all",
        }
        | prompt
        | answer_llm
        | StrOutputParser()
    )
    logger.info(
        "RAG chain built | mode=%s | domain=%s | final_k=%d | answer_model=%s",
        mode.value,
        domain or "all",
        final_k,
        getattr(answer_llm, "model_name", "unknown"),
    )
    return chain


# ── Top-level factory ─────────────────────────────────────────────────────────

def create_rag_pipeline(
    query_domain: Optional[str] = None,
    mode: QueryMode = QueryMode.STANDARD,
    config_path: str = "config.yaml",
) -> Any:
    """Load the FAISS index and build the full RAG pipeline for a given query mode.

    LLM routing:
        STANDARD → reranker: gpt-4o-mini, answer: gpt-4o-mini, final_k: 5
        RESEARCH → reranker: gpt-4o-mini, answer: gpt-4o,      final_k: 10

    Args:
        query_domain: Optional domain key (e.g. "ml", "physics") for filtered retrieval.
            If None, searches all indexed documents.
        mode: QueryMode.STANDARD for factual queries; QueryMode.RESEARCH for synthesis.
        config_path: Path to the YAML configuration file.

    Returns:
        A runnable LCEL chain accepting a question string and returning an answer string.

    Raises:
        FileNotFoundError: If config.yaml or the FAISS index directory does not exist.
        EnvironmentError: If required API keys are missing from the environment.
        KeyError: If the requested LLM tier is absent from config.yaml.
    """
    config = load_config(config_path)

    embedding_model: str = config.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    index_path: str = config.get("vector_store", {}).get(
        "faiss_index_path", "data/faiss_index"
    )
    retrieval_cfg: dict = config.get("retrieval", {})
    rerank_k: int = retrieval_cfg.get("rerank_k", 20)
    final_k: int = (
        retrieval_cfg.get("research_k", 10)
        if mode == QueryMode.RESEARCH
        else retrieval_cfg.get("k", 5)
    )
    bm25_weight: float = retrieval_cfg.get("bm25_weight", 0.5)
    vector_weight: float = retrieval_cfg.get("vector_weight", 0.5)

    vectorstore = load_faiss_index(index_path, embedding_model)
    corpus_docs: list[Document] = list(vectorstore.docstore._dict.values())

    retriever = build_ensemble_retriever(
        vectorstore=vectorstore,
        corpus_docs=corpus_docs,
        k=rerank_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        domain=query_domain,
    )

    # gpt-4o-mini is always used for re-ranking regardless of query mode
    standard_llm = build_llm(config, tier="standard")
    answer_llm = build_llm(config, tier=mode.value)

    return build_rag_chain(
        retriever=retriever,
        answer_llm=answer_llm,
        reranker_llm=standard_llm,
        mode=mode,
        domain=query_domain,
        final_k=final_k,
    )
