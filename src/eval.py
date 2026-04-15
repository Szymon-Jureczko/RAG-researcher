"""src/eval.py — Automated RAG Triad evaluation pipeline (LLM-as-a-Judge).

Evaluates three core RAG quality metrics via the RAGAS library:

  Faithfulness      — Is every claim in the generated answer directly supported
                      by the retrieved context chunks?
                      How it's graded: the judge LLM extracts all factual
                      statements from the answer, then verifies each one is
                      entailed by the context.  Score = supported / total claims.
                      Target > 0.80 — values below this indicate hallucination.

  Answer Relevancy  — Does the answer actually address the question that was asked?
                      How it's graded: the judge LLM generates N hypothetical
                      questions from the answer, embeds them, and computes cosine
                      similarity to the original question.  Score ∈ [0, 1].
                      Target > 0.80 — low scores mean the answer drifted off-topic.

  Context Precision — Are the highest-ranked retrieved chunks the most useful ones?
                      How it's graded: each chunk is classified relevant/irrelevant
                      w.r.t. the reference (ground-truth) answer, then average
                      precision at rank k is computed (AP@K).  Score ∈ [0, 1].
                      Target > 0.70 — low scores mean the retriever is surfacing
                      noise ahead of signal.

Judge model: gpt-4o-mini (cheap and fast; reserved for eval runs only).
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (same model as the RAG stack).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

from langchain_core.documents import Document

from src.config import get_env_var, load_config
from src.rag_chain import (
    QueryMode,
    build_ensemble_retriever,
    build_llm,
    build_rag_chain,
    load_faiss_index,
    rerank_documents,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Allow override via env var so CI/CD can point at a different config file.
_EVAL_CONFIG_PATH: str = os.getenv("EVAL_CONFIG_PATH", "config.yaml")


# ── Golden dataset ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoldenRecord:
    """A single labeled evaluation example.

    Attributes:
        question: Natural language query sent to the RAG pipeline.
        reference_answer: Ground-truth answer.  Used by Context Precision to
            decide whether each retrieved chunk was actually useful.
        domain: Optional domain key matching config.yaml (e.g. "ml", "physics",
            "bio").  None means search across all indexed documents.
    """

    question: str
    reference_answer: str
    domain: str | None = None


def load_golden_dataset() -> list[GoldenRecord]:
    """Return the evaluation dataset.

    **To swap in a CSV file**, replace this function body with:

    .. code-block:: python

        import csv
        with open("data/golden_dataset.csv", newline="") as fh:
            reader = csv.DictReader(fh)
            return [
                GoldenRecord(
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    domain=row.get("domain") or None,
                )
                for row in reader
            ]

    The return type ``list[GoldenRecord]`` is the stable contract the rest of
    the pipeline depends on — keep it unchanged when swapping sources.

    Returns:
        Hardcoded list of GoldenRecord instances covering the three configured
        domains (ml, physics, bio).
    """
    return [
        GoldenRecord(
            question=(
                "What are the main advantages of transformer architectures "
                "over RNNs for sequence modeling?"
            ),
            reference_answer=(
                "Transformers process sequences in parallel via self-attention, "
                "eliminating the sequential bottleneck of RNNs. They handle "
                "long-range dependencies more effectively and scale better with "
                "data and compute."
            ),
        ),
        GoldenRecord(
            question=(
                "How does the Vision Transformer (ViT) apply self-attention "
                "to image data?"
            ),
            reference_answer=(
                "ViT splits an image into fixed-size patches, linearly embeds "
                "each patch, and adds positional encodings. Self-attention is "
                "then applied across all patch embeddings, allowing each patch "
                "to attend to every other patch regardless of spatial distance."
            ),
        ),
        GoldenRecord(
            question=(
                "What is quantum entanglement and why is it useful "
                "in quantum computing?"
            ),
            reference_answer=(
                "Quantum entanglement correlates qubits such that measuring one "
                "instantly determines its partner's state. This enables "
                "exponential parallelism and is a core resource in algorithms "
                "like Shor's and Grover's."
            ),
        ),
        GoldenRecord(
            question=(
                "How do transformer-based models contribute to protein "
                "structure prediction?"
            ),
            reference_answer=(
                "Models like AlphaFold 2 and ESMFold use attention over multiple "
                "sequence alignments and pairwise residue representations to "
                "predict 3D protein structures with near-experimental accuracy, "
                "dramatically accelerating drug discovery workflows."
            ),
        ),
    ]


# ── Corpus helpers ────────────────────────────────────────────────────────────

def load_corpus_docs(config_path: str = _EVAL_CONFIG_PATH) -> list[Document]:
    """Load every document stored in the FAISS index as LangChain Document objects.

    These are the raw chunks that were embedded during ingestion.  Passing them
    to ``generate_synthetic_dataset`` lets the LLM synthesise questions that are
    100% grounded in the actual corpus — no domain knowledge required from the
    developer.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        All Document objects stored in the FAISS docstore.

    Raises:
        FileNotFoundError: If the FAISS index directory does not exist.
    """
    cfg = load_config(config_path)
    embedding_model: str = cfg.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    index_path: str = cfg.get("vector_store", {}).get(
        "faiss_index_path", "data/faiss_index"
    )
    vectorstore = load_faiss_index(index_path, embedding_model)
    docs: list[Document] = list(vectorstore.docstore._dict.values())
    logger.info("Loaded %d document chunks from FAISS index.", len(docs))
    return docs


def generate_synthetic_dataset(
    docs: list[Document],
    n: int = 10,
    config_path: str = _EVAL_CONFIG_PATH,
    save_path: str | None = None,
) -> list[GoldenRecord]:
    """Auto-generate a golden dataset from the indexed corpus using RAGAS.

    This makes the evaluation pipeline fully domain-agnostic.  You never need
    to write a single question by hand.  The LLM reads chunks from *your*
    documents and generates realistic questions along with reference answers,
    covering whatever subject matter those documents contain.

    Generation strategy used by RAGAS ``TestsetGenerator``:

    * **Simple**  — single-chunk factual questions ("What is X?")
    * **Reasoning** — multi-hop questions requiring synthesis across chunks
    * **Multi-context** — questions that can only be answered by combining
      information from two or more chunks

    Using all three types exercises every layer of the RAG stack.

    Args:
        docs: Corpus chunks loaded from the FAISS index via ``load_corpus_docs``.
            Any collection of LangChain ``Document`` objects works here — you
            can pass PDFs, web pages, or any other source.
        n: Number of question/answer pairs to synthesise.  10–20 gives a
            statistically stable signal without excessive API cost.
        config_path: Path to ``config.yaml`` (reads embedding model name).
        save_path: If provided, the generated dataset is saved as a CSV at
            this path so it can be reused across runs without re-generating.

    Returns:
        ``list[GoldenRecord]`` — same contract as ``load_golden_dataset()``,
        so it plugs straight into ``run_evaluation()`` without any other
        changes.

    Raises:
        ImportError: If ``ragas.testset`` is not available (ragas < 0.2).
        RuntimeError: If the generator returns no valid samples.
    """
    try:
        from ragas.testset import TestsetGenerator
    except ImportError as exc:
        raise ImportError(
            "ragas.testset requires ragas>=0.2.0: pip install 'ragas>=0.2.0'"
        ) from exc

    cfg = load_config(config_path)
    embedding_model: str = cfg.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    judge_llm = _build_judge_llm()
    judge_embeddings = _build_judge_embeddings(embedding_model)

    # RAGAS TestsetGenerator runs one LLM call per document for knowledge-graph
    # construction (entity extraction, relationship building) before it generates
    # any questions.  Passing all corpus chunks causes O(chunks) API calls and
    # makes a 15-question run take 10–20 minutes.
    #
    # Rule of thumb: ~5× the desired testset_size is enough variety without
    # wasting API budget.  We shuffle so the sample is representative, not just
    # the first N chunks that happened to be stored.
    import random as _random
    _max_docs_for_generation = max(n * 5, 30)
    if len(docs) > _max_docs_for_generation:
        sampled_docs = _random.sample(docs, _max_docs_for_generation)
        logger.info(
            "Sampled %d chunks (from %d total) for testset generation to limit API cost.",
            _max_docs_for_generation, len(docs),
        )
    else:
        sampled_docs = docs

    logger.info(
        "Generating %d synthetic questions from %d corpus chunks (judge: gpt-4o-mini)…",
        n, len(sampled_docs),
    )
    generator = TestsetGenerator(llm=judge_llm, embedding_model=judge_embeddings)
    testset = generator.generate_with_langchain_docs(sampled_docs, testset_size=n)

    records: list[GoldenRecord] = [
        GoldenRecord(
            question=sample.user_input,
            # reference is the LLM-generated ground-truth answer from the source chunk.
            # It plays the same role as the human-written answer in the hardcoded dataset.
            reference_answer=sample.reference or "",
        )
        for sample in testset.samples
        if sample.user_input
    ]

    if not records:
        raise RuntimeError(
            "TestsetGenerator returned no valid samples. "
            "Ensure the FAISS index is populated before running synthetic generation."
        )

    logger.info("Generated %d synthetic golden records.", len(records))

    # ── Optional persistence ───────────────────────────────────────────────
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["question", "reference_answer", "domain"])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "question": r.question,
                    "reference_answer": r.reference_answer,
                    "domain": r.domain or "",
                })
        logger.info("Saved synthetic dataset to '%s' for reuse.", save_path)

    return records


# ── RAG pipeline wrapper ───────────────────────────────────────────────────────

class RAGEvaluator:
    """Thin wrapper around the existing RAG pipeline that exposes raw contexts.

    The standard ``create_rag_pipeline`` LCEL chain returns only a string
    answer.  RAGAS also needs the retrieved context chunks as a list.  This
    class builds the identical set of components and surfaces both artefacts
    through ``query_rag`` without modifying ``rag_chain.py``.

    Components are lazy-loaded on the first call to ``query_rag`` so that
    instantiation is cheap and test setup does not block immediately.

    Args:
        config_path: Path to ``config.yaml``.
        mode: ``QueryMode.STANDARD`` (gpt-4o-mini answer) or
            ``QueryMode.RESEARCH`` (gpt-4o answer).
    """

    def __init__(
        self,
        config_path: str = _EVAL_CONFIG_PATH,
        mode: QueryMode = QueryMode.STANDARD,
    ) -> None:
        self._config_path = config_path
        self.mode = mode
        self._retriever = None
        self._chain = None
        self._reranker_llm = None
        self._final_k: int = 5
        self._initialized: bool = False

    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        """Lazy-load the FAISS index and build retriever + chain once."""
        if self._initialized:
            return

        cfg = load_config(self._config_path)
        embedding_model: str = cfg.get("embeddings", {}).get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        index_path: str = cfg.get("vector_store", {}).get(
            "faiss_index_path", "data/faiss_index"
        )
        retrieval_cfg: dict = cfg.get("retrieval", {})
        rerank_k: int = retrieval_cfg.get("rerank_k", 20)
        self._final_k = (
            retrieval_cfg.get("research_k", 10)
            if self.mode == QueryMode.RESEARCH
            else retrieval_cfg.get("k", 5)
        )
        bm25_weight: float = retrieval_cfg.get("bm25_weight", 0.5)
        vector_weight: float = retrieval_cfg.get("vector_weight", 0.5)

        logger.info("Loading FAISS index from '%s'…", index_path)
        vectorstore = load_faiss_index(index_path, embedding_model)
        corpus_docs = list(vectorstore.docstore._dict.values())

        self._retriever = build_ensemble_retriever(
            vectorstore=vectorstore,
            corpus_docs=corpus_docs,
            k=rerank_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )
        self._reranker_llm = build_llm(cfg, tier="standard")
        answer_llm = build_llm(cfg, tier=self.mode.value)
        self._chain = build_rag_chain(
            retriever=self._retriever,
            answer_llm=answer_llm,
            reranker_llm=self._reranker_llm,
            mode=self.mode,
            final_k=self._final_k,
        )
        self._initialized = True
        logger.info("RAGEvaluator ready | mode=%s | final_k=%d", self.mode.value, self._final_k)

    # ------------------------------------------------------------------
    def query_rag(self, question: str, domain: str | None = None) -> dict:
        """Run the RAG pipeline and return both the generated answer and the
        raw context chunks that informed it.

        Retrieval and re-ranking are performed explicitly here so that the
        exact context strings are captured for RAGAS.  The LCEL chain is then
        invoked separately for answer generation.

        Note:
            This results in two retrieval passes (one for context capture, one
            inside the chain).  Both passes use the same retriever, re-ranker,
            and ``k`` value, so their results are deterministically equivalent.
            The overhead is acceptable for an offline evaluation script.

        Args:
            question: Natural language question to evaluate.
            domain: Optional domain key for metadata-filtered retrieval.
                Unused in the retriever built by ``_initialize`` (domain
                filtering can be added by extending this method).

        Returns:
            A dict with keys:

            * ``"answer"``   (``str``)        — generated answer string.
            * ``"contexts"`` (``list[str]``)  — page_content of each re-ranked
              chunk, in rank order.
        """
        self._initialize()

        # ── Context capture path ────────────────────────────────────────────
        # Retrieve and re-rank explicitly so we have the Document objects.
        raw_docs = self._retriever.invoke(question)
        reranked_docs = rerank_documents(
            raw_docs, question, self._reranker_llm, k=self._final_k
        )

        # ── Answer generation path ──────────────────────────────────────────
        # The LCEL chain performs its own equivalent retrieval internally.
        answer: str = self._chain.invoke(question)

        return {
            "answer": answer,
            "contexts": [doc.page_content for doc in reranked_docs],
        }


# ── RAGAS judge configuration ──────────────────────────────────────────────────

def _build_judge_llm() -> LangchainLLMWrapper:
    """Instantiate the gpt-4o-mini judge LLM used by all RAGAS metrics.

    Using gpt-4o-mini rather than gpt-4o keeps evaluation cost low while
    still providing accurate hallucination and relevance grading.

    Returns:
        A zero-temperature ``LangchainLLMWrapper`` around ``gpt-4o-mini``.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is not set.
    """
    api_key = get_env_var("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)
    return LangchainLLMWrapper(llm)


def _build_judge_embeddings(model_name: str) -> LangchainEmbeddingsWrapper:
    """Wrap the project's existing HuggingFace embedding model for RAGAS.

    RAGAS Answer Relevancy requires an embeddings model to compute cosine
    similarity between generated and hypothetical questions.  Reusing the
    same model already configured in ``config.yaml`` avoids an extra
    ``OpenAIEmbeddings`` round-trip and keeps costs at zero.

    Args:
        model_name: HuggingFace model identifier (read from ``config.yaml``).

    Returns:
        A ``LangchainEmbeddingsWrapper`` around a ``HuggingFaceEmbeddings`` instance.
    """
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return LangchainEmbeddingsWrapper(hf_embeddings)


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    records: list[GoldenRecord],
    evaluator: RAGEvaluator,
    config_path: str = _EVAL_CONFIG_PATH,
) -> pd.DataFrame:
    """Query the RAG pipeline for every golden record and score with RAGAS.

    Builds a ``SingleTurnSample`` per record containing:

    * ``user_input``        — the question
    * ``response``          — the answer generated by the pipeline
    * ``retrieved_contexts``— the re-ranked context chunks
    * ``reference``         — the ground-truth answer (required by Context Precision)

    Args:
        records: Labeled evaluation examples from ``load_golden_dataset()``.
        evaluator: Initialized ``RAGEvaluator`` wrapping the live RAG pipeline.
        config_path: Path to ``config.yaml`` (used to read the embedding model name).

    Returns:
        A ``pd.DataFrame`` with one row per question and columns for each metric.

    Raises:
        RuntimeError: If all RAG queries fail and no samples are collected.
    """
    cfg = load_config(config_path)
    embedding_model: str = cfg.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    judge_llm = _build_judge_llm()
    judge_embeddings = _build_judge_embeddings(embedding_model)

    # ── Metric definitions ─────────────────────────────────────────────────
    #
    #  All three metrics use the same gpt-4o-mini judge to keep costs uniform.
    #  AnswerRelevancy additionally uses HuggingFace embeddings (free/local).
    #
    metrics = [
        # Faithfulness: zero hallucination tolerance check.
        # Every factual claim in `response` must be traceable to `retrieved_contexts`.
        Faithfulness(llm=judge_llm),

        # Answer Relevancy: on-topic check.
        # Reverse-engineers questions from `response` and measures alignment with `user_input`.
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),

        # Context Precision: retriever quality check.
        # Ranks chunks as useful/not-useful against `reference`; rewards ranking good chunks first.
        ContextPrecision(llm=judge_llm),
    ]

    # ── Collect pipeline outputs ───────────────────────────────────────────
    samples: list[SingleTurnSample] = []
    row_metadata: list[dict] = []

    for idx, record in enumerate(records, start=1):
        logger.info(
            "Querying pipeline [%d/%d]: %s…",
            idx, len(records), record.question[:70],
        )
        try:
            output = evaluator.query_rag(record.question, domain=record.domain)
        except Exception as exc:
            logger.error("Pipeline query failed for record %d: %s", idx, exc)
            continue

        samples.append(
            SingleTurnSample(
                user_input=record.question,
                response=output["answer"],
                retrieved_contexts=output["contexts"],
                # `reference` is the human-verified answer; Context Precision uses
                # it to judge whether each retrieved chunk was genuinely relevant.
                reference=record.reference_answer,
            )
        )
        row_metadata.append({
            "question": record.question,
            "domain": record.domain or "all",
        })

    if not samples:
        raise RuntimeError(
            "No samples were collected — all RAG pipeline queries failed. "
            "Check that OPENAI_API_KEY is set and the FAISS index is populated."
        )

    # ── Run RAGAS ─────────────────────────────────────────────────────────
    logger.info("Sending %d samples to RAGAS (judge: gpt-4o-mini)…", len(samples))
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, metrics=metrics)

    # ── Assemble output DataFrame ──────────────────────────────────────────
    scores_df: pd.DataFrame = result.to_pandas()

    # Column names RAGAS uses (stable across 0.2.x).
    # If a metric is missing (e.g. API error), the column may be absent — handle gracefully.
    desired_metric_cols = ["faithfulness", "answer_relevancy", "context_precision"]
    available_metric_cols = [c for c in desired_metric_cols if c in scores_df.columns]

    meta_df = pd.DataFrame(row_metadata)
    return pd.concat([meta_df, scores_df[available_metric_cols]], axis=1)


# ── Pretty-print output ────────────────────────────────────────────────────────

# Thresholds below which a metric is flagged as failing.
_METRIC_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
}


def print_results(df: pd.DataFrame) -> None:
    """Print evaluation scores in a clean, human-readable terminal format.

    Displays a per-question breakdown followed by a summary row with pass/fail
    indicators against the recommended thresholds.

    Args:
        df: DataFrame returned by ``run_evaluation()``.
    """
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 130)

    print("\n" + "═" * 90)
    print("  RAG TRIAD EVALUATION RESULTS")
    print("  Judge: gpt-4o-mini  |  Embeddings: all-MiniLM-L6-v2")
    print("═" * 90)
    print(df.to_string(index=False))
    print("─" * 90)

    # Summary: mean score per metric with pass/fail indicator.
    metric_cols = [c for c in _METRIC_THRESHOLDS if c in df.columns]
    means = df[metric_cols].mean()

    print("\n  AGGREGATE SCORES  (✓ = above threshold, ✗ = below threshold)\n")
    for col in metric_cols:
        val = means[col]
        threshold = _METRIC_THRESHOLDS[col]
        icon = "✓" if val >= threshold else "✗"
        label = col.replace("_", " ").title()
        bar_filled = int(val * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"    {icon}  {label:<22}  [{bar}]  {val:.3f}  (threshold ≥ {threshold:.2f})")

    print("\n" + "─" * 90 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval",
        description=(
            "RAG Triad evaluation pipeline (LLM-as-a-Judge).\n"
            "By default runs the hardcoded 4-question smoke test.\n"
            "Use --synthetic to auto-generate questions from your own corpus."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Generate questions automatically from the indexed corpus instead of "
            "using the hardcoded golden dataset.  Works for ANY domain — no manual "
            "question writing required."
        ),
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=10,
        metavar="N",
        help="Number of synthetic questions to generate (default: 10, used with --synthetic).",
    )
    parser.add_argument(
        "--save-dataset",
        metavar="PATH",
        default=None,
        help=(
            "Save the generated synthetic dataset to a CSV file at PATH for reuse. "
            "On subsequent runs load it with load_golden_dataset() to skip re-generation "
            "and save API costs."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in QueryMode],
        default=QueryMode.STANDARD.value,
        help="RAG query mode: 'standard' (gpt-4o-mini) or 'research' (gpt-4o). Default: standard.",
    )
    return parser


def main() -> None:
    """Run the full RAG Triad evaluation pipeline end-to-end.

    Dataset source (choose one):

    * **Default** — hardcoded 4-question smoke test, domain-specific.
    * **--synthetic** — auto-generated questions from your FAISS corpus.
      Domain-agnostic: works for any subject matter you have indexed.
    * **CSV reuse** — edit ``load_golden_dataset()`` to read a saved CSV
      from a previous ``--synthetic --save-dataset`` run.  Zero API cost.

    Examples::

        # Quick smoke test (hardcoded questions, no API cost for generation):
        python -m src.eval

        # Generate 20 questions from your own indexed corpus (any domain):
        python -m src.eval --synthetic --n-questions 20

        # Generate, save for reuse, and evaluate:
        python -m src.eval --synthetic --n-questions 15 --save-dataset data/golden_dataset.csv
    """
    args = _build_arg_parser().parse_args()
    logger.info("Starting RAG Triad evaluation…")

    if args.synthetic:
        # ── Domain-agnostic path ───────────────────────────────────────────
        # Load every chunk from the FAISS index, then ask the LLM to write
        # questions from them.  No domain knowledge from the developer needed.
        docs = load_corpus_docs(config_path=_EVAL_CONFIG_PATH)
        records = generate_synthetic_dataset(
            docs=docs,
            n=args.n_questions,
            config_path=_EVAL_CONFIG_PATH,
            save_path=args.save_dataset,
        )
    else:
        # ── Smoke-test path ────────────────────────────────────────────────
        records = load_golden_dataset()

    logger.info("Evaluating %d records (mode: %s)…", len(records), args.mode)
    evaluator = RAGEvaluator(
        config_path=_EVAL_CONFIG_PATH,
        mode=QueryMode(args.mode),
    )
    df = run_evaluation(records, evaluator)
    print_results(df)


if __name__ == "__main__":
    main()
