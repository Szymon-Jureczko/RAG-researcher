"""src/eval.py — RAG evaluation pipeline (LLM-as-a-Judge).

Initial scaffold with a hardcoded golden dataset. Metric integration TBD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.rag_chain import QueryMode, create_rag_pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoldenRecord:
    """A single labeled evaluation example."""

    question: str
    reference_answer: str
    domain: str | None = None


def load_golden_dataset() -> list[GoldenRecord]:
    """Hardcoded smoke-test dataset used until a CSV is wired in."""
    return [
        GoldenRecord(
            question="What are the main advantages of transformer architectures over RNNs?",
            reference_answer=(
                "Transformers process sequences in parallel via self-attention, "
                "handle long-range dependencies, and scale better with compute."
            ),
        ),
        GoldenRecord(
            question="How does the Vision Transformer (ViT) apply self-attention to images?",
            reference_answer=(
                "ViT splits an image into patches, embeds them, adds positional "
                "encodings, then applies self-attention across patch embeddings."
            ),
        ),
    ]


def run_evaluation() -> None:
    records = load_golden_dataset()
    chain = create_rag_pipeline(mode=QueryMode.STANDARD)
    for r in records:
        answer = chain.invoke(r.question)
        logger.info("Q: %s\nA: %s\nREF: %s", r.question, answer, r.reference_answer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
