"""src/eval.py — RAG evaluation pipeline (LLM-as-a-Judge).

Adds RAGAS Faithfulness and AnswerRelevancy metrics. Context Precision TBD.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from src.config import get_env_var, load_config
from src.rag_chain import QueryMode, create_rag_pipeline

load_dotenv()
logger = logging.getLogger(__name__)
_EVAL_CONFIG_PATH: str = os.getenv("EVAL_CONFIG_PATH", "config.yaml")


@dataclass(frozen=True)
class GoldenRecord:
    question: str
    reference_answer: str
    domain: str | None = None


def load_golden_dataset() -> list[GoldenRecord]:
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


def _build_judge_llm() -> LangchainLLMWrapper:
    api_key = get_env_var("OPENAI_API_KEY")
    return LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)
    )


def _build_judge_embeddings(model_name: str) -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model_name))


def run_evaluation(records: list[GoldenRecord]) -> pd.DataFrame:
    cfg = load_config(_EVAL_CONFIG_PATH)
    embedding_model = cfg.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    judge_llm = _build_judge_llm()
    judge_embeddings = _build_judge_embeddings(embedding_model)
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
    ]
    chain = create_rag_pipeline(mode=QueryMode.STANDARD)
    samples: list[SingleTurnSample] = []
    for r in records:
        # NOTE: contexts are not yet captured; placeholder until the wrapper lands
        answer = chain.invoke(r.question)
        samples.append(
            SingleTurnSample(
                user_input=r.question,
                response=answer,
                retrieved_contexts=[],
                reference=r.reference_answer,
            )
        )
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = run_evaluation(load_golden_dataset())
    print(df)
