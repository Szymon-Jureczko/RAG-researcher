"""Airflow DAG for scheduled research-paper ingestion.

Scans the local PDF directory defined in config.yaml and re-indexes any PDFs
found there.  This is useful when PDFs are deposited by an external process
(e.g. a papermill job, an S3 sync, or a manual upload via the Streamlit UI).

arXiv ingestion is intentionally not scheduled here — it is driven on-demand
through the Streamlit UI so users can search for any topic they need.

Schedule: daily at 02:00 UTC (configurable via `schedule`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── Resolve project root (two levels up from dags/) ──
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# ── DAG default args ──
_DEFAULT_ARGS: dict[str, Any] = {
    "owner": "research-rag",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _ingest_local_pdfs(config_path: str) -> None:
    """Airflow-callable: index all PDFs in the configured local directory.

    Imports project modules at execution time so the Airflow scheduler does not
    need the full ML dependency tree at DAG parse time.

    Args:
        config_path: Absolute path to config.yaml.
    """
    import sys
    sys.path.insert(0, _PROJECT_ROOT)

    import yaml
    from src.crawlers import fetch_local_pdfs
    from src.pipeline import run_pipeline

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    pdf_dir: str = (
        config.get("sources", {}).get("local", {}).get("pdf_dir", "data/papers/")
    )

    try:
        docs = fetch_local_pdfs(pdf_dir)
    except FileNotFoundError:
        logger.warning("PDF directory '%s' not found. Nothing to index.", pdf_dir)
        return

    if not docs:
        logger.warning("No PDFs found in '%s'. Nothing to index.", pdf_dir)
        return

    run_pipeline(docs, config_path)
    logger.info("Ingestion complete: %d documents indexed from '%s'.", len(docs), pdf_dir)


# ── Build the DAG ──
_config_path = str(Path(_PROJECT_ROOT, "config.yaml"))

with DAG(
    dag_id="research_paper_ingestion",
    default_args=_DEFAULT_ARGS,
    description="Daily re-index of PDFs in the local papers directory",
    schedule="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["research-rag", "ingestion"],
) as dag:
    PythonOperator(
        task_id="ingest_local_pdfs",
        python_callable=_ingest_local_pdfs,
        op_kwargs={"config_path": _config_path},
    )

