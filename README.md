# RAG Researcher

A highly scalable, domain-elastic Retrieval-Augmented Generation (RAG) system for research papers. Ingests PDFs from arXiv, PubMed, Semantic Scholar, Wikipedia, and local directories; indexes them with a hybrid BM25 + vector search pipeline; and answers natural-language questions with LLM-generated responses grounded in your document corpus.

---

## Features

- **Multi-source ingestion** — arXiv, PubMed, Semantic Scholar (with auto-retry), Wikipedia, and local PDFs
- **Hybrid retrieval** — BM25 keyword search + FAISS vector search fused via Reciprocal-Rank Fusion (EnsembleRetriever)
- **LLM re-ranker** — gpt-4o-mini re-ranks the top-20 candidates down to the best 5–10 chunks before generation
- **Dual query modes** — *Standard* (gpt-4o-mini, fast) and *Deep Research* (gpt-4o / Claude, cross-paper synthesis)
- **Relevance filter** — cosine-similarity pre-filter drops off-topic results before they enter the index
- **LLM metadata extraction** — optional gpt-4o-mini pass enriches every document with title, authors, year, keywords, and DOI
- **Streamlit UI** — full-featured browser interface with ingestion controls, query mode picker, source viewer, and an ingested-papers browser
- **RAGAS evaluation** — LLM-as-a-Judge pipeline measuring Faithfulness, Answer Relevancy, and Context Precision; supports hardcoded golden datasets and fully synthetic question generation
- **Scheduled ingestion** — Apache Airflow DAG re-indexes local PDFs daily at 02:00 UTC
- **Docker-ready** — single `docker compose up` starts the Streamlit app; optional Weaviate sidecar for production-scale hybrid search

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Sources                                                         │
│  arXiv · PubMed · Semantic Scholar · Wikipedia · local PDFs     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ src/crawlers.py
                            │ fetch + relevance filter
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Ingestion pipeline                            src/pipeline.py  │
│  PDF → Markdown (PyMuPDF4LLM)                                   │
│  → RecursiveCharacterTextSplitter (1 000 chars / 200 overlap)   │
│  → optional LLM metadata enrichment (gpt-4o-mini)              │
│  → HuggingFace MiniLM embeddings → FAISS index                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │ BM25 retriever                    │ FAISS retriever
          │ (rank-bm25)                       │ (sentence-transformers)
          └─────────────────┬─────────────────┘
                            │ EnsembleRetriever (RRF, 0.5 / 0.5)
                            │ top-20 candidates
                            ▼
              LLM re-ranker (gpt-4o-mini) → top 5 / 10 chunks
                            │
                            ▼                    src/rag_chain.py
              Answer LLM ──────────────────────────────────────
              Standard mode  → gpt-4o-mini
              Research mode  → gpt-4o  (or Claude 3.5 Sonnet)
                            │
                            ▼
                     Generated answer

UI: src/app.py (Streamlit)   Scheduler: dags/ingestion_dag.py (Airflow)
Evaluation: src/eval.py (RAGAS)
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `src/config.py` | YAML config loader and env-var helper |
| `src/crawlers.py` | Fetch + raw parsing from all sources; relevance filter |
| `src/pipeline.py` | PDF→Markdown, chunking, metadata enrichment, FAISS index |
| `src/rag_chain.py` | BM25+FAISS retriever, LLM re-ranker, LCEL chain, LLM factory |
| `src/app.py` | Streamlit web UI |
| `src/eval.py` | RAGAS evaluation pipeline (LLM-as-a-Judge) |
| `dags/ingestion_dag.py` | Airflow DAG for scheduled ingestion |

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM orchestration | LangChain 0.3, LangChain Expression Language (LCEL) |
| LLMs | OpenAI gpt-4o / gpt-4o-mini, Anthropic Claude 3.5 Sonnet |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | FAISS (local / dev), Weaviate (production optional) |
| Keyword search | rank-bm25 |
| Document parsing | PyMuPDF4LLM |
| Frontend | Streamlit 1.32+ |
| Evaluation | RAGAS 0.2+ |
| Orchestration | Apache Airflow 2.8+ |
| Containers | Docker + Docker Compose |

---

## Quickstart

### Prerequisites

- Python 3.10+
- An OpenAI API key (required for LLM generation, re-ranking, and evaluation)
- Anthropic API key (optional — only needed when `llm.research.model_name` is set to a `claude-*` model)

### 1. Clone and install

```bash
git clone https://github.com/Szymon-Jureczko/RAG-researcher.git
cd RAG-researcher
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

> **Note:** `apache-airflow` is heavy and excluded from the editable install. Install it separately if you want to run the DAG locally: `pip install apache-airflow>=2.8.0`

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in at minimum:
# OPENAI_API_KEY=sk-...
```

All available variables:

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM generation, re-ranking, eval |
| `ANTHROPIC_API_KEY` | No | Claude research-tier LLM |
| `WEAVIATE_URL` | No | Production hybrid search (default: FAISS) |
| `WEAVIATE_API_KEY` | No | Weaviate auth |
| `HUGGINGFACE_TOKEN` | No | Gated HuggingFace models |
| `PDF_DIR` | No | Overrides `config.yaml` `sources.local.pdf_dir` |

### 3. Run the Streamlit app

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 4. Ingest papers and ask questions

1. Enter a search query in the **Ingestion** sidebar (e.g. `"transformer attention mechanisms"`)
2. Select one or more sources (arXiv, PubMed, Semantic Scholar, Wikipedia) or upload PDFs directly
3. Click **Run ingestion**
4. Type a question in the main input and receive a grounded answer

---

## Docker

```bash
# Start the Streamlit app
docker compose up --build

# Start the app + Weaviate (production hybrid search)
docker compose --profile weaviate up --build
```

Set `WEAVIATE_URL=http://weaviate:8080` in `.env` to switch from FAISS to Weaviate.

---

## Configuration reference (`config.yaml`)

```yaml
sources:
  arxiv:
    max_results: 100        # papers per arXiv query
  pubmed:
    max_results: 50
  semantic_scholar:
    max_results: 50         # hard cap: 100 per API call
  wikipedia:
    max_results: 3          # full articles; each can be ~10 000 words
  local:
    pdf_dir: "data/papers/" # directory scanned by the Airflow DAG

chunking:
  chunk_size: 1000          # characters per chunk (~250 tokens)
  chunk_overlap: 200

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"

vector_store:
  faiss_index_path: "data/faiss_index"
  weaviate_url: null        # set via WEAVIATE_URL env var

llm:
  standard:                 # gpt-4o-mini — ~95% of all requests
    model_name: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 1024
  research:                 # gpt-4o — synthesis and hypothesis generation
    model_name: "gpt-4o"   # swap to "claude-3-5-sonnet-20241022" for Claude
    temperature: 0.2
    max_tokens: 4096
  metadata:                 # gpt-4o-mini — document metadata extraction
    model_name: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 512

retrieval:
  k: 5                      # final chunks passed to the answer LLM
  rerank_k: 20              # initial retrieval pool fed to the re-ranker
  research_k: 10            # final chunks in Research mode
  bm25_weight: 0.5
  vector_weight: 0.5
```

---

## Evaluation (RAGAS)

The evaluation pipeline scores three RAG quality metrics using gpt-4o-mini as the judge:

| Metric | What it measures | Target |
|---|---|---|
| **Faithfulness** | Every claim in the answer is supported by retrieved context | > 0.80 |
| **Answer Relevancy** | The answer addresses the actual question asked | > 0.80 |
| **Context Precision** | The highest-ranked chunks are the most useful ones | > 0.70 |

### Run with the hardcoded smoke-test dataset

```bash
python -m src.eval
```

### Auto-generate questions from your own corpus

```bash
# Generate 20 questions from whatever documents you have indexed
python -m src.eval --synthetic --n-questions 20

# Save for reuse (avoids re-generation cost on repeated runs)
python -m src.eval --synthetic --n-questions 15 --save-dataset data/golden_dataset.csv

# Evaluate with the research-tier LLM
python -m src.eval --mode research
```

---

## Airflow DAG

The `research_paper_ingestion` DAG runs daily at 02:00 UTC and re-indexes every PDF found in `data/papers/`. It is designed to pair with an external file-drop process (S3 sync, papermill, or Streamlit upload).

arXiv ingestion is intentionally **not** scheduled — it is driven on-demand through the Streamlit UI so queries can target any topic dynamically.

### Run locally with the Airflow standalone server

```bash
pip install apache-airflow>=2.8.0
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow standalone
# copy the DAG
cp dags/ingestion_dag.py $AIRFLOW_HOME/dags/
```

---

## Project structure

```
RAG-researcher/
├── config.yaml                # central configuration
├── Dockerfile                 # production image (Streamlit service)
├── docker-compose.yml         # app + optional Weaviate sidecar
├── pyproject.toml
├── requirements.txt
├── .env.example               # environment variable template
├── dags/
│   └── ingestion_dag.py       # Airflow daily ingestion DAG
├── data/
│   ├── faiss_index/           # persisted FAISS index (gitignored)
│   └── papers/                # local PDF storage (gitignored)
└── src/
    ├── __init__.py
    ├── app.py                 # Streamlit UI
    ├── config.py              # YAML loader + env-var helper
    ├── crawlers.py            # multi-source fetchers + relevance filter
    ├── pipeline.py            # chunking, embedding, FAISS indexing
    ├── rag_chain.py           # BM25+FAISS retriever, re-ranker, LCEL chain
    └── eval.py                # RAGAS evaluation pipeline
```

---

## Using Claude instead of gpt-4o

Edit `config.yaml`:

```yaml
llm:
  research:
    model_name: "claude-3-5-sonnet-20241022"
    temperature: 0.2
    max_tokens: 4096
```

Add `ANTHROPIC_API_KEY` to `.env`. The `langchain-anthropic` package is already included in `requirements.txt`.

---

## License

MIT
