# Project Context
We are building a highly scalable, domain-elastic Retrieval-Augmented Generation (RAG) system. It processes over 10,000 PDFs from sources like arXiv, PubMed, and local directories, utilizing a hybrid search approach (BM25 + vectors).

# Core Tech Stack
- **Language**: Python 3.10+
- **Orchestration**: LangChain, LangGraph (if applicable)
- **Document Parsing**: PyMuPDF4LLM (`pymupdf4llm`), ArxivLoader, DirectoryLoader
- **Vector Stores & Search**: FAISS (local/testing) and Weaviate (production hybrid search)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Frontend**: Streamlit
- **Pipeline Orchestration**: Apache Airflow

# Python Coding Standards
- **Type Hints**: Strictly enforce Python type hints (`typing` module) for all function arguments and return types.
- **Docstrings**: Use Google-style docstrings for every class and function. Include `Args`, `Returns`, and `Raises`.
- **Error Handling**: Never use bare `except:` blocks. Always catch specific exceptions and log them using Python's standard `logging` module.
- **Immutability**: Prefer immutable data structures where appropriate (e.g., `dataclasses` with `frozen=True` or Pydantic models).
- **Environment Variables**: Never hardcode API keys or file paths. Always use `os.getenv` or `python-dotenv` to fetch from the `.env` file.

# LangChain Specific Rules
- **Modern Imports**: LangChain updates frequently. Always use modern import paths.
  - *Correct*: `from langchain_community.document_loaders import ArxivLoader`
  - *Incorrect*: `from langchain.document_loaders import ArxivLoader`
- **Metadata Tagging**: Every document chunk must retain metadata (`domain`, `doi`, `date`, `source`) to ensure domain-specific filtering during retrieval.
- **Chains**: Prefer LangChain Expression Language (LCEL) using the `|` operator for chaining components over legacy `Chain` classes when generating new pipelines.

# Architecture Boundaries
- **Configuration**: All dynamic variables (domains, chunk sizes, model names) are driven by `config.yaml`. Read from this file before hardcoding logic.
- **Separation of Concerns**:
  - `src/crawlers.py` handles ONLY fetching and raw parsing.
  - `src/pipeline.py` handles ONLY chunking, embedding, and indexing.
  - `src/rag_chain.py` handles ONLY the LLM retrieval and generation logic.
