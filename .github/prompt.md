---
description: Generate Universal Research Paper RAG components using LangChain
name: research-rag
argument-hint: domain="ml" OR query="diffusion models"
---

# Universal Research Paper RAG Generator

## Context
You're building a **scalable, domain-elastic RAG system** that processes 10k+ PDFs from arXiv/PubMed/local files. Key requirements:
- LangChain + PyMuPDF4LLM for PDF parsing
- FAISS/Weaviate hybrid search (BM25 + vectors)
- Multi-domain filtering (`domain: "ml/physics/bio"`)
- Semantic chunking (1000 tokens/chunk)
- Daily ingestion pipeline for new papers

## Project Structure to Generate
```text
research-rag/
├── config.yaml          # domains, APIs, chunk sizes
├── src/
│   ├── crawlers.py      # ArxivLoader, PubMed, DirectoryLoader
│   ├── pipeline.py      # load→split→embed→index
│   ├── rag_chain.py     # RetrievalQA + hybrid search
│   └── app.py           # Streamlit UI
├── dags/                # Airflow ingestion DAGs
├── requirements.txt
└── .env                 # API keys, paths
