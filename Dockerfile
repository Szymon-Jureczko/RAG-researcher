FROM python:3.12-slim

WORKDIR /app

# System deps required by sentence-transformers, faiss, and pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source (layer cache)
COPY requirements.txt pyproject.toml ./
# apache-airflow is heavy — skip it here; run the Airflow service separately if needed
RUN pip install --no-cache-dir $(grep -v "apache-airflow" requirements.txt | grep -v "^\s*#" | grep -v "^\s*$" | tr '\n' ' ')

# Copy source and install the package
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
