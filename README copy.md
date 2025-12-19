## Social Media VectorDB Ingestion

This project ingests social-media posts and comments from an Excel file, chunks the text, creates embeddings with Sentence Transformers (default: **e5-large**), and stores them in a configurable vector backend (AWS S3, ChromaDB, or pgvector).

### Features

- **Excel ingestion** for posts/comments (Facebook, Instagram, Twitter, etc.).
- **Config-driven pipeline** (model, chunking, vector backend).
- **Pluggable chunking strategies**, including recursive character chunking.
- **Pluggable vector stores**:
  - AWS S3 object-based vector store.
  - ChromaDB.
  - PostgreSQL with pgvector.

### Project Structure

```text
social_media_vectordb/
  README.md
  requirements.txt
  .env.example
  src/
    __init__.py
    constants.py
    config.py
    data_loader.py
    chunking.py
    embedding.py
    logging_utils.py
    main.py
    vector_store/
      __init__.py
      base.py
      s3_vector_store.py
      chroma_vector_store.py
      pgvector_store.py
  tests/
    __init__.py
```

### Setup

```bash
cd /home/prashant/social_media_vectordb
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your values (AWS credentials, database URLs, etc.).

### Configuration

Key settings are configured via environment variables (see `.env.example`) and loaded in `config.Config`:

- **Model**: `EMBEDDING_MODEL` (default: `intfloat/e5-large-v2`).
- **Chunking**:
  - `CHUNKING_STRATEGY` (e.g., `recursive`, `fixed`).
  - `CHUNK_SIZE`, `CHUNK_OVERLAP`.
- **Vector backend**: `VECTOR_STORE_BACKEND` (`s3`, `chromadb`, `pgvector`).

### Running the Pipeline

```bash
source .venv/bin/activate
python -m src.main \
  --excel-path data/social_media_data.xlsx \
  --sheet-name Sheet1
```

Optional flags:

- `--backend s3|chromadb|pgvector`
- `--chunking-strategy recursive|fixed`

### Notes

- The AWS S3 backend in this scaffold stores embeddings as serialized numpy arrays and a simple JSON index. It is designed as a **starting point**; you can adapt it to any specific "S3 vector engine" you use.
- The pgvector backend assumes the `vector` extension is available on the target database.


