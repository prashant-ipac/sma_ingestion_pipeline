"""
Configuration management for the social_media_vectordb project.

Values are primarily sourced from environment variables (.env supported).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List, Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv

from .constants import (
    SUPPORTED_BACKENDS,
    SUPPORTED_CHUNKING_STRATEGIES,
    SUPPORTED_EMBEDDING_PROVIDERS,
    DEFAULT_TEXT_COLUMNS,
    DEFAULT_EXCEL_SHEET_NAME,
    DEFAULT_EMBEDDING_DIM,
)

# Load .env once at import time
load_dotenv()


def _get_env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [part.strip() for part in raw.split(",") if part.strip()]


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "y", "on")


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class Config:
    # -----------------------
    # Embeddings (common)
    # -----------------------
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    embedding_dim: int = _get_env_int("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    embedding_batch_size: int = _get_env_int("EMBEDDING_BATCH_SIZE", 32)

    # SentenceTransformers-specific
    embedding_use_onnx: bool = _get_env_bool("EMBEDDING_USE_ONNX", False)
    embedding_device: str | None = os.getenv("EMBEDDING_DEVICE", None)

    # Voyage-specific
    voyage_api_key: str | None = os.getenv("VOYAGE_API_KEY", None)
    voyage_input_type: str | None = os.getenv("VOYAGE_INPUT_TYPE", None)  # query|document|None
    voyage_truncation: bool | None = (
        None if os.getenv("VOYAGE_TRUNCATION") is None else _get_env_bool("VOYAGE_TRUNCATION", True)
    )
    voyage_output_dimension: int | None = (
        None if os.getenv("VOYAGE_OUTPUT_DIMENSION") is None else _get_env_int("VOYAGE_OUTPUT_DIMENSION", 1024)
    )
    voyage_output_dtype: str = os.getenv("VOYAGE_OUTPUT_DTYPE", "float")  # float|int8|uint8|binary|ubinary
    voyage_timeout: int | None = (
        None if os.getenv("VOYAGE_TIMEOUT") is None else _get_env_int("VOYAGE_TIMEOUT", 60)
    )
    voyage_max_retries: int = _get_env_int("VOYAGE_MAX_RETRIES", 2)

    # -----------------------
    # Chunking
    # -----------------------
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "recursive")
    chunk_size: int = _get_env_int("CHUNK_SIZE", 512)
    chunk_overlap: int = _get_env_int("CHUNK_OVERLAP", 50)

    # -----------------------
    # Vector store backend
    # -----------------------
    backend: str = os.getenv("VECTOR_STORE_BACKEND", "s3")

    # -----------------------
    # AWS S3 backend
    # -----------------------
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_s3_bucket_name: str = os.getenv("AWS_S3_BUCKET_NAME", "")
    aws_s3_embeddings_prefix: str = os.getenv("AWS_S3_EMBEDDINGS_PREFIX", "embeddings/")
    aws_s3_index_key: str = os.getenv("AWS_S3_INDEX_KEY", "embeddings/index.json")

    # -----------------------
    # ChromaDB backend
    # -----------------------
    chromadb_path: str = os.getenv("CHROMADB_PATH", "./chromadb_store")

    # -----------------------
    # pgvector backend
    # -----------------------
    pgvector_dsn: str = os.getenv(
        "PGVECTOR_DSN", "postgresql://user:password@localhost:5432/social_vectors"
    )
    pgvector_table_name: str = os.getenv("PGVECTOR_TABLE_NAME", "embeddings")

    # -----------------------
    # Milvus backend
    # -----------------------
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = _get_env_int("MILVUS_PORT", 19530)
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "social_media_embeddings")
    milvus_user: str = os.getenv("MILVUS_USER", "")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "")

    # -----------------------
    # AtlasDB / MongoDB
    # -----------------------
    # Recommended: supply full ATLASDB_URI in .env
    atlasdb_uri: str = os.getenv("ATLASDB_URI", "")
    atlasdb_database_name: str = os.getenv("ATLASDB_DATABASE_NAME", "socialmediaanalytics")
    atlasdb_collection_name: str = os.getenv("ATLASDB_COLLECTION_NAME", "instagram")
    atlasdb_embedding_dim: int = _get_env_int("ATLASDB_EMBEDDING_DIM", 1024)
    atlasdb_max_batch_size: int = _get_env_int("ATLASDB_MAX_BATCH_SIZE", 1000)
    atlasdb_index_name: str = os.getenv("ATLASDB_INDEX_NAME", "vector_index")

    # Optional helpers (only used to build URI if ATLASDB_URI is not set)
    atlasdb_username: str | None = os.getenv("ATLASDB_USERNAME", None)
    atlasdb_password: str | None = os.getenv("ATLASDB_PASSWORD", None)
    atlasdb_cluster_host: str = os.getenv("ATLASDB_CLUSTER_HOST", "cluster0.mongodb.net")

    # -----------------------
    # Logging
    # -----------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # -----------------------
    # Excel / data loading
    # -----------------------
    default_sheet_name: str = os.getenv("EXCEL_SHEET_NAME", DEFAULT_EXCEL_SHEET_NAME)
    text_columns: List[str] = field(
        default_factory=lambda: _get_env_list("TEXT_COLUMNS", DEFAULT_TEXT_COLUMNS)
    )

    def finalize(self) -> None:
        """
        Post-processing (e.g., construct atlasdb_uri if not explicitly set).
        Call this once after instantiating Config.
        """
        if not self.atlasdb_uri and self.atlasdb_username and self.atlasdb_password:
            u = quote_plus(self.atlasdb_username)
            p = quote_plus(self.atlasdb_password)
            self.atlasdb_uri = (
                f"mongodb+srv://{u}:{p}@{self.atlasdb_cluster_host}/"
                f"?retryWrites=true&w=majority"
            )

    def validate(self) -> None:
        # Embedding provider validation
        if self.embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Unsupported embedding provider '{self.embedding_provider}'. "
                f"Supported: {SUPPORTED_EMBEDDING_PROVIDERS}"
            )

        if self.embedding_provider == "voyage":
            if not self.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY must be set when EMBEDDING_PROVIDER=voyage.")
            if self.voyage_input_type not in (None, "query", "document"):
                raise ValueError("VOYAGE_INPUT_TYPE must be one of: query, document, or empty.")
            allowed_dtypes = {"float", "int8", "uint8", "binary", "ubinary"}
            if self.voyage_output_dtype not in allowed_dtypes:
                raise ValueError(f"VOYAGE_OUTPUT_DTYPE must be one of: {sorted(allowed_dtypes)}")
            if self.embedding_batch_size > 1000:
                raise ValueError("EMBEDDING_BATCH_SIZE must be <= 1000 for Voyage.")

        # Backend validation
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. Supported backends: {SUPPORTED_BACKENDS}"
            )

        # Chunking validation
        if self.chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
            raise ValueError(
                f"Unsupported chunking strategy '{self.chunking_strategy}'. "
                f"Supported strategies: {SUPPORTED_CHUNKING_STRATEGIES}"
            )

        # Backend-specific required fields
        if self.backend == "s3" and not self.aws_s3_bucket_name:
            raise ValueError("AWS_S3_BUCKET_NAME must be set when using the S3 backend.")

        if self.backend == "atlasdb" and not self.atlasdb_uri:
            raise ValueError(
                "ATLASDB_URI must be set when using atlasdb backend "
                "(or set ATLASDB_USERNAME + ATLASDB_PASSWORD + ATLASDB_CLUSTER_HOST)."
            )

    def summary(self) -> dict:
        # Avoid logging secrets like VOYAGE_API_KEY or ATLASDB password.
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_device": self.embedding_device,
            "embedding_use_onnx": self.embedding_use_onnx,
            "voyage_input_type": self.voyage_input_type,
            "voyage_truncation": self.voyage_truncation,
            "voyage_output_dimension": self.voyage_output_dimension,
            "voyage_output_dtype": self.voyage_output_dtype,
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "backend": self.backend,
            "atlasdb_database_name": self.atlasdb_database_name,
            "atlasdb_collection_name": self.atlasdb_collection_name,
            "atlasdb_uri_set": bool(self.atlasdb_uri),
            "aws_s3_bucket_name_set": bool(self.aws_s3_bucket_name),
            "chromadb_path": self.chromadb_path,
            "pgvector_table_name": self.pgvector_table_name,
            "milvus_collection_name": self.milvus_collection_name,
            "log_level": self.log_level,
            "default_sheet_name": self.default_sheet_name,
            "text_columns": self.text_columns,
        }


