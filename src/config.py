"""
Configuration management for the social_media_vectordb project.

Values are primarily sourced from environment variables.
"""

from dataclasses import dataclass, field
import os
from typing import List

from dotenv import load_dotenv

from .constants import (
    SUPPORTED_BACKENDS,
    SUPPORTED_CHUNKING_STRATEGIES,
    DEFAULT_TEXT_COLUMNS,
    DEFAULT_EXCEL_SHEET_NAME,
    DEFAULT_EMBEDDING_DIM,
)


load_dotenv()


def _get_env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass
class Config:
    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))
    embedding_use_onnx: bool = os.getenv("EMBEDDING_USE_ONNX", "false").lower() == "true"
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_device: str | None = os.getenv("EMBEDDING_DEVICE", None)

    # Chunking
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "recursive")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Vector store backend
    backend: str = os.getenv("VECTOR_STORE_BACKEND", "s3")

    # AWS S3 backend configuration
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_s3_bucket_name: str = os.getenv("AWS_S3_BUCKET_NAME", "")
    aws_s3_embeddings_prefix: str = os.getenv(
        "AWS_S3_EMBEDDINGS_PREFIX", "embeddings/"
    )
    aws_s3_index_key: str = os.getenv("AWS_S3_INDEX_KEY", "embeddings/index.json")

    # ChromaDB backend configuration
    chromadb_path: str = os.getenv("CHROMADB_PATH", "./chromadb_store")

    # pgvector backend configuration
    pgvector_dsn: str = os.getenv(
        "PGVECTOR_DSN", "postgresql://user:password@localhost:5432/social_vectors"
    )
    pgvector_table_name: str = os.getenv("PGVECTOR_TABLE_NAME", "embeddings")

    # Milvus backend configuration
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "social_media_embeddings")
    milvus_user: str = os.getenv("MILVUS_USER", "db_9bc9fd3f17d2fc4")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "Xt6+sy!)+g[<vVkE")

    #atlasdb backend configuration
    atlasdb_uri: str = os.getenv("ATLASDB_URI", "mongodb://localhost:27017")
    atlasdb_database_name: str = os.getenv("ATLASDB_DATABASE_NAME", "social_media_embeddings")
    atlasdb_collection_name: str = os.getenv("ATLASDB_COLLECTION_NAME", "embeddings")
    atlasdb_embedding_dim: int = int(os.getenv("ATLASDB_EMBEDDING_DIM", "1024"))
    atlasdb_max_batch_size: int = int(os.getenv("ATLASDB_MAX_BATCH_SIZE", "1000"))
    atlasdb_index_name: str = os.getenv("ATLASDB_INDEX_NAME", "embeddings_index")


    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Excel / data loading
    default_sheet_name: str = os.getenv(
        "EXCEL_SHEET_NAME", DEFAULT_EXCEL_SHEET_NAME
    )
    # Use default_factory to avoid mutable default issues with dataclasses
    text_columns: List[str] = field(
        default_factory=lambda: _get_env_list("TEXT_COLUMNS", DEFAULT_TEXT_COLUMNS)
    )

    def validate(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. "
                f"Supported backends: {SUPPORTED_BACKENDS}"
            )

        if self.chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
            raise ValueError(
                f"Unsupported chunking strategy '{self.chunking_strategy}'. "
                f"Supported strategies: {SUPPORTED_CHUNKING_STRATEGIES}"
            )

        if self.backend == "s3" and not self.aws_s3_bucket_name:
            raise ValueError(
                "AWS_S3_BUCKET_NAME must be set when using the S3 backend."
            )


