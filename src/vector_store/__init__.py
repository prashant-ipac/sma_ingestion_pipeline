"""
Vector store backend factory and exports.
"""

from .base import VectorStore
from .s3_vector_store import S3VectorStore
from .chroma_vector_store import ChromaVectorStore
from .pgvector_store import PgVectorStore

__all__ = [
    "VectorStore",
    "S3VectorStore",
    "ChromaVectorStore",
    "PgVectorStore",
]


