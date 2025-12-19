"""
pgvector-backed PostgreSQL vector store.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from .base import VectorStore
from ..logging_utils import get_logger


logger = get_logger(__name__)


class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, table_name: str, embedding_dim: int) -> None:
        self.dsn = dsn
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = True
        cur = conn.cursor()
        logger.info("Ensuring pgvector extension and table '%s' exist", self.table_name)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector({self.embedding_dim}) NOT NULL
            );
            """
        )
        cur.close()
        conn.close()

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Iterable[Mapping[str, object]] | None = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        metadatas_list = list(metadatas)

        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have the same length.")

        logger.info(
            "Inserting %d embeddings into table '%s' via pgvector",
            len(embeddings),
            self.table_name,
        )

        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()

        records = []
        for embedding, text, metadata in zip(embeddings, texts, metadatas_list):
            vec = np.asarray(embedding, dtype=float)
            records.append(
                (
                    text,
                    metadata,
                    list(vec),
                )
            )

        execute_values(
            cur,
            f"INSERT INTO {self.table_name} (text, metadata, embedding) VALUES %s",
            records,
        )

        conn.commit()
        cur.close()
        conn.close()


