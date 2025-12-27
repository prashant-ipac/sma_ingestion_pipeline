"""
pgvector-backed PostgreSQL vector store.
"""

from __future__ import annotations

import json
import uuid
from typing import Iterable, Mapping, Sequence

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from .base import VectorStore
from ..data_formatter import create_payload
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
                id UUID PRIMARY KEY,
                text TEXT NOT NULL,
                payload JSONB,
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
        payloads: Iterable[Mapping[str, object]] | None = None,
        ids: Iterable[str] | None = None,
    ) -> None:
        """
        Add embeddings to pgvector with structured payload format.

        Args:
            embeddings: Numpy array of embeddings
            texts: Sequence of text strings
            metadatas: Legacy metadata format (for backward compatibility)
            payloads: New structured payload format (preferred)
            ids: Optional list of UUID strings. If not provided, UUIDs will be generated.
        """
        if payloads is None:
            # Legacy mode: convert simple metadatas to payload format
            if metadatas is None:
                metadatas = [{} for _ in texts]
            metadatas_list = list(metadatas)
            payloads = [
                create_payload(
                    text=text,
                    ingested_from=meta.get("source", "excel"),
                    file_name=meta.get("file_name", ""),
                    row_number=meta.get("index", meta.get("row_number", 0)),
                )
                for text, meta in zip(texts, metadatas_list)
            ]
        else:
            payloads = list(payloads)

        if len(embeddings) != len(texts) or len(embeddings) != len(payloads):
            raise ValueError("Embeddings, texts, and payloads must have the same length.")

        # Generate UUIDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = list(ids)

        logger.info(
            "Inserting %d embeddings into table '%s' via pgvector",
            len(embeddings),
            self.table_name,
        )

        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()

        records = []
        for embedding, text, payload, entry_id in zip(embeddings, texts, payloads, ids):
            vec = np.asarray(embedding, dtype=float)
            records.append(
                (
                    entry_id,
                    text,
                    json.dumps(payload),
                    list(vec),
                )
            )

        execute_values(
            cur,
            f"INSERT INTO {self.table_name} (id, text, payload, embedding) VALUES %s",
            records,
        )

        conn.commit()
        cur.close()
        conn.close()


