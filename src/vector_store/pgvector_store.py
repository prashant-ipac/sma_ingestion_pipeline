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

    def query_by_timestamp(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        start_epoch: int | None = None,
        end_epoch: int | None = None,
        limit: int | None = None,
    ) -> dict:
        """
        Query embeddings filtered by timestamp.

        Args:
            year: Filter by year
            month: Filter by month (1-12)
            day: Filter by day (1-31)
            start_epoch: Filter by start epoch timestamp (inclusive)
            end_epoch: Filter by end epoch timestamp (inclusive)
            limit: Maximum number of results to return

        Returns:
            Dictionary with keys: 'ids', 'embeddings', 'texts', 'metadatas'
        """
        import json

        # Build SQL WHERE clause
        conditions = []
        params = []

        if year is not None:
            conditions.append("payload->'timestamp'->>'year' = %s")
            params.append(str(year))

        if month is not None:
            conditions.append("payload->'timestamp'->>'month' = %s")
            params.append(str(month))

        if day is not None:
            conditions.append("payload->'timestamp'->>'day' = %s")
            params.append(str(day))

        if start_epoch is not None:
            conditions.append("(payload->'timestamp'->>'epoch')::bigint >= %s")
            params.append(start_epoch)

        if end_epoch is not None:
            conditions.append("(payload->'timestamp'->>'epoch')::bigint <= %s")
            params.append(end_epoch)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT id, text, payload, embedding
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY id
        """
        
        if limit:
            query += f" LIMIT {limit}"

        logger.info(
            "Querying pgvector table '%s' with timestamp filter: %s",
            self.table_name,
            conditions,
        )

        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Convert results to dict format
        ids = [str(row[0]) for row in rows]
        texts = [row[1] for row in rows]
        payloads = [row[2] for row in rows]
        embeddings_list = [row[3] for row in rows]

        embeddings_array = np.array(embeddings_list, dtype=np.float32) if embeddings_list else np.array([])

        result = {
            "embeddings": embeddings_array,
            "ids": ids,
            "texts": texts,
            "metadatas": [{"payload": payload} for payload in payloads],
        }

        logger.info("Retrieved %d embeddings matching timestamp filter", len(ids))
        return result


