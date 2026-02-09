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
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                platform_post_id TEXT,

                state TEXT,
                posted_date DATE,
                username TEXT,
                post_url TEXT,
                video_id TEXT,
                post_type_raw TEXT,
                post_type_norm TEXT,

                text TEXT,
                engagement_total INTEGER,

                embedding vector({self.embedding_dim}),
                embedding_model TEXT,
                embedding_provider TEXT,

                extras JSONB DEFAULT '{{}}'::jsonb,
                ingested_from TEXT DEFAULT 'excel',
                source_file TEXT,
                source_sheet TEXT,
                source_row_number INTEGER,

                created_at TIMESTAMPTZ DEFAULT now()
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
        Add embeddings to pgvector, storing values directly in table columns.

        Args:
            embeddings: Numpy array of embeddings
            texts: Sequence of text strings
            metadatas: Optional metadata (legacy, unused if payloads provided)
            payloads: Optional structured data with column mappings (used for DB column values)
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
        """
        texts_list = list(texts)
        embeddings_array = np.asarray(embeddings)
        
        if len(embeddings_array) != len(texts_list):
            raise ValueError("Embeddings and texts must have the same length.")

        # Use provided payloads or create empty defaults
        if payloads is None:
            payloads = [{} for _ in texts_list]
        else:
            payloads = list(payloads)

        if len(payloads) != len(texts_list):
            raise ValueError("Payloads and texts must have the same length.")

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            ids = list(ids)

        logger.info(
            "Inserting %d embeddings into table '%s' via pgvector",
            len(texts_list),
            self.table_name,
        )

        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()

        records = []
        for embedding, text, payload, entry_id in zip(embeddings_array, texts_list, payloads, ids):
            # Convert embedding to list of native Python floats
            vec = np.asarray(embedding, dtype=float)
            vec_list = [float(v) for v in vec]

            # Extract values from payload dict, use defaults if not present
            platform = payload.get("platform", "unknown") if isinstance(payload, dict) else "unknown"
            platform_post_id = payload.get("platform_post_id") if isinstance(payload, dict) else None
            state = payload.get("state") if isinstance(payload, dict) else None
            posted_date = payload.get("posted_date") if isinstance(payload, dict) else None
            username = payload.get("username") if isinstance(payload, dict) else None
            post_url = payload.get("post_url") if isinstance(payload, dict) else None
            video_id = payload.get("video_id") if isinstance(payload, dict) else None
            post_type_raw = payload.get("post_type_raw") if isinstance(payload, dict) else None
            post_type_norm = payload.get("post_type_norm") if isinstance(payload, dict) else None
            engagement_total = payload.get("engagement_total") if isinstance(payload, dict) else None
            embedding_model = payload.get("embedding_model") if isinstance(payload, dict) else None
            embedding_provider = payload.get("embedding_provider") if isinstance(payload, dict) else None
            extras = payload.get("extras", {}) if isinstance(payload, dict) else {}
            ingested_from = payload.get("ingested_from", "excel") if isinstance(payload, dict) else "excel"
            source_file = payload.get("source_file") if isinstance(payload, dict) else None
            source_sheet = payload.get("source_sheet") if isinstance(payload, dict) else None
            source_row_number = payload.get("source_row_number") if isinstance(payload, dict) else None
            created_at = payload.get("created_at") if isinstance(payload, dict) else None

            # Ensure proper types
            engagement_total = int(engagement_total) if engagement_total is not None else None
            source_row_number = int(source_row_number) if source_row_number is not None else None
            
            # Convert extras to JSON string if it's a dict
            if isinstance(extras, dict):
                extras_json = json.dumps(extras)
            else:
                extras_json = str(extras) if extras else '{}'

            records.append(
                (
                    str(entry_id),
                    str(platform) if platform else "unknown",
                    str(platform_post_id) if platform_post_id else None,
                    str(state) if state else None,
                    str(posted_date) if posted_date else None,
                    str(username) if username else None,
                    str(post_url) if post_url else None,
                    str(video_id) if video_id else None,
                    str(post_type_raw) if post_type_raw else None,
                    str(post_type_norm) if post_type_norm else None,
                    str(text),
                    engagement_total,
                    vec_list,
                    str(embedding_model) if embedding_model else None,
                    str(embedding_provider) if embedding_provider else None,
                    extras_json,
                    str(ingested_from) if ingested_from else "excel",
                    str(source_file) if source_file else None,
                    str(source_sheet) if source_sheet else None,
                    source_row_number,
                    str(created_at) if created_at else None,
                )
            )

        query = f"""
            INSERT INTO {self.table_name} (
                id, platform, platform_post_id, state, posted_date, username,
                post_url, video_id, post_type_raw, post_type_norm, text, engagement_total,
                embedding, embedding_model, embedding_provider, extras, ingested_from,
                source_file, source_sheet, source_row_number, created_at
            ) VALUES %s
        """
        execute_values(cur, query, records)

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


