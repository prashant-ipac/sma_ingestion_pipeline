"""
AWS S3-based vector store.

This implementation stores:
  - Embeddings as `.npy` blobs under a configurable prefix.
  - A lightweight JSON index mapping ids to metadata.

It is intended as a scaffold that you can adapt to a specific
AWS "S3 vector engine" or search layer later.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Iterable, List, Mapping, Sequence

import boto3
import numpy as np

from .base import VectorStore
from ..data_formatter import create_payload
from ..logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class S3VectorIndexEntry:
    id: str
    text: str
    embedding_key: str
    payload: Mapping[str, object]


class S3VectorStore(VectorStore):
    def __init__(
        self,
        bucket_name: str,
        region_name: str,
        embeddings_prefix: str = "embeddings/",
        index_key: str = "embeddings/index.json",
    ) -> None:
        self.bucket_name = bucket_name
        self.embeddings_prefix = embeddings_prefix.rstrip("/") + "/"
        self.index_key = index_key
        self.s3 = boto3.client("s3", region_name=region_name)

    def _load_index(self) -> List[S3VectorIndexEntry]:
        try:
            logger.info("Loading S3 vector index from s3://%s/%s", self.bucket_name, self.index_key)
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=self.index_key)
            data = obj["Body"].read().decode("utf-8")
            raw = json.loads(data)
            return [
                S3VectorIndexEntry(
                    id=entry["id"],
                    text=entry["text"],
                    embedding_key=entry["embedding_key"],
                    payload=entry.get("payload", entry.get("metadata", {})),  # Support both formats
                )
                for entry in raw
            ]
        except self.s3.exceptions.NoSuchKey:
            logger.info("No existing S3 index found; starting fresh.")
            return []
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load S3 index: %s", exc)
            return []

    def _save_index(self, entries: List[S3VectorIndexEntry]) -> None:
        payload = json.dumps([asdict(e) for e in entries]).encode("utf-8")
        logger.info(
            "Saving S3 vector index with %d entries to s3://%s/%s",
            len(entries),
            self.bucket_name,
            self.index_key,
        )
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=self.index_key,
            Body=payload,
            ContentType="application/json",
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Iterable[Mapping[str, object]] | None = None,
        payloads: Iterable[Mapping[str, object]] | None = None,
        ids: Iterable[str] | None = None,
    ) -> None:
        """
        Add embeddings to S3 with structured payload format.

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

        index_entries = self._load_index()

        logger.info(
            "Uploading %d embeddings to S3 bucket '%s' (prefix=%s)",
            len(embeddings),
            self.bucket_name,
            self.embeddings_prefix,
        )

        for embedding, text, payload, entry_id in zip(embeddings, texts, payloads, ids):
            emb_key = f"{self.embeddings_prefix}{entry_id}.npy"
            buffer = BytesIO()
            np.save(buffer, np.asarray(embedding))
            buffer.seek(0)

            self.s3.upload_fileobj(buffer, self.bucket_name, emb_key)

            index_entries.append(
                S3VectorIndexEntry(
                    id=entry_id,
                    text=text,
                    embedding_key=emb_key,
                    payload=payload,
                )
            )

        self._save_index(index_entries)


