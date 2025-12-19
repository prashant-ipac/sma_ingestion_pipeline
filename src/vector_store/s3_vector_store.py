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
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Iterable, List, Mapping, Sequence

import boto3
import numpy as np

from .base import VectorStore
from ..logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class S3VectorIndexEntry:
    id: str
    text: str
    embedding_key: str
    metadata: Mapping[str, object]


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
                    metadata=entry.get("metadata", {}),
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
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        metadatas_list = list(metadatas)

        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have the same length.")

        index_entries = self._load_index()
        start_idx = len(index_entries)

        logger.info(
            "Uploading %d embeddings to S3 bucket '%s' (prefix=%s)",
            len(embeddings),
            self.bucket_name,
            self.embeddings_prefix,
        )

        for offset, (embedding, text, metadata) in enumerate(
            zip(embeddings, texts, metadatas_list)
        ):
            idx = start_idx + offset
            emb_key = f"{self.embeddings_prefix}{idx}.npy"
            buffer = BytesIO()
            np.save(buffer, np.asarray(embedding))
            buffer.seek(0)

            self.s3.upload_fileobj(buffer, self.bucket_name, emb_key)

            index_entries.append(
                S3VectorIndexEntry(
                    id=str(idx),
                    text=text,
                    embedding_key=emb_key,
                    metadata=metadata,
                )
            )

        self._save_index(index_entries)


