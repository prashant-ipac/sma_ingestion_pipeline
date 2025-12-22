"""
ChromaDB vector store implementation.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import chromadb
import numpy as np

from .base import VectorStore
from ..logging_utils import get_logger


logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        path: str,
        collection_name: str = "social_media_embeddings",
        max_batch_size: int = 5000,
    ):
        self.path = path
        self.collection_name = collection_name
        self.max_batch_size = max_batch_size
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

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

        total = len(texts)
        start_idx = self.collection.count()

        logger.info(
            "Adding %d embeddings to Chroma collection '%s' at '%s' (max_batch_size=%d)",
            total,
            self.collection_name,
            self.path,
            self.max_batch_size,
        )

        for batch_start in range(0, total, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, total)
            ids = [str(start_idx + i) for i in range(batch_start, batch_end)]
            batch_embeddings = [
                np.asarray(e).tolist() for e in embeddings[batch_start:batch_end]
            ]
            batch_texts = list(texts[batch_start:batch_end])
            batch_metadatas = metadatas_list[batch_start:batch_end]

            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )


