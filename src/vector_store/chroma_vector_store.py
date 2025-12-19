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
    def __init__(self, path: str, collection_name: str = "social_media_embeddings"):
        self.path = path
        self.collection_name = collection_name
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

        ids = [str(i) for i in range(self.collection.count(), self.collection.count() + len(texts))]

        logger.info(
            "Adding %d embeddings to Chroma collection '%s' at '%s'",
            len(embeddings),
            self.collection_name,
            self.path,
        )

        self.collection.add(
            ids=ids,
            embeddings=[np.asarray(e).tolist() for e in embeddings],
            documents=list(texts),
            metadatas=metadatas_list,
        )


