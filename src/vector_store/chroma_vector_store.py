"""
ChromaDB vector store implementation.
"""

from __future__ import annotations

import uuid
from typing import Iterable, List, Mapping, Sequence

import chromadb
import numpy as np

from .base import VectorStore
from ..data_formatter import create_vector_store_entry
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
        payloads: Iterable[Mapping[str, object]] | None = None,
        ids: Iterable[str] | None = None,
    ) -> None:
        """
        Add embeddings to ChromaDB with structured payload format.

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
            # Create minimal payloads from legacy metadata
            from ..data_formatter import create_payload
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

        total = len(texts)

        logger.info(
            "Adding %d embeddings to Chroma collection '%s' at '%s' (max_batch_size=%d)",
            total,
            self.collection_name,
            self.path,
            self.max_batch_size,
        )

        for batch_start in range(0, total, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, total)
            batch_ids = ids[batch_start:batch_end]
            batch_embeddings = [
                np.asarray(e).tolist() for e in embeddings[batch_start:batch_end]
            ]
            batch_texts = list(texts[batch_start:batch_end])
            # Store the full payload in metadata
            batch_metadatas = [
                {"payload": payload} for payload in payloads[batch_start:batch_end]
            ]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )

    def get_all_embeddings(
        self,
        include_texts: bool = True,
        include_metadatas: bool = True,
    ) -> dict:
        """
        Retrieve all embeddings from the ChromaDB collection.

        Args:
            include_texts: Whether to include texts in the result
            include_metadatas: Whether to include metadatas in the result

        Returns:
            Dictionary with keys:
                - 'embeddings': numpy array of embeddings
                - 'ids': list of IDs
                - 'texts': list of texts (if include_texts=True)
                - 'metadatas': list of metadatas (if include_metadatas=True)
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("Collection is empty, returning empty results")
            result = {
                "embeddings": np.array([]),
                "ids": [],
            }
            if include_texts:
                result["texts"] = []
            if include_metadatas:
                result["metadatas"] = []
            return result

        logger.info(
            "Retrieving all %d embeddings from Chroma collection '%s'",
            count,
            self.collection_name,
        )

        # Get all data from ChromaDB
        results = self.collection.get(include=["embeddings", "documents", "metadatas"])

        # Convert embeddings to numpy array
        embeddings_list = results.get("embeddings", [])
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        result = {
            "embeddings": embeddings_array,
            "ids": results.get("ids", []),
        }

        if include_texts:
            result["texts"] = results.get("documents", [])

        if include_metadatas:
            result["metadatas"] = results.get("metadatas", [])

        logger.info("Retrieved %d embeddings successfully", len(embeddings_array))
        return result


