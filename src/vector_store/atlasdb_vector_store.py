"""
MongoDB Atlas vector store implementation.
"""

from __future__ import annotations

import json
import uuid
from typing import Iterable, Mapping, Sequence

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection

from .base import VectorStore
from ..data_formatter import create_payload
from ..logging_utils import get_logger


logger = get_logger(__name__)


class AtlasVectorStore(VectorStore):
    def __init__(
        self,
        uri: str,
        database_name: str = "vector_db",
        collection_name: str = "social_media_embeddings",
        embedding_dim: int = 1024,
        max_batch_size: int = 1000,
        index_name: str = "vector_index",
    ):
        """
        Initialize MongoDB Atlas vector store.

        Args:
            uri: MongoDB Atlas connection string
            database_name: Database name
            collection_name: Collection name
            embedding_dim: Dimension of the embedding vectors
            max_batch_size: Maximum batch size for inserting embeddings
            index_name: Atlas Vector Search index name
        """
        self.uri = uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.max_batch_size = max_batch_size
        self.index_name = index_name

        self.client = MongoClient(self.uri)
        self.db = self.client[self.database_name]
        self.collection: Collection = self.db[self.collection_name]

        logger.info(
            "Connected to MongoDB Atlas database '%s', collection '%s'",
            self.database_name,
            self.collection_name,
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
        Add embeddings to MongoDB Atlas with structured payload format.
        """
        if payloads is None:
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

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = list(ids)

        total = len(texts)

        logger.info(
            "Adding %d embeddings to Atlas collection '%s' (max_batch_size=%d)",
            total,
            self.collection_name,
            self.max_batch_size,
        )

        documents = []
        for i in range(total):
            documents.append(
                {
                    "_id": ids[i],
                    "text": texts[i],
                    "payload": payloads[i],
                    "embedding": np.asarray(embeddings[i], dtype=np.float32).tolist(),
                }
            )

        for batch_start in range(0, total, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, total)
            batch_docs = documents[batch_start:batch_end]

            self.collection.insert_many(batch_docs, ordered=False)
            logger.debug(
                "Inserted batch %d-%d into Atlas collection",
                batch_start,
                batch_end - 1,
            )

        logger.info("Successfully added %d embeddings to MongoDB Atlas", total)

    def get_all_embeddings(
        self,
        include_texts: bool = True,
        include_metadatas: bool = True,
    ) -> dict:
        """
        Retrieve all embeddings from the MongoDB Atlas collection.
        """
        cursor = self.collection.find({})

        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for doc in cursor:
            ids.append(doc["_id"])
            embeddings.append(doc["embedding"])

            if include_texts:
                texts.append(doc.get("text", ""))

            if include_metadatas:
                metadatas.append({"payload": doc.get("payload", {})})

        if not ids:
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

        logger.info("Retrieved %d embeddings from Atlas collection", len(ids))

        result = {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "ids": ids,
        }

        if include_texts:
            result["texts"] = texts
        if include_metadatas:
            result["metadatas"] = metadatas

        return result

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
        Query embeddings filtered by timestamp fields stored in payload.
        """
        logger.info(
            "Querying Atlas collection '%s' with timestamp filter",
            self.collection_name,
        )

        mongo_filter = {}

        if year is not None:
            mongo_filter["payload.timestamp.year"] = year
        if month is not None:
            mongo_filter["payload.timestamp.month"] = month
        if day is not None:
            mongo_filter["payload.timestamp.day"] = day

        if start_epoch is not None or end_epoch is not None:
            epoch_filter = {}
            if start_epoch is not None:
                epoch_filter["$gte"] = start_epoch
            if end_epoch is not None:
                epoch_filter["$lte"] = end_epoch
            mongo_filter["payload.timestamp.epoch"] = epoch_filter

        cursor = self.collection.find(mongo_filter).limit(limit or 0)

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for doc in cursor:
            ids.append(doc["_id"])
            texts.append(doc.get("text", ""))
            embeddings.append(doc["embedding"])
            metadatas.append({"payload": doc.get("payload", {})})

        if not ids:
            return {
                "embeddings": np.array([]),
                "ids": [],
                "texts": [],
                "metadatas": [],
            }

        logger.info("Retrieved %d embeddings matching timestamp filter", len(ids))

        return {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "ids": ids,
            "texts": texts,
            "metadatas": metadatas,
        }
