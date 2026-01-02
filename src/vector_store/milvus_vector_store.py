"""
Milvus vector store implementation.
"""

from __future__ import annotations

import json
import uuid
from typing import Iterable, Mapping, Sequence

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .base import VectorStore
from ..data_formatter import create_payload
from ..logging_utils import get_logger


logger = get_logger(__name__)


class MilvusVectorStore(VectorStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "social_media_embeddings",
        embedding_dim: int = 1024,
        max_batch_size: int = 5000,
        user: str | None = None,
        password: str | None = None,
    ):
        """
        Initialize Milvus vector store.

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to store embeddings
            embedding_dim: Dimension of the embedding vectors
            max_batch_size: Maximum batch size for inserting embeddings
            user: Milvus username for authentication
            password: Milvus password for authentication
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.max_batch_size = max_batch_size

        # Connect to Milvus with authentication if provided
        connect_params = {
            "alias": "default",
            "host": host,
            "port": port,
        }
        if user and password:
            connect_params["user"] = user
            connect_params["password"] = password

        connections.connect(**connect_params)

        # Create or get collection
        self.collection = self._ensure_collection()

    def _ensure_collection(self) -> Collection:
        """Create collection if it doesn't exist, otherwise return existing collection."""
        if utility.has_collection(self.collection_name):
            logger.info("Using existing Milvus collection '%s'", self.collection_name)
            collection = Collection(self.collection_name)
        else:
            logger.info("Creating new Milvus collection '%s'", self.collection_name)
            # Define schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name="payload",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Social media embeddings collection",
            )

            collection = Collection(
                name=self.collection_name,
                schema=schema,
            )

            # Create index on embedding field for efficient similarity search
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params,
            )
            logger.info("Created index on embedding field")

        return collection

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Iterable[Mapping[str, object]] | None = None,
        payloads: Iterable[Mapping[str, object]] | None = None,
        ids: Iterable[str] | None = None,
    ) -> None:
        """
        Add embeddings to Milvus with structured payload format.

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
            "Adding %d embeddings to Milvus collection '%s' at %s:%d (max_batch_size=%d)",
            total,
            self.collection_name,
            self.host,
            self.port,
            self.max_batch_size,
        )

        # Prepare data for insertion
        # Milvus expects data as a list of lists where each inner list represents a field
        ids_list = [str(id_val) for id_val in ids]
        texts_list = [str(text) for text in texts]
        payloads_list = [json.dumps(payload, ensure_ascii=False) for payload in payloads]
        embeddings_list = [
            np.asarray(e, dtype=np.float32).tolist() for e in embeddings
        ]

        # Insert in batches
        for batch_start in range(0, total, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, total)
            batch_ids = ids_list[batch_start:batch_end]
            batch_texts = texts_list[batch_start:batch_end]
            batch_payloads = payloads_list[batch_start:batch_end]
            batch_embeddings = embeddings_list[batch_start:batch_end]

            # Milvus expects data in field order: [ids, texts, payloads, embeddings]
            data = [
                batch_ids,
                batch_texts,
                batch_payloads,
                batch_embeddings,
            ]

            self.collection.insert(data)
            logger.debug(
                "Inserted batch %d-%d into Milvus collection",
                batch_start,
                batch_end - 1,
            )

        # Flush to ensure data is persisted
        self.collection.flush()
        logger.info("Successfully added %d embeddings to Milvus", total)

    def get_all_embeddings(
        self,
        include_texts: bool = True,
        include_metadatas: bool = True,
    ) -> dict:
        """
        Retrieve all embeddings from the Milvus collection.

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
        # Load collection into memory for querying
        self.collection.load()

        # Query all data - use an expression that matches all entities
        # In Milvus, we can use "id != ''" to match all entities
        results = self.collection.query(
            expr='id != ""',  # Expression that matches all entities
            output_fields=["id", "text", "payload", "embedding"],
        )

        if not results:
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

        logger.info("Retrieved %d embeddings from Milvus collection", len(results))

        # Extract data from results
        ids = [r["id"] for r in results]
        embeddings_list = [r["embedding"] for r in results]
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        result = {
            "embeddings": embeddings_array,
            "ids": ids,
        }

        if include_texts:
            result["texts"] = [r["text"] for r in results]

        if include_metadatas:
            # Deserialize JSON payload strings back to dictionaries
            metadatas_parsed = []
            for r in results:
                payload_str = r.get("payload", "")
                if payload_str:
                    try:
                        payload_dict = json.loads(payload_str)
                        metadatas_parsed.append({"payload": payload_dict})
                    except (json.JSONDecodeError, TypeError):
                        metadatas_parsed.append({"payload": {}})
                else:
                    metadatas_parsed.append({"payload": {}})
            result["metadatas"] = metadatas_parsed

        logger.info("Retrieved %d embeddings successfully", len(embeddings_array))
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
        # Load collection into memory for querying
        self.collection.load()

        # Milvus doesn't support complex JSON queries directly, so we need to:
        # 1. Query all data
        # 2. Filter in Python based on payload content

        logger.info(
            "Querying Milvus collection '%s' with timestamp filter (year=%s, month=%s, day=%s, start_epoch=%s, end_epoch=%s)",
            self.collection_name,
            year,
            month,
            day,
            start_epoch,
            end_epoch,
        )

        # Query all data (Milvus doesn't support JSON field filtering directly)
        # Use an expression that matches all entities, then filter in Python
        query_limit = limit * 10 if limit else None
        all_results = self.collection.query(
            expr='id != ""',  # Expression that matches all entities
            output_fields=["id", "text", "payload", "embedding"],
            limit=query_limit,
        )

        if not all_results:
            logger.warning("Collection is empty, returning empty results")
            return {
                "embeddings": np.array([]),
                "ids": [],
                "texts": [],
                "metadatas": [],
            }

        # Filter results based on timestamp criteria
        filtered_results = []
        for r in all_results:
            payload_str = r.get("payload", "")
            if not payload_str:
                continue

            try:
                payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
                timestamp = payload.get("timestamp", {})

                # Check filters
                match = True
                if year is not None and timestamp.get("year") != year:
                    match = False
                if match and month is not None and timestamp.get("month") != month:
                    match = False
                if match and day is not None and timestamp.get("day") != day:
                    match = False
                if match and start_epoch is not None:
                    epoch = timestamp.get("epoch")
                    if epoch is None or int(epoch) < start_epoch:
                        match = False
                if match and end_epoch is not None:
                    epoch = timestamp.get("epoch")
                    if epoch is None or int(epoch) > end_epoch:
                        match = False

                if match:
                    filtered_results.append(r)
                    if limit and len(filtered_results) >= limit:
                        break
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        # Convert to result format
        if not filtered_results:
            return {
                "embeddings": np.array([]),
                "ids": [],
                "texts": [],
                "metadatas": [],
            }

        ids = [r["id"] for r in filtered_results]
        texts = [r["text"] for r in filtered_results]
        embeddings_list = [r["embedding"] for r in filtered_results]
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        # Deserialize JSON payload strings
        metadatas = []
        for r in filtered_results:
            payload_str = r.get("payload", "")
            if payload_str:
                try:
                    payload_dict = json.loads(payload_str)
                    metadatas.append({"payload": payload_dict})
                except (json.JSONDecodeError, TypeError):
                    metadatas.append({"payload": {}})
            else:
                metadatas.append({"payload": {}})

        result = {
            "embeddings": embeddings_array,
            "ids": ids,
            "texts": texts,
            "metadatas": metadatas,
        }

        logger.info("Retrieved %d embeddings matching timestamp filter", len(ids))
        return result

