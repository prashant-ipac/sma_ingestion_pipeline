"""
ChromaDB vector store implementation.
"""

from __future__ import annotations

import json
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
            # Store the full payload as JSON string in metadata (ChromaDB only supports flat primitives)
            # Also extract timestamp fields for filtering
            batch_metadatas = []
            for payload in payloads[batch_start:batch_end]:
                meta = {"payload": json.dumps(payload, ensure_ascii=False)}
                # Extract timestamp fields for filtering
                if "timestamp" in payload and isinstance(payload["timestamp"], dict):
                    ts = payload["timestamp"]
                    if "epoch" in ts:
                        meta["timestamp_epoch"] = int(ts["epoch"])
                    if "year" in ts:
                        meta["timestamp_year"] = int(ts["year"])
                    if "month" in ts:
                        meta["timestamp_month"] = int(ts["month"])
                    if "day" in ts:
                        meta["timestamp_day"] = int(ts["day"])
                batch_metadatas.append(meta)

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
            # Deserialize JSON payload strings back to dictionaries
            metadatas_raw = results.get("metadatas", [])
            metadatas_parsed = []
            for meta in metadatas_raw:
                if meta and "payload" in meta:
                    try:
                        # Parse the JSON string back to dict
                        payload_dict = json.loads(meta["payload"])
                        metadatas_parsed.append({"payload": payload_dict})
                    except (json.JSONDecodeError, TypeError):
                        # If parsing fails, return as-is
                        metadatas_parsed.append(meta)
                else:
                    metadatas_parsed.append(meta)
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
        # Build metadata filter for ChromaDB
        # ChromaDB uses $and for combining conditions and $gte/$lte for ranges
        conditions = []
        
        if year is not None:
            conditions.append({"timestamp_year": year})
        if month is not None:
            conditions.append({"timestamp_month": month})
        if day is not None:
            conditions.append({"timestamp_day": day})
        
        # For epoch range, we need to use ChromaDB's $gte and $lte operators
        if start_epoch is not None or end_epoch is not None:
            epoch_filter = {}
            if start_epoch is not None:
                epoch_filter["$gte"] = start_epoch
            if end_epoch is not None:
                epoch_filter["$lte"] = end_epoch
            if epoch_filter:
                conditions.append({"timestamp_epoch": epoch_filter})

        # Build where clause - use $and if multiple conditions, otherwise use single condition
        if len(conditions) > 1:
            where_clause = {"$and": conditions}
        elif len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = None

        logger.info(
            "Querying ChromaDB collection '%s' with timestamp filter: %s",
            self.collection_name,
            where_clause,
        )

        # Query ChromaDB
        if where_clause:
            try:
                results = self.collection.get(
                    where=where_clause,
                    limit=limit,
                    include=["embeddings", "documents", "metadatas"],
                )
            except Exception as e:
                # If metadata fields don't exist (old data), fall back to getting all and filtering in Python
                logger.warning(
                    "ChromaDB metadata filter failed (possibly old data format): %s. "
                    "Falling back to Python-side filtering.",
                    e,
                )
                results = self.collection.get(
                    limit=None,  # Get all to filter in Python
                    include=["embeddings", "documents", "metadatas"],
                )
                # Filter in Python
                filtered_ids = []
                filtered_embeddings = []
                filtered_texts = []
                filtered_metas = []
                
                for idx, meta in enumerate(results.get("metadatas", [])):
                    # Parse payload to check timestamp
                    payload_str = meta.get("payload", "") if meta else ""
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
                            filtered_ids.append(results.get("ids", [])[idx])
                            if results.get("embeddings"):
                                filtered_embeddings.append(results.get("embeddings", [])[idx])
                            filtered_texts.append(results.get("documents", [])[idx])
                            filtered_metas.append(meta)
                            if limit and len(filtered_ids) >= limit:
                                break
                    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
                        continue
                
                # Reconstruct results dict
                results = {
                    "ids": filtered_ids,
                    "embeddings": filtered_embeddings,
                    "documents": filtered_texts,
                    "metadatas": filtered_metas,
                }
        else:
            # No filter, get all
            results = self.collection.get(
                limit=limit,
                include=["embeddings", "documents", "metadatas"],
            )

        # Convert embeddings to numpy array
        embeddings_list = results.get("embeddings", [])
        embeddings_array = np.array(embeddings_list, dtype=np.float32) if embeddings_list else np.array([])

        result = {
            "embeddings": embeddings_array,
            "ids": results.get("ids", []),
            "texts": results.get("documents", []),
            "metadatas": [],
        }

        # Deserialize JSON payload strings
        metadatas_raw = results.get("metadatas", [])
        for meta in metadatas_raw:
            if meta and "payload" in meta:
                try:
                    payload_dict = json.loads(meta["payload"])
                    result["metadatas"].append({"payload": payload_dict})
                except (json.JSONDecodeError, TypeError):
                    result["metadatas"].append(meta)
            else:
                result["metadatas"].append(meta)

        logger.info("Retrieved %d embeddings matching timestamp filter", len(result["ids"]))
        return result


