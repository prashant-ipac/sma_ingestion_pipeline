"""
MongoDB Atlas vector store implementation.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterable, Mapping, Sequence

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, PyMongoError, ServerSelectionTimeoutError

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
        server_selection_timeout_ms: int = 20000,
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
            server_selection_timeout_ms: Connection timeout for cluster selection
        """
        self.uri = uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.max_batch_size = max_batch_size
        self.index_name = index_name

        logger.info(
            "Initializing AtlasVectorStore: db=%s collection=%s embedding_dim=%d max_batch_size=%d index=%s",
            self.database_name,
            self.collection_name,
            self.embedding_dim,
            self.max_batch_size,
            self.index_name,
        )

        t0 = time.time()
        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=server_selection_timeout_ms,
            )

            # Verify connectivity early (this prevents silent hangs later)
            self.client.admin.command("ping")
            dt = time.time() - t0

            self.db = self.client[self.database_name]
            self.collection: Collection = self.db[self.collection_name]

            logger.info(
                "Connected to MongoDB Atlas (ping ok) in %.2fs | db='%s' collection='%s'",
                dt,
                self.database_name,
                self.collection_name,
            )

            try:
                info = self.client.server_info()
                logger.debug("Mongo server info: version=%s", info.get("version"))
            except Exception:
                logger.debug("Unable to fetch server_info()", exc_info=True)

        except ServerSelectionTimeoutError:
            logger.exception(
                "MongoDB connection failed (server selection timeout). "
                "Check ATLASDB_URI, IP allowlist, DNS/network, and credentials."
            )
            raise
        except Exception:
            logger.exception("MongoDB connection failed. Check ATLASDB_URI and network.")
            raise

    def close(self) -> None:
        try:
            self.client.close()
            logger.info("MongoDB client closed.")
        except Exception:
            logger.debug("Failed to close MongoDB client.", exc_info=True)

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
        total = len(texts)

        if total == 0:
            logger.warning("add_embeddings called with 0 texts. Nothing to insert.")
            return

        # Build payloads if needed
        if payloads is None:
            if metadatas is None:
                metadatas = [{} for _ in texts]
            metadatas_list = list(metadatas)
            payloads = [
                create_payload(
                    text=text,
                    ingested_from=str(meta.get("source", "excel")),
                    file_name=str(meta.get("file_name", "")),
                    row_number=int(meta.get("index", meta.get("row_number", 0)) or 0),
                )
                for text, meta in zip(texts, metadatas_list)
            ]
        else:
            payloads = list(payloads)

        if len(embeddings) != total or len(payloads) != total:
            raise ValueError(
                f"Embeddings/texts/payloads length mismatch: embeddings={len(embeddings)} texts={total} payloads={len(payloads)}"
            )

        # Validate embedding dimension
        try:
            emb_dim = int(getattr(embeddings, "shape", (0, 0))[1])
            if emb_dim and emb_dim != self.embedding_dim:
                logger.warning(
                    "Embedding dim mismatch: expected=%d actual=%d. "
                    "This may break Atlas vector search index if configured for expected dim.",
                    self.embedding_dim,
                    emb_dim,
                )
        except Exception:
            logger.debug("Unable to validate embedding dimension.", exc_info=True)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(total)]
        else:
            ids = list(ids)

        logger.info(
            "Inserting into Atlas: items=%d collection='%s' batch_size=%d",
            total,
            self.collection_name,
            self.max_batch_size,
        )

        # Build documents (convert each embedding to float32 list)
        t_build = time.time()
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
        logger.info("Prepared %d documents in %.2fs", total, time.time() - t_build)

        # DEBUG: show one sample doc structure
        if logger.isEnabledFor(10) and documents:
            sample = documents[0]
            logger.debug(
                "Sample doc: _id=%s text_len=%d payload_keys=%s embedding_len=%d",
                sample.get("_id"),
                len(sample.get("text") or ""),
                list((sample.get("payload") or {}).keys()),
                len(sample.get("embedding") or []),
            )

        inserted_total = 0
        t_total = time.time()

        for batch_start in range(0, total, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, total)
            batch_docs = documents[batch_start:batch_end]
            batch_no = (batch_start // self.max_batch_size) + 1
            batch_count = len(batch_docs)

            t0 = time.time()
            try:
                result = self.collection.insert_many(batch_docs, ordered=False)
                dt = time.time() - t0
                inserted = len(result.inserted_ids) if result and result.inserted_ids else batch_count
                inserted_total += inserted

                logger.info(
                    "Batch %d inserted: range=%d-%d count=%d inserted=%d time=%.2fs (progress=%d/%d)",
                    batch_no,
                    batch_start,
                    batch_end - 1,
                    batch_count,
                    inserted,
                    dt,
                    min(batch_end, total),
                    total,
                )

            except BulkWriteError as bwe:
                dt = time.time() - t0
                details = bwe.details or {}
                write_errors = details.get("writeErrors", [])
                n_inserted = details.get("nInserted", 0)

                inserted_total += n_inserted

                # Count duplicates (code 11000)
                dupes = [e for e in write_errors if e.get("code") == 11000]
                other = [e for e in write_errors if e.get("code") != 11000]

                logger.warning(
                    "Batch %d BulkWriteError: inserted=%d errors=%d dupes=%d other=%d time=%.2fs",
                    batch_no,
                    n_inserted,
                    len(write_errors),
                    len(dupes),
                    len(other),
                    dt,
                )

                if other:
                    # Log first few non-duplicate errors
                    for e in other[:3]:
                        logger.error(
                            "Non-duplicate write error: code=%s msg=%s index=%s",
                            e.get("code"),
                            e.get("errmsg"),
                            e.get("index"),
                        )
                    raise  # non-duplicate errors should stop the pipeline

                # If only duplicates happened, continue
                logger.info(
                    "Continuing after duplicates in batch %d. (duplicates happen if you re-run ingestion with same ids)",
                    batch_no,
                )

            except PyMongoError:
                dt = time.time() - t0
                logger.exception(
                    "Mongo insert failed on batch %d (range=%d-%d) after %.2fs",
                    batch_no,
                    batch_start,
                    batch_end - 1,
                    dt,
                )
                raise

        logger.info(
            "Atlas insert completed: requested=%d inserted=%d total_time=%.2fs",
            total,
            inserted_total,
            time.time() - t_total,
        )

    def get_all_embeddings(
        self,
        include_texts: bool = True,
        include_metadatas: bool = True,
    ) -> dict:
        """
        Retrieve all embeddings from the MongoDB Atlas collection.
        """
        logger.info("Fetching all embeddings from collection='%s'...", self.collection_name)

        t0 = time.time()
        cursor = self.collection.find({})

        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for doc in cursor:
            ids.append(doc["_id"])
            embeddings.append(doc.get("embedding"))

            if include_texts:
                texts.append(doc.get("text", ""))

            if include_metadatas:
                metadatas.append({"payload": doc.get("payload", {})})

        dt = time.time() - t0

        if not ids:
            logger.warning("Collection '%s' is empty.", self.collection_name)
            result = {"embeddings": np.array([]), "ids": []}
            if include_texts:
                result["texts"] = []
            if include_metadatas:
                result["metadatas"] = []
            return result

        logger.info("Fetched %d embeddings in %.2fs", len(ids), dt)

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
        logger.info("Querying collection='%s' with timestamp filters...", self.collection_name)

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

        t0 = time.time()
        cursor = self.collection.find(mongo_filter).limit(limit or 0)

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for doc in cursor:
            ids.append(doc["_id"])
            texts.append(doc.get("text", ""))
            embeddings.append(doc.get("embedding"))
            metadatas.append({"payload": doc.get("payload", {})})

        dt = time.time() - t0

        if not ids:
            logger.info("No documents matched the timestamp filter. time=%.2fs", dt)
            return {"embeddings": np.array([]), "ids": [], "texts": [], "metadatas": []}

        logger.info("Retrieved %d docs matching filter in %.2fs", len(ids), dt)

        return {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "ids": ids,
            "texts": texts,
            "metadatas": metadatas,
        }
