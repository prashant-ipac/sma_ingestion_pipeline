from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()


def wait_until_queryable(collection, index_name: str, timeout_s: int = 180, poll_s: int = 5) -> None:
    """
    Polls Atlas Search index state until it becomes queryable or times out.
    """
    print(f"Polling until index '{index_name}' is queryable (timeout={timeout_s}s)...")

    start = time.time()
    while True:
        indexes = list(collection.list_search_indexes(index_name))
        if indexes:
            idx = indexes[0]
            # Atlas reports `queryable: True` when ready
            if idx.get("queryable") is True:
                print(f"✅ Index '{index_name}' is ready for querying.")
                return
            else:
                status = idx.get("status") or idx.get("state") or "building"
                print(f"… still building (queryable={idx.get('queryable')}) status={status}")
        else:
            print("… index not visible yet (still creating)")

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Index '{index_name}' not queryable after {timeout_s}s")

        time.sleep(poll_s)


def main() -> None:
    uri = os.getenv("ATLASDB_URI")
    db_name = os.getenv("ATLASDB_DATABASE_NAME", "socialmediaanalytics")

    # Use CLI override collection name if you want; else from env
    collection_name = os.getenv("ATLASDB_COLLECTION_NAME", "instagram")

    index_name = os.getenv("ATLASDB_INDEX_NAME", "vector_index")

    # Dimension: prefer ATLASDB_EMBEDDING_DIM, else EMBEDDING_DIM, else 1024
    dim = int(os.getenv("ATLASDB_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIM") or "1024")

    # Similarity: choose one that matches how you plan to search.
    # If you normalized embeddings -> cosine is common. If not, dotProduct can work.
    similarity = os.getenv("ATLASDB_SIMILARITY", "cosine")  # cosine|dotProduct|euclidean

    # Optional: scalar quantization speeds search, sometimes hurts recall a bit.
    # Set ATLASDB_QUANTIZATION=scalar to enable, else keep None.
    quantization = os.getenv("ATLASDB_QUANTIZATION", "").strip().lower()  # "" or "scalar"

    if not uri:
        raise ValueError("ATLASDB_URI is missing in .env")

    print(f"Connecting to Atlas… db='{db_name}' collection='{collection_name}'")
    client = MongoClient(uri, serverSelectionTimeoutMS=20000)

    # Verify connection
    client.admin.command("ping")
    print("✅ Connected (ping ok)")

    collection = client[db_name][collection_name]

    # If index already exists, print and exit (or delete manually in Atlas UI)
    existing = list(collection.list_search_indexes())
    existing_names = [i.get("name") for i in existing]
    print("Existing search indexes:", existing_names)

    if index_name in existing_names:
        print(f"ℹ️ Index '{index_name}' already exists. No action taken.")
        client.close()
        return

    # ---- Index definition for your ingestion schema ----
    # Vector field: embedding
    # Filters: add what you commonly filter on.
    # NOTE: Only include filter fields that actually exist in your documents.
    fields = [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": dim,
            "similarity": similarity,
            **({"quantization": quantization} if quantization else {}),
        },
        # Optional filters (safe + useful):
        {"type": "filter", "path": "payload.platform"},
        {"type": "filter", "path": "payload.content_type"},
        {"type": "filter", "path": "payload.timestamp.year"},
        {"type": "filter", "path": "payload.timestamp.month"},
        {"type": "filter", "path": "payload.timestamp.day"},
        # If you store language/hashtags often, you can add them too:
        {"type": "filter", "path": "payload.content.language"},
    ]

    search_index_model = SearchIndexModel(
        definition={"fields": fields},
        name=index_name,
        type="vectorSearch",
    )

    print(f"Creating search index '{index_name}' (dim={dim}, similarity={similarity}, quantization={quantization or 'none'})…")
    created_name = collection.create_search_index(model=search_index_model)
    print(f"New search index named '{created_name}' is building.")

    wait_until_queryable(collection, created_name, timeout_s=240, poll_s=5)
    client.close()
    print("✅ Done.")


if __name__ == "__main__":
    main()
