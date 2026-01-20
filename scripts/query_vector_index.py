import os
from dotenv import load_dotenv
from pymongo import MongoClient
import voyageai

load_dotenv()

ATLASDB_URI = os.getenv("ATLASDB_URI")
DB_NAME = os.getenv("ATLASDB_DATABASE_NAME", "socialmediaanalytics")
COLLECTION = os.getenv("ATLASDB_COLLECTION_NAME", "instagram")
INDEX_NAME = os.getenv("ATLASDB_INDEX_NAME", "instagram_vector_index")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-4-lite")  # set to what you used for ingestion

client = MongoClient(ATLASDB_URI)
col = client[DB_NAME][COLLECTION]

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

query_text = "SIR in Bihar"

# IMPORTANT: use input_type="query" for searches
emb = vo.embed([query_text], model=VOYAGE_MODEL, input_type="query").embeddings[0]

pipeline = [
    {
        "$vectorSearch": {
            "index": INDEX_NAME,
            "path": "embedding",
            "queryVector": emb,
            "numCandidates": 300,
            "limit": 10,
            # Optional filter (must be indexed as filter fields)
            # "filter": {"payload.timestamp.year": 2025},
        }
    },
    {
        "$project": {
            "_id": 1,
            "score": {"$meta": "vectorSearchScore"},
            "text": 1,
            "payload": 1,
        }
    },
]

results = list(col.aggregate(pipeline))
for r in results:
    print(r["score"], r.get("text", "")[:120])

client.close()
