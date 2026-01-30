import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

def to_pgvector_literal(v):
    return "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"

def main():
    load_dotenv()

    dsn = os.getenv("PGVECTOR_DSN", "postgresql://postgres:postgress@localhost:5433/sma-data")
    parquet_path = os.getenv("EMB_PARQUET_PATH", "C:/Users/Admin/Downloads/facebook_embeddings_bge_m3_1024.parquet")
    table = os.getenv("PGVECTOR_TABLE_NAME", "social_media_posts")

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

    df = pd.read_parquet(parquet_path)
    df["id"] = df["id"].astype(str)

    # Basic sanity check
    sample_len = len(df.iloc[0]["embedding"])
    if sample_len != 1024:
        raise ValueError(f"Embedding dim mismatch: got {sample_len}, expected 1024")

    print("Loaded parquet rows:", len(df))

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            # 1) Create TEMP staging table
            cur.execute("""
                CREATE TEMP TABLE tmp_embeddings (
                    id TEXT PRIMARY KEY,
                    embedding vector(1024)
                ) ON COMMIT DROP;
            """)

            # 2) Bulk insert into staging table
            records = [(row_id, to_pgvector_literal(vec)) for row_id, vec in zip(df["id"], df["embedding"])]

            execute_values(
                cur,
                "INSERT INTO tmp_embeddings (id, embedding) VALUES %s "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
                records,
                page_size=2000
            )

            # 3) Update main table via join
            cur.execute(f"""
                UPDATE {table} t
                SET embedding = s.embedding,
                    embedding_model = %s,
                    embedding_provider = %s
                FROM tmp_embeddings s
                WHERE t.id = s.id;
            """, (model_name, provider))

            # 4) Report how many rows updated
            cur.execute("SELECT COUNT(*) FROM tmp_embeddings;")
            staged = cur.fetchone()[0]
            print("Staged rows:", staged)

        conn.commit()

    print("Done ✅")

if __name__ == "__main__":
    main()
