import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

def main():
    load_dotenv()
    dsn = os.getenv("PGVECTOR_DSN", "postgresql://postgres:postgress@localhost:5433/sma-data")

    out_path = os.getenv("EXPORT_OUT", "facebook_id_text.parquet")

    query = """
    SELECT id, text
    FROM social_media_posts
    WHERE platform = 'facebook'
      AND text IS NOT NULL
      AND btrim(text) <> ''
    ORDER BY id;
    """

    print("Connecting to DB...")
    with psycopg2.connect(dsn) as conn:
        print("Running query...")
        df = pd.read_sql(query, conn)

    print(f"Fetched {len(df)} rows. Writing {out_path} ...")
    df.to_parquet("facebook_id_text.parquet", index=False)
    print("Done ✅")

if __name__ == "__main__":
    main()
