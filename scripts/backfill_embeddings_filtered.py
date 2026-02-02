import os
import time
import argparse
import logging
from typing import List, Tuple, Optional

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


def to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def build_where_clause(
    platforms: Optional[List[str]],
    state: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[str, List[object]]:
    clauses = [
        "embedding IS NULL",
        "text IS NOT NULL",
        "btrim(text) <> ''",
    ]
    params: List[object] = []

    if platforms:
        placeholders = ",".join(["%s"] * len(platforms))
        clauses.append(f"platform IN ({placeholders})")
        params.extend([p.strip().lower() for p in platforms if p.strip()])

    if state:
        clauses.append("state = %s")
        params.append(state)

    if date_from:
        clauses.append("posted_date >= %s")
        params.append(date_from)

    if date_to:
        clauses.append("posted_date <= %s")
        params.append(date_to)

    return " AND ".join(clauses), params


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--platforms", type=str, default=None, help="Comma list, e.g. instagram,twitter")
    parser.add_argument("--state", type=str, default=None, help="Exact match, e.g. Tamil Nadu")
    parser.add_argument("--date-from", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "128")))
    parser.add_argument("--limit-total", type=int, default=0, help="Stop after N rows (0 = no limit)")
    args = parser.parse_args()

    platforms = args.platforms.split(",") if args.platforms else None

    dsn = os.getenv("PGVECTOR_DSN", "postgresql://postgres:postgress@localhost:5433/sma-data")
    table = os.getenv("PGVECTOR_TABLE_NAME", "social_media_posts")
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    device = os.getenv("EMBEDDING_DEVICE", None)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("backfill_filtered")

    logger.info("DB: %s", dsn.split("@")[-1])
    logger.info("Filters: platforms=%s state=%s date_from=%s date_to=%s",
                platforms, args.state, args.date_from, args.date_to)

    logger.info("Loading model: %s (device=%s)", model_name, device)
    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)

    where_sql, where_params = build_where_clause(platforms, args.state, args.date_from, args.date_to)

    total_done = 0
    start = time.time()

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            while True:
                if args.limit_total and total_done >= args.limit_total:
                    logger.info("Reached limit_total=%d. Stopping ✅", args.limit_total)
                    break

                limit_now = args.batch_size
                if args.limit_total:
                    limit_now = min(limit_now, args.limit_total - total_done)

                cur.execute(
                    f"""
                    SELECT id, text
                    FROM {table}
                    WHERE {where_sql}
                    LIMIT %s
                    """,
                    (*where_params, limit_now),
                )

                rows: List[Tuple[str, str]] = cur.fetchall()
                if not rows:
                    logger.info("No more matching rows with NULL embeddings. Done ✅")
                    break

                ids = [r[0] for r in rows]
                texts = [r[1] for r in rows]

                docs = [f"passage: {t}" for t in texts]

                logger.info("Embedding %d rows...", len(docs))
                embs = model.encode(
                    docs,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                update_sql = f"""
                    UPDATE {table}
                    SET embedding = %s::vector,
                        embedding_model = %s,
                        embedding_provider = %s
                    WHERE id = %s
                """

                for i, _id in enumerate(ids):
                    vec_literal = to_pgvector_literal(embs[i].astype(float).tolist())
                    cur.execute(update_sql, (vec_literal, model_name, provider, _id))

                conn.commit()
                total_done += len(ids)

                elapsed = time.time() - start
                logger.info("Committed %d rows (elapsed %.1fs)", total_done, elapsed)

    logger.info("Finished. Total updated=%d", total_done)


if __name__ == "__main__":
    main()
