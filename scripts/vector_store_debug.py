"""
Quick CLI helpers for semantic search and schema/content inspection.

Examples:
  # ChromaDB semantic search
  python scripts/vector_store_debug.py semantic-search "what is the post about?" --backend chromadb --top-k 5

  # Inspect schemas/content
  python scripts/vector_store_debug.py inspect --backend pgvector --limit 5
"""

from __future__ import annotations

import json
from typing import List

import psycopg2
import typer
from rich.console import Console
from rich.table import Table

from src.config import Config
from src.embedding import EmbeddingModel
from src.vector_store.chroma_vector_store import ChromaVectorStore


app = typer.Typer(help="Debug utilities for vector stores.")
console = Console()


def _embed_query(cfg: Config, query: str) -> List[float]:
    model = EmbeddingModel(cfg.embedding_model)
    vec = model.encode([query])[0]
    return [float(v) for v in vec.tolist()]


@app.command("semantic-search")
def semantic_search(
    query: str = typer.Argument(..., help="Natural language search query."),
    backend: str = typer.Option("chromadb", "--backend", "-b", help="chromadb|pgvector"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return."),
) -> None:
    """
    Run a quick semantic search against the configured vector store.
    """
    cfg = Config()
    cfg.backend = backend
    cfg.validate()

    embedding = _embed_query(cfg, query)

    if backend == "chromadb":
        store = ChromaVectorStore(path=cfg.chromadb_path)
        results = store.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
    elif backend == "pgvector":
        vec_literal = "[" + ", ".join(f"{v:.8f}" for v in embedding) + "]"
        conn = psycopg2.connect(cfg.pgvector_dsn)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, text, metadata, embedding <-> %s::vector AS distance
            FROM {cfg.pgvector_table_name}
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
            """,
            (vec_literal, vec_literal, top_k),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        docs = [row[1] for row in rows]
        metas = [row[2] for row in rows]
        dists = [row[3] for row in rows]
    else:
        raise typer.BadParameter("Semantic search supported only for chromadb or pgvector.")

    table = Table(title=f"Top {len(docs)} results (backend={backend})")
    table.add_column("#", justify="right")
    table.add_column("Distance")
    table.add_column("Metadata")
    table.add_column("Text", overflow="fold")

    for idx, (dist, meta, doc) in enumerate(zip(dists, metas, docs), start=1):
        meta_repr = json.dumps(meta or {}, ensure_ascii=False)
        table.add_row(str(idx), f"{dist:.4f}", meta_repr, doc)

    console.print(table)


@app.command("inspect")
def inspect(
    backend: str = typer.Option("chromadb", "--backend", "-b", help="chromadb|pgvector"),
    limit: int = typer.Option(5, "--limit", "-n", help="Rows to preview."),
) -> None:
    """
    Display schemas and a small sample of stored content.
    """
    cfg = Config()
    cfg.backend = backend
    cfg.validate()

    if backend == "chromadb":
        store = ChromaVectorStore(path=cfg.chromadb_path)
        count = store.collection.count()
        console.print(f"[bold]Collection:[/bold] {store.collection_name}")
        console.print(f"[bold]Path:[/bold] {store.path}")
        console.print(f"[bold]Count:[/bold] {count}")

        if count:
            preview = store.collection.get(include=["documents", "metadatas"], limit=limit)
            docs = preview.get("documents", [])
            metas = preview.get("metadatas", [])

            table = Table(title=f"First {len(docs)} documents")
            table.add_column("#", justify="right")
            table.add_column("Metadata")
            table.add_column("Text", overflow="fold")
            for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
                table.add_row(str(idx), json.dumps(meta or {}, ensure_ascii=False), doc)
            console.print(table)
    elif backend == "pgvector":
        conn = psycopg2.connect(cfg.pgvector_dsn)
        cur = conn.cursor()

        console.print(f"[bold]Table:[/bold] {cfg.pgvector_table_name}")
        console.print(f"[bold]DSN:[/bold] {cfg.pgvector_dsn}")

        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
            """,
            (cfg.pgvector_table_name,),
        )
        columns = cur.fetchall()
        schema_table = Table(title="Schema")
        schema_table.add_column("Column")
        schema_table.add_column("Type")
        for col, typ in columns:
            schema_table.add_row(col, typ)
        console.print(schema_table)

        cur.execute(
            f"""
            SELECT id, text, metadata
            FROM {cfg.pgvector_table_name}
            ORDER BY id ASC
            LIMIT %s;
            """,
            (limit,),
        )
        rows = cur.fetchall()
        data_table = Table(title=f"First {len(rows)} rows")
        data_table.add_column("id", justify="right")
        data_table.add_column("metadata")
        data_table.add_column("text", overflow="fold")
        for row_id, text, meta in rows:
            data_table.add_row(str(row_id), json.dumps(meta or {}, ensure_ascii=False), text)
        console.print(data_table)

        cur.close()
        conn.close()
    else:
        raise typer.BadParameter("Inspect supported only for chromadb or pgvector.")


if __name__ == "__main__":
    app()

