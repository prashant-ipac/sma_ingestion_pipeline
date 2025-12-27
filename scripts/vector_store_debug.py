"""
Quick CLI helpers for semantic search and schema/content inspection.

Examples:
  # ChromaDB semantic search
  python scripts/vector_store_debug.py semantic-search "what is the post about?" --backend chromadb --top-k 5

  # Inspect schemas/content
  python scripts/vector_store_debug.py inspect --backend pgvector --limit 5
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import List

import typer
from rich.console import Console
from rich.table import Table

from src.config import Config
from src.embedding import EmbeddingModel
from src.vector_store.chroma_vector_store import ChromaVectorStore

# Conditional import for pgvector
try:
    import psycopg2
except ImportError:
    psycopg2 = None


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
            include=["ids", "documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
    elif backend == "pgvector":
        if psycopg2 is None:
            raise typer.BadParameter("psycopg2 is required for pgvector backend. Install it with: pip install psycopg2-binary")
        vec_literal = "[" + ", ".join(f"{v:.8f}" for v in embedding) + "]"
        conn = psycopg2.connect(cfg.pgvector_dsn)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, text, payload, embedding <-> %s::vector AS distance
            FROM {cfg.pgvector_table_name}
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
            """,
            (vec_literal, vec_literal, top_k),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        ids = [str(row[0]) for row in rows]
        docs = [row[1] for row in rows]
        metas = [row[2] for row in rows]
        dists = [row[3] for row in rows]
    else:
        raise typer.BadParameter("Semantic search supported only for chromadb or pgvector.")

    table = Table(title=f"Top {len(docs)} results (backend={backend})")
    table.add_column("#", justify="right")
    table.add_column("ID", style="cyan")
    table.add_column("Distance", style="yellow")
    table.add_column("Platform", style="green")
    table.add_column("Author", style="magenta")
    table.add_column("Text", overflow="fold", max_width=50)

    for idx, (entry_id, dist, meta, doc) in enumerate(zip(ids, dists, metas, docs), start=1):
        # Extract payload from metadata
        if isinstance(meta, dict):
            payload_raw = meta.get("payload", meta)
            # ChromaDB stores payload as JSON string, so parse it
            if isinstance(payload_raw, str):
                try:
                    payload = json.loads(payload_raw)
                except (json.JSONDecodeError, TypeError):
                    payload = payload_raw
            else:
                payload = payload_raw
        else:
            payload = meta or {}
        
        platform = payload.get("platform", "N/A") if isinstance(payload, dict) else "N/A"
        author = payload.get("author_name", payload.get("author_id", "N/A")) if isinstance(payload, dict) else "N/A"
        
        table.add_row(
            str(idx),
            entry_id[:8] + "..." if len(entry_id) > 8 else entry_id,
            f"{dist:.4f}",
            str(platform),
            str(author),
            doc[:50] + "..." if len(doc) > 50 else doc,
        )

    console.print(table)


@app.command("inspect")
def inspect(
    backend: str = typer.Option("chromadb", "--backend", "-b", help="chromadb|pgvector"),
    limit: int = typer.Option(5, "--limit", "-n", help="Rows to preview."),
    show_schema: bool = typer.Option(True, "--schema/--no-schema", help="Show schema information"),
    show_payload: bool = typer.Option(True, "--payload/--no-payload", help="Show full payload structure"),
) -> None:
    """
    Display schemas and a small sample of stored content with new format.
    """
    cfg = Config()
    cfg.backend = backend
    cfg.validate()

    if backend == "chromadb":
        store = ChromaVectorStore(path=cfg.chromadb_path)
        count = store.collection.count()
        
        console.print("\n[bold cyan]=== ChromaDB Schema ===[/bold cyan]")
        console.print(f"[bold]Collection:[/bold] {store.collection_name}")
        console.print(f"[bold]Path:[/bold] {store.path}")
        console.print(f"[bold]Total Records:[/bold] {count}")
        
        if show_schema:
            console.print("\n[bold green]Schema Structure:[/bold green]")
            schema_info = Table(title="Expected Schema Format")
            schema_info.add_column("Field", style="cyan")
            schema_info.add_column("Type", style="yellow")
            schema_info.add_column("Description", style="green")
            schema_info.add_row("id", "UUID (string)", "Unique identifier for each entry")
            schema_info.add_row("vector", "List[float]", "Embedding vector")
            schema_info.add_row("document", "string", "Text content")
            schema_info.add_row("metadata.payload", "JSON string", "Structured payload (stored as JSON string due to ChromaDB limitations)")
            console.print(schema_info)

        if count:
            preview = store.collection.get(include=["ids", "documents", "metadatas"], limit=limit)
            ids = preview.get("ids", [])
            docs = preview.get("documents", [])
            metas = preview.get("metadatas", [])

            console.print(f"\n[bold green]Sample Data (first {len(docs)} entries):[/bold green]")
            for idx, (entry_id, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
                console.print(f"\n[bold cyan]Entry #{idx}[/bold cyan]")
                console.print(f"  [yellow]ID:[/yellow] {entry_id}")
                console.print(f"  [yellow]Text:[/yellow] {doc[:100]}{'...' if len(doc) > 100 else ''}")
                
                if meta:
                    # ChromaDB stores payload as JSON string, so we need to parse it
                    payload_raw = meta.get("payload", meta)
                    if isinstance(payload_raw, str):
                        try:
                            payload = json.loads(payload_raw)
                        except (json.JSONDecodeError, TypeError):
                            payload = payload_raw
                    else:
                        payload = payload_raw
                    
                    if show_payload:
                        console.print(f"  [yellow]Payload:[/yellow]")
                        payload_str = json.dumps(payload, indent=4, ensure_ascii=False)
                        # Show first 500 chars of payload
                        if len(payload_str) > 500:
                            console.print(f"    {payload_str[:500]}...")
                        else:
                            console.print(f"    {payload_str}")
                    else:
                        # Show summary of payload structure
                        if isinstance(payload, dict):
                            console.print(f"  [yellow]Payload Keys:[/yellow] {', '.join(payload.keys())[:100]}")
                else:
                    console.print(f"  [yellow]Metadata:[/yellow] (empty)")
    elif backend == "pgvector":
        if psycopg2 is None:
            raise typer.BadParameter("psycopg2 is required for pgvector backend. Install it with: pip install psycopg2-binary")
        conn = psycopg2.connect(cfg.pgvector_dsn)
        cur = conn.cursor()

        console.print("\n[bold cyan]=== pgvector Schema ===[/bold cyan]")
        console.print(f"[bold]Table:[/bold] {cfg.pgvector_table_name}")
        console.print(f"[bold]DSN:[/bold] {cfg.pgvector_dsn}")

        # Get count
        cur.execute(f"SELECT COUNT(*) FROM {cfg.pgvector_table_name};")
        count = cur.fetchone()[0]
        console.print(f"[bold]Total Records:[/bold] {count}")

        if show_schema:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
                """,
                (cfg.pgvector_table_name,),
            )
            columns = cur.fetchall()
            schema_table = Table(title="Database Schema")
            schema_table.add_column("Column", style="cyan")
            schema_table.add_column("Type", style="yellow")
            schema_table.add_column("Nullable", style="green")
            for col, typ, nullable in columns:
                schema_table.add_row(col, typ, nullable)
            console.print(schema_table)

            console.print("\n[bold green]Expected Schema Format:[/bold green]")
            expected_schema = Table()
            expected_schema.add_column("Field", style="cyan")
            expected_schema.add_column("Type", style="yellow")
            expected_schema.add_column("Description", style="green")
            expected_schema.add_row("id", "UUID", "Primary key (UUID)")
            expected_schema.add_row("text", "TEXT", "Text content")
            expected_schema.add_row("payload", "JSONB", "Structured payload with all metadata")
            expected_schema.add_row("embedding", "vector(N)", "Embedding vector")
            console.print(expected_schema)

        if count:
            cur.execute(
                f"""
                SELECT id, text, payload
                FROM {cfg.pgvector_table_name}
                ORDER BY id ASC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            
            console.print(f"\n[bold green]Sample Data (first {len(rows)} entries):[/bold green]")
            for idx, (row_id, text, payload) in enumerate(rows, start=1):
                console.print(f"\n[bold cyan]Entry #{idx}[/bold cyan]")
                console.print(f"  [yellow]ID:[/yellow] {row_id}")
                console.print(f"  [yellow]Text:[/yellow] {text[:100]}{'...' if len(text) > 100 else ''}")
                
                if payload:
                    if show_payload:
                        console.print(f"  [yellow]Payload:[/yellow]")
                        payload_str = json.dumps(payload, indent=4, ensure_ascii=False)
                        if len(payload_str) > 500:
                            console.print(f"    {payload_str[:500]}...")
                        else:
                            console.print(f"    {payload_str}")
                    else:
                        if isinstance(payload, dict):
                            console.print(f"  [yellow]Payload Keys:[/yellow] {', '.join(payload.keys())[:100]}")
                else:
                    console.print(f"  [yellow]Payload:[/yellow] (empty)")

        cur.close()
        conn.close()
    else:
        raise typer.BadParameter("Inspect supported only for chromadb or pgvector.")
    
    console.print("\n[bold green]✓ Schema inspection complete[/bold green]")


if __name__ == "__main__":
    app()

