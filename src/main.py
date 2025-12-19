"""
CLI entrypoint for the social_media_vectordb ingestion pipeline.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.progress import track

from .config import Config
from .logging_utils import configure_logging, get_logger
from .data_loader import load_texts_from_excel
from .chunking import chunk_texts
from .embedding import EmbeddingModel
from .vector_store import S3VectorStore, ChromaVectorStore, PgVectorStore


app = typer.Typer(help="Social media Excel → embeddings → vector store pipeline")
console = Console()
logger = get_logger(__name__)


@app.command()
def ingest(
    excel_path: str = typer.Argument(..., help="Path to the Excel file with social media data."),
    sheet_name: Optional[str] = typer.Option(
        None, "--sheet-name", "-s", help="Sheet name in the Excel file (defaults to config/default)."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override vector store backend (s3|chromadb|pgvector)."
    ),
    chunking_strategy: Optional[str] = typer.Option(
        None, "--chunking-strategy", "-c", help="Override chunking strategy (recursive|fixed)."
    ),
) -> None:
    """
    Run the full ingestion pipeline:
    Excel → texts → chunks → embeddings → vector store.
    """
    cfg = Config()
    if backend:
        cfg.backend = backend
    if chunking_strategy:
        cfg.chunking_strategy = chunking_strategy

    cfg.validate()
    configure_logging(cfg.log_level)

    console.print("[bold cyan]Starting ingestion pipeline...[/bold cyan]")
    console.print(f"[bold]Excel:[/bold] {excel_path}")
    console.print(f"[bold]Backend:[/bold] {cfg.backend}")
    console.print(f"[bold]Chunking:[/bold] {cfg.chunking_strategy}")
    console.print(f"[bold]Model:[/bold] {cfg.embedding_model}")

    # Load data
    texts = load_texts_from_excel(
        excel_path=excel_path,
        sheet_name=sheet_name or cfg.default_sheet_name,
        text_columns=cfg.text_columns,
    )

    # Chunk
    chunks = chunk_texts(
        texts=texts,
        strategy=cfg.chunking_strategy,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # Embeddings
    model = EmbeddingModel(cfg.embedding_model)
    embeddings = model.encode(chunks)

    # Vector store backend selection
    if cfg.backend == "s3":
        store = S3VectorStore(
            bucket_name=cfg.aws_s3_bucket_name,
            region_name=cfg.aws_region,
            embeddings_prefix=cfg.aws_s3_embeddings_prefix,
            index_key=cfg.aws_s3_index_key,
        )
    elif cfg.backend == "chromadb":
        store = ChromaVectorStore(path=cfg.chromadb_path)
    elif cfg.backend == "pgvector":
        store = PgVectorStore(
            dsn=cfg.pgvector_dsn,
            table_name=cfg.pgvector_table_name,
            embedding_dim=cfg.embedding_dim,
        )
    else:  # defensive, already validated
        raise typer.BadParameter(f"Unsupported backend: {cfg.backend}")

    console.print(
        f"[bold green]Storing {len(chunks)} embeddings into backend '{cfg.backend}'...[/bold green]"
    )

    # Very simple metadata example; can be extended as needed.
    metadatas = [{"source": "excel", "index": i} for i in range(len(chunks))]

    # Track progress if desired (per-chunk progress not strictly needed here).
    # We wrap the call for visual feedback.
    for _ in track(range(1), description="Writing to vector store..."):
        store.add_embeddings(embeddings=embeddings, texts=chunks, metadatas=metadatas)

    console.print("[bold green]Ingestion completed successfully.[/bold green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


