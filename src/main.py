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
from .data_loader import load_texts_from_excel, load_structured_data_from_excel
from .chunking import chunk_texts
from .embedding import EmbeddingModel
from .vector_store import S3VectorStore, ChromaVectorStore, PgVectorStore, MilvusVectorStore


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
        None, "--backend", "-b", help="Override vector store backend (s3|chromadb|pgvector|milvus)."
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

    # Load structured data with payloads
    try:
        texts, payloads = load_structured_data_from_excel(
            excel_path=excel_path,
            sheet_name=sheet_name or cfg.default_sheet_name,
            text_columns=cfg.text_columns,
            embedding_model=cfg.embedding_model,
        )
        console.print(f"[bold green]Loaded {len(texts)} structured entries from Excel[/bold green]")
    except Exception as e:
        logger.warning(f"Failed to load structured data, falling back to simple text loading: {e}")
        # Fallback to simple text loading
        texts = load_texts_from_excel(
            excel_path=excel_path,
            sheet_name=sheet_name or cfg.default_sheet_name,
            text_columns=cfg.text_columns,
        )
        from .data_formatter import create_payload
        from pathlib import Path
        payloads = [
            create_payload(
                text=text,
                ingested_from="excel",
                file_name=Path(excel_path).name,
                row_number=i + 1,
                embedding_model=cfg.embedding_model,
            )
            for i, text in enumerate(texts)
        ]

    # Chunk
    chunks = chunk_texts(
        texts=texts,
        strategy=cfg.chunking_strategy,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # When chunking, we need to map chunks back to original payloads
    # For now, we'll create new payloads for chunks that reference the original
    chunk_payloads = []
    chunk_idx = 0
    for i, text in enumerate(texts):
        # Find how many chunks this text produced
        # This is approximate - we'll use the original payload for each chunk
        # In a more sophisticated implementation, you'd track chunk-to-text mapping
        if chunk_idx < len(chunks):
            # Use the payload from the source text
            original_payload = payloads[i].copy()
            original_payload["source"]["chunk_index"] = chunk_idx
            chunk_payloads.append(original_payload)
            chunk_idx += 1

    # If we have fewer chunk payloads than chunks, pad with the last payload
    while len(chunk_payloads) < len(chunks):
        if payloads:
            chunk_payloads.append(payloads[-1].copy())
        else:
            from .data_formatter import create_payload
            from pathlib import Path
            chunk_payloads.append(
                create_payload(
                    text=chunks[len(chunk_payloads)],
                    ingested_from="excel",
                    file_name=Path(excel_path).name,
                    row_number=len(chunk_payloads) + 1,
                    embedding_model=cfg.embedding_model,
                )
            )

    # Embeddings
    model = EmbeddingModel(
        model_name=cfg.embedding_model,
        use_onnx=cfg.embedding_use_onnx,
        batch_size=cfg.embedding_batch_size,
        device=cfg.embedding_device,
    )
    embeddings = model.encode(chunks)

    # Update payloads with embedding model name
    for payload in chunk_payloads:
        payload["embedding_model"] = cfg.embedding_model

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
    elif cfg.backend == "milvus":
        store = MilvusVectorStore(
            host=cfg.milvus_host,
            port=cfg.milvus_port,
            collection_name=cfg.milvus_collection_name,
            embedding_dim=cfg.embedding_dim,
            user=cfg.milvus_user,
            password=cfg.milvus_password,
        )
    else:  # defensive, already validated
        raise typer.BadParameter(f"Unsupported backend: {cfg.backend}")

    console.print(
        f"[bold green]Storing {len(chunks)} embeddings into backend '{cfg.backend}'...[/bold green]"
    )

    # Track progress if desired (per-chunk progress not strictly needed here).
    # We wrap the call for visual feedback.
    for _ in track(range(1), description="Writing to vector store..."):
        store.add_embeddings(
            embeddings=embeddings,
            texts=chunks,
            payloads=chunk_payloads,
        )

    console.print("[bold green]Ingestion completed successfully.[/bold green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


