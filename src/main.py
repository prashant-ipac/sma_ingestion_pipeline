"""
CLI entrypoint for the social_media_vectordb ingestion pipeline.
"""

from __future__ import annotations

import time
import difflib
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import track

from .config import Config
from .logging_utils import configure_logging, get_logger
from .data_loader import load_texts_from_excel, load_structured_data_from_excel
from .chunking import chunk_texts
from .embedding import EmbeddingModel
from .vector_store import (
    S3VectorStore,
    ChromaVectorStore,
    PgVectorStore,
    MilvusVectorStore,
    AtlasVectorStore,
)

app = typer.Typer(help="Social media Excel → embeddings → vector store pipeline")
console = Console()


@contextmanager
def log_step(logger, step_name: str, **kwargs):
    """
    Logs a step start/end with duration + optional context.
    """
    ctx = " ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
    logger.info("▶ START: %s%s", step_name, (f" | {ctx}" if ctx else ""))
    t0 = time.time()
    try:
        yield
        dt = time.time() - t0
        logger.info("✅ END: %s | %.2fs", step_name, dt)
    except Exception:
        dt = time.time() - t0
        logger.exception("❌ FAIL: %s | %.2fs", step_name, dt)
        raise


def _suggest(value: str, options: list[str]) -> str | None:
    matches = difflib.get_close_matches(value, options, n=1, cutoff=0.6)
    return matches[0] if matches else None


@app.command()
def ingest(
    excel_path: str = typer.Argument(..., help="Path to the Excel file with social media data."),
    sheet_name: Optional[str] = typer.Option(
        None, "--sheet-name", "-s", help="Sheet name in the Excel file (defaults to config/default)."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override vector store backend (s3|chromadb|pgvector|milvus|atlasdb)."
    ),
    chunking_strategy: Optional[str] = typer.Option(
        None, "--chunking-strategy", "-c", help="Override chunking strategy (recursive|fixed)."
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Override log level (DEBUG|INFO|WARNING|ERROR)."
    ),
) -> None:
    """
    Run the full ingestion pipeline:
    Excel → texts → chunks → embeddings → vector store.
    """

    # Create config first (loads .env)
    cfg = Config()

    # Configure logging ASAP (before any logger usage)
    configure_logging(log_level or cfg.log_level)
    logger = get_logger(__name__)

    # CLI override config
    if backend:
        cfg.backend = backend
    if chunking_strategy:
        cfg.chunking_strategy = chunking_strategy
    
    cfg.finalize()
    cfg.validate()

    logger.info("Resolved config summary: %s", cfg.summary())


    # Resolve sheet name
    resolved_sheet = sheet_name or cfg.default_sheet_name

    # Header logs (always show)
    console.print("[bold cyan]Starting ingestion pipeline...[/bold cyan]")
    console.print(f"[bold]Excel:[/bold] {excel_path}")
    console.print(f"[bold]Sheet:[/bold] {resolved_sheet}")
    console.print(f"[bold]Backend:[/bold] {cfg.backend}")
    console.print(f"[bold]Chunking:[/bold] {cfg.chunking_strategy}")
    console.print(f"[bold]Model:[/bold] {cfg.embedding_model}")
    console.print(f"[bold]Batch size:[/bold] {cfg.embedding_batch_size}")

    logger.info("Run parameters: excel_path=%s sheet=%s backend=%s chunking=%s model=%s",
                excel_path, resolved_sheet, cfg.backend, cfg.chunking_strategy, cfg.embedding_model)

    # Validate config with friendly suggestions
    try:
        cfg.validate()
    except Exception as e:
        # Provide "did you mean" hints for common typos
        if backend:
            from .constants import SUPPORTED_BACKENDS
            s = _suggest(cfg.backend, list(SUPPORTED_BACKENDS))
            if s:
                console.print(f"[bold red]Invalid backend '{cfg.backend}'. Did you mean '{s}'?[/bold red]")
        if chunking_strategy:
            from .constants import SUPPORTED_CHUNKING_STRATEGIES
            s = _suggest(cfg.chunking_strategy, list(SUPPORTED_CHUNKING_STRATEGIES))
            if s:
                console.print(f"[bold red]Invalid chunking strategy '{cfg.chunking_strategy}'. Did you mean '{s}'?[/bold red]")

        console.print(f"[bold red]Config validation failed:[/bold red] {e}")
        logger.exception("Config validation failed")
        raise typer.Exit(code=1)

    # Basic sanity checks
    if not Path(excel_path).exists():
        console.print(f"[bold red]Excel file not found:[/bold red] {excel_path}")
        logger.error("Excel file not found: %s", excel_path)
        raise typer.Exit(code=1)

    # ------------------------------------
    # 1) Load structured data
    # ------------------------------------
    with log_step(logger, "Load Excel data", sheet=resolved_sheet, text_columns=cfg.text_columns):
        try:
            texts, payloads = load_structured_data_from_excel(
                excel_path=excel_path,
                sheet_name=resolved_sheet,
                text_columns=cfg.text_columns,
                embedding_model=cfg.embedding_model,
            )
            logger.info("Loaded structured rows: texts=%d payloads=%d", len(texts), len(payloads))
            console.print(f"[bold green]Loaded {len(texts)} structured entries from Excel[/bold green]")
        except Exception as e:
            logger.warning("Structured loader failed; falling back to simple loader. Error=%s", e, exc_info=True)

            texts = load_texts_from_excel(
                excel_path=excel_path,
                sheet_name=resolved_sheet,
                text_columns=cfg.text_columns,
            )
            logger.info("Loaded plain texts: %d", len(texts))

            from .data_formatter import create_payload

            file_name = Path(excel_path).name
            payloads = [
                create_payload(
                    text=text,
                    ingested_from="excel",
                    file_name=file_name,
                    row_number=i + 1,
                    embedding_model=cfg.embedding_model,
                )
                for i, text in enumerate(texts)
            ]
            logger.info("Created fallback payloads: %d", len(payloads))

    if not texts:
        console.print("[bold yellow]No texts found to ingest. Exiting.[/bold yellow]")
        logger.warning("No texts found from Excel. Nothing to ingest.")
        raise typer.Exit(code=0)

    # ------------------------------------
    # 2) Chunking
    # ------------------------------------
    with log_step(logger, "Chunk texts", strategy=cfg.chunking_strategy, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap):
        chunks = chunk_texts(
            texts=texts,
            strategy=cfg.chunking_strategy,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        logger.info("Chunking output: chunks=%d (from texts=%d)", len(chunks), len(texts))
        console.print(f"[bold green]Chunked into {len(chunks)} chunks[/bold green]")

    if not chunks:
        console.print("[bold yellow]Chunking produced 0 chunks. Exiting.[/bold yellow]")
        logger.warning("Chunking produced 0 chunks. Check chunking strategy/settings.")
        raise typer.Exit(code=0)

    # ------------------------------------
    # 3) Map payloads to chunks
    # ------------------------------------
    with log_step(logger, "Map payloads to chunks", payloads=len(payloads), chunks=len(chunks)):
        # Your original mapping was approximate; keep it but add strong logging.
        chunk_payloads = []
        chunk_idx = 0

        for i in range(min(len(texts), len(payloads))):
            if chunk_idx >= len(chunks):
                break
            p = payloads[i].copy()
            p.setdefault("source", {})
            p["source"]["chunk_index"] = chunk_idx
            chunk_payloads.append(p)
            chunk_idx += 1

        # Pad if needed
        while len(chunk_payloads) < len(chunks):
            p = (payloads[-1].copy() if payloads else {"source": {}})
            p.setdefault("source", {})
            p["source"]["chunk_index"] = len(chunk_payloads)
            chunk_payloads.append(p)

        if len(chunk_payloads) != len(chunks):
            logger.warning("Payload mismatch: chunk_payloads=%d chunks=%d", len(chunk_payloads), len(chunks))
        else:
            logger.info("Payload mapping complete: %d payloads for %d chunks", len(chunk_payloads), len(chunks))

    # ------------------------------------
    # 4) Embeddings
    # ------------------------------------
    logger.info(
        "Embedding config: provider=%s model=%s batch_size=%d device=%s onnx=%s input_type=%s out_dim=%s dtype=%s",
        cfg.embedding_provider,
        cfg.embedding_model,
        cfg.embedding_batch_size,
        cfg.embedding_device,
        cfg.embedding_use_onnx,
        getattr(cfg, "voyage_input_type", None),
        getattr(cfg, "voyage_output_dimension", None),
        getattr(cfg, "voyage_output_dtype", None),
    )

    with log_step(logger, "Create embeddings", model=cfg.embedding_model, batch_size=cfg.embedding_batch_size):
        model = EmbeddingModel(
            model_name=cfg.embedding_model,
            provider=cfg.embedding_provider,
            batch_size=cfg.embedding_batch_size,
            device=cfg.embedding_device,
            use_onnx=cfg.embedding_use_onnx,
            voyage_api_key=cfg.voyage_api_key,
            voyage_input_type=cfg.voyage_input_type,
            voyage_truncation=cfg.voyage_truncation,
            voyage_output_dimension=cfg.voyage_output_dimension,
            voyage_output_dtype=cfg.voyage_output_dtype,
            voyage_timeout=cfg.voyage_timeout,
            voyage_max_retries=cfg.voyage_max_retries,
        )
        embeddings = model.encode(chunks)


        # Log shape
        try:
            emb_shape = getattr(embeddings, "shape", None)
        except Exception:
            emb_shape = None

        logger.info("Embeddings created: shape=%s dtype=%s", emb_shape, getattr(embeddings, "dtype", None))
        console.print(f"[bold green]Embeddings created: {emb_shape}[/bold green]")

    # Update payloads with embedding model name
    for p in chunk_payloads:
        p["embedding_model"] = cfg.embedding_model

    # ------------------------------------
    # 5) Select vector store backend
    # ------------------------------------
    with log_step(logger, "Init vector store", backend=cfg.backend):
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
        elif cfg.backend == "atlasdb":
            store = AtlasVectorStore(
                uri=cfg.atlasdb_uri,
                database_name=cfg.atlasdb_database_name,
                collection_name=(sheet_name or cfg.atlasdb_collection_name),
                embedding_dim=cfg.atlasdb_embedding_dim,         # use atlas-specific dim
                max_batch_size=cfg.atlasdb_max_batch_size,
                index_name=cfg.atlasdb_index_name,
            )
        else:
            raise typer.BadParameter(f"Unsupported backend: {cfg.backend}")

    console.print(f"[bold green]Storing {len(chunks)} embeddings into backend '{cfg.backend}'...[/bold green]")
    logger.info("Writing embeddings: backend=%s items=%d", cfg.backend, len(chunks))

    # ------------------------------------
    # 6) Write to store
    # ------------------------------------
    with log_step(logger, "Write embeddings to vector store", backend=cfg.backend, items=len(chunks)):
        for _ in track(range(1), description="Writing to vector store..."):
            store.add_embeddings(
                embeddings=embeddings,
                texts=chunks,
                payloads=chunk_payloads,
            )

    console.print("[bold green]Ingestion completed successfully.[/bold green]")
    logger.info("Ingestion completed successfully.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()



    #1234@Sma


