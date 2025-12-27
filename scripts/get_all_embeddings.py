"""
Script to retrieve and display all embeddings from ChromaDB vector store.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from src.config import Config
from src.logging_utils import configure_logging
from src.vector_store.chroma_vector_store import ChromaVectorStore


app = typer.Typer(help="Retrieve all embeddings from ChromaDB")
console = Console()


@app.command()
def get_all(
    chromadb_path: str = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to ChromaDB store (defaults to config value)",
    ),
    collection_name: str = typer.Option(
        "social_media_embeddings",
        "--collection",
        "-c",
        help="Collection name",
    ),
    output_format: str = typer.Option(
        "summary",
        "--format",
        "-f",
        help="Output format: 'summary', 'detailed', 'numpy', or 'csv'",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of embeddings to display (for detailed view)",
    ),
    save_path: str = typer.Option(
        None,
        "--save",
        "-s",
        help="Save embeddings to .npy file (optional)",
    ),
) -> None:
    """
    Retrieve all embeddings from ChromaDB and display them.
    """
    cfg = Config()
    configure_logging(cfg.log_level)

    # Use provided path or config default
    db_path = chromadb_path or cfg.chromadb_path

    console.print(f"[bold cyan]Loading ChromaDB from:[/bold cyan] {db_path}")
    console.print(f"[bold cyan]Collection:[/bold cyan] {collection_name}")

    try:
        store = ChromaVectorStore(path=db_path, collection_name=collection_name)
        data = store.get_all_embeddings(include_texts=True, include_metadatas=True)

        embeddings = data["embeddings"]
        ids = data["ids"]
        texts = data.get("texts", [])
        metadatas = data.get("metadatas", [])

        if len(embeddings) == 0:
            console.print("[yellow]No embeddings found in the collection.[/yellow]")
            return

        # Save to file if requested
        if save_path:
            np.save(save_path, embeddings)
            console.print(f"[green]Saved embeddings to:[/green] {save_path}")

        # Display based on format
        if output_format == "summary":
            _display_summary(embeddings, ids, texts, metadatas)
        elif output_format == "detailed":
            _display_detailed(embeddings, ids, texts, metadatas, limit)
        elif output_format == "numpy":
            _display_numpy(embeddings, ids)
        elif output_format == "csv":
            _display_csv(embeddings, ids, texts, metadatas)
        else:
            console.print(f"[red]Unknown format: {output_format}[/red]")
            raise typer.BadParameter(f"Format must be one of: summary, detailed, numpy, csv")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _display_summary(
    embeddings: np.ndarray,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    """Display summary statistics."""
    console.print("\n[bold green]=== Embeddings Summary ===[/bold green]\n")

    table = Table(title="Collection Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Embeddings", str(len(embeddings)))
    table.add_row("Embedding Dimension", str(embeddings.shape[1]) if len(embeddings) > 0 else "N/A")
    table.add_row("Data Type", str(embeddings.dtype))
    table.add_row("Memory Size (MB)", f"{embeddings.nbytes / (1024 * 1024):.2f}")

    if texts:
        avg_text_length = sum(len(t) for t in texts) / len(texts) if texts else 0
        table.add_row("Avg Text Length", f"{avg_text_length:.1f}")

    if metadatas:
        table.add_row("Has Metadata", "Yes")
        if metadatas and metadatas[0]:
            sample_keys = list(metadatas[0].keys())
            table.add_row("Metadata Keys", ", ".join(sample_keys[:5]))

    console.print(table)

    # Show sample embeddings
    console.print("\n[bold green]=== Sample Embeddings (first 3) ===[/bold green]\n")
    for i in range(min(3, len(embeddings))):
        console.print(f"[cyan]ID:[/cyan] {ids[i]}")
        if texts:
            text_preview = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
            console.print(f"[cyan]Text:[/cyan] {text_preview}")
        console.print(f"[cyan]Embedding shape:[/cyan] {embeddings[i].shape}")
        console.print(f"[cyan]First 5 values:[/cyan] {embeddings[i][:5]}")
        if metadatas and metadatas[i]:
            console.print(f"[cyan]Metadata:[/cyan] {metadatas[i]}")
        console.print()


def _display_detailed(
    embeddings: np.ndarray,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    limit: int | None,
) -> None:
    """Display detailed view of embeddings."""
    console.print("\n[bold green]=== Detailed Embeddings View ===[/bold green]\n")

    display_count = limit if limit else len(embeddings)
    display_count = min(display_count, len(embeddings))

    table = Table(title=f"Embeddings (showing {display_count} of {len(embeddings)})")
    table.add_column("ID", style="cyan")
    table.add_column("Text Preview", style="yellow", max_width=50)
    table.add_column("Embedding Shape", style="green")
    table.add_column("Metadata", style="magenta", max_width=30)

    for i in range(display_count):
        text_preview = texts[i][:50] + "..." if texts and len(texts[i]) > 50 else (texts[i] if texts else "N/A")
        metadata_str = str(metadatas[i])[:30] + "..." if metadatas and metadatas[i] else "{}"
        table.add_row(
            ids[i],
            text_preview,
            str(embeddings[i].shape),
            metadata_str,
        )

    console.print(table)


def _display_numpy(
    embeddings: np.ndarray,
    ids: list[str],
) -> None:
    """Display numpy array information."""
    console.print("\n[bold green]=== NumPy Array Information ===[/bold green]\n")
    console.print(f"Shape: {embeddings.shape}")
    console.print(f"Dtype: {embeddings.dtype}")
    console.print(f"Min value: {embeddings.min():.6f}")
    console.print(f"Max value: {embeddings.max():.6f}")
    console.print(f"Mean value: {embeddings.mean():.6f}")
    console.print(f"Std deviation: {embeddings.std():.6f}")
    console.print(f"\nFirst embedding:\n{embeddings[0]}")


def _display_csv(
    embeddings: np.ndarray,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    """Display embeddings in CSV-like format."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["ID", "Text"]
    if metadatas and metadatas[0]:
        header.extend(metadatas[0].keys())
    header.append("Embedding_Vector")
    writer.writerow(header)

    # Data rows
    for i in range(len(embeddings)):
        row = [ids[i]]
        row.append(texts[i] if texts else "")
        if metadatas and metadatas[i]:
            row.extend([str(metadatas[i].get(k, "")) for k in metadatas[0].keys()])
        row.append(",".join(map(str, embeddings[i])))
        writer.writerow(row)

    console.print(output.getvalue())


if __name__ == "__main__":
    app()

