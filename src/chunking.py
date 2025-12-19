"""
Chunking utilities with pluggable strategies.
"""

from typing import Iterable, List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from .logging_utils import get_logger


logger = get_logger(__name__)


def chunk_texts_recursive(
    texts: Iterable[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: List[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    logger.info("Recursive chunking produced %d chunks", len(chunks))
    return chunks


def chunk_texts_fixed(
    texts: Iterable[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Simple character-based fixed-size chunking using CharacterTextSplitter.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: List[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    logger.info("Fixed chunking produced %d chunks", len(chunks))
    return chunks


def chunk_texts(
    texts: Iterable[str],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Dispatch function to select the chunking strategy.
    """
    strategy = strategy.lower()
    logger.info(
        "Chunking texts with strategy=%s, chunk_size=%d, overlap=%d",
        strategy,
        chunk_size,
        chunk_overlap,
    )

    if strategy == "recursive":
        return chunk_texts_recursive(texts, chunk_size, chunk_overlap)
    if strategy == "fixed":
        return chunk_texts_fixed(texts, chunk_size, chunk_overlap)

    raise ValueError(f"Unsupported chunking strategy: {strategy}")



