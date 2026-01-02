"""
Project-wide constants that are unlikely to change at runtime.
"""

from typing import Final, List

SUPPORTED_BACKENDS: Final[List[str]] = ["s3", "chromadb", "pgvector", "milvus"]

SUPPORTED_CHUNKING_STRATEGIES: Final[List[str]] = [
    "recursive",
    "fixed",
]

# Default Excel-related constants
DEFAULT_EXCEL_SHEET_NAME: Final[str] = "Sheet1"

# Reasonable guesses for column names in the Excel file
DEFAULT_TEXT_COLUMNS: Final[List[str]] = [
    "post",
    "comment",
    "text",
    "message",
]

# Default embedding dimension for intfloat/e5-large-v2
DEFAULT_EMBEDDING_DIM: Final[int] = 1024


