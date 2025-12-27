"""
Data loading utilities for Excel-based social media datasets.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .data_formatter import format_from_excel_row
from .logging_utils import get_logger


logger = get_logger(__name__)


def load_texts_from_excel(
    excel_path: str,
    sheet_name: Optional[str],
    text_columns: List[str],
) -> List[str]:
    """
    Load textual data from an Excel file.

    The function scans the provided columns (in order) and collects all
    non-null text values.

    This is a legacy function for backward compatibility.
    For structured data, use load_structured_data_from_excel instead.
    """
    logger.info(
        "Loading Excel file '%s' (sheet=%s, columns=%s)",
        excel_path,
        sheet_name or "<default>",
        text_columns,
    )

    df = pd.read_excel(excel_path, sheet_name=sheet_name or 0)

    # Allow case-insensitive matching between expected and actual column names.
    lower_to_actual = {str(c).lower(): str(c) for c in df.columns}
    available_cols = []
    for expected in text_columns:
        if expected in df.columns:
            available_cols.append(expected)
        else:
            lowered = expected.lower()
            if lowered in lower_to_actual:
                available_cols.append(lower_to_actual[lowered])

    if not available_cols:
        raise ValueError(
            f"None of the expected text columns {text_columns} "
            f"were found in the Excel file. Available columns: {list(df.columns)}"
        )

    logger.info("Using text columns: %s", available_cols)

    texts: List[str] = []
    for col in available_cols:
        col_values: Iterable[str] = df[col].dropna().astype(str).tolist()
        texts.extend(col_values)

    logger.info("Loaded %d text entries from Excel", len(texts))
    return texts


def load_structured_data_from_excel(
    excel_path: str,
    sheet_name: Optional[str],
    text_columns: List[str],
    embedding_model: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[Dict]]:
    """
    Load structured data from an Excel file with full metadata extraction.

    Returns both texts and structured payloads according to the vector store schema.

    Args:
        excel_path: Path to the Excel file
        sheet_name: Name of the sheet to read (or None for default)
        text_columns: List of column names that contain text content
        embedding_model: Name of the embedding model (for payload)
        column_mapping: Optional mapping from standard fields to Excel column names

    Returns:
        Tuple of (texts, payloads) where:
            - texts: List of text strings
            - payloads: List of payload dictionaries
    """
    logger.info(
        "Loading structured data from Excel file '%s' (sheet=%s, columns=%s)",
        excel_path,
        sheet_name or "<default>",
        text_columns,
    )

    df = pd.read_excel(excel_path, sheet_name=sheet_name or 0)
    file_name = Path(excel_path).name

    # Allow case-insensitive matching between expected and actual column names.
    lower_to_actual = {str(c).lower(): str(c) for c in df.columns}
    available_cols = []
    for expected in text_columns:
        if expected in df.columns:
            available_cols.append(expected)
        else:
            lowered = expected.lower()
            if lowered in lower_to_actual:
                available_cols.append(lower_to_actual[lowered])

    if not available_cols:
        raise ValueError(
            f"None of the expected text columns {text_columns} "
            f"were found in the Excel file. Available columns: {list(df.columns)}"
        )

    logger.info("Using text columns: %s", available_cols)

    texts: List[str] = []
    payloads: List[Dict] = []

    # Process each row
    for idx, row in df.iterrows():
        # Use the first available text column for each row
        for col in available_cols:
            text = str(row[col]) if pd.notna(row[col]) else ""
            if text and text.strip():
                # Create payload from row data
                payload = format_from_excel_row(
                    row=row,
                    text_column=col,
                    file_name=file_name,
                    row_number=int(idx) + 1,  # 1-indexed
                    embedding_model=embedding_model,
                    column_mapping=column_mapping,
                )
                texts.append(text)
                payloads.append(payload)
                break  # Only use one text column per row

    logger.info("Loaded %d structured entries from Excel", len(texts))
    return texts, payloads



