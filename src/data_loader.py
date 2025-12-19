"""
Data loading utilities for Excel-based social media datasets.
"""

from typing import Iterable, List, Optional

import pandas as pd

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



