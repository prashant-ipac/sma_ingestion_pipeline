"""
Data loading utilities for Excel-based social media datasets.
"""

from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .data_formatter import format_from_excel_row
from .logging_utils import get_logger

logger = get_logger(__name__)


def _suggest_columns(expected: List[str], actual: List[str]) -> Dict[str, str]:
    """
    For each expected col, suggest a close match from actual columns.
    """
    suggestions: Dict[str, str] = {}
    actual_lower = [c.lower() for c in actual]
    for exp in expected:
        matches = difflib.get_close_matches(exp.lower(), actual_lower, n=1, cutoff=0.7)
        if matches:
            # find original casing
            idx = actual_lower.index(matches[0])
            suggestions[exp] = actual[idx]
    return suggestions


def _resolve_available_columns(df: pd.DataFrame, text_columns: List[str]) -> List[str]:
    """
    Case-insensitive matching between expected and actual column names.
    """
    lower_to_actual = {str(c).lower(): str(c) for c in df.columns}
    available_cols: List[str] = []
    for expected in text_columns:
        if expected in df.columns:
            available_cols.append(expected)
        else:
            lowered = expected.lower()
            if lowered in lower_to_actual:
                available_cols.append(lower_to_actual[lowered])
    return available_cols


def _read_excel_with_logs(excel_path: str, sheet_name: Optional[str]):
    """
    Read Excel with strong logs + timing.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(
        "Reading Excel: path=%s sheet=%s size=%.2fMB",
        str(path),
        sheet_name or "<default>",
        file_size_mb,
    )

    # If user passed a sheet name string, log available sheets (helps debug typos).
    # This is cheap compared to full read (pandas reads workbook metadata).
    if sheet_name and isinstance(sheet_name, str):
        try:
            xl = pd.ExcelFile(excel_path)
            sheets = xl.sheet_names
            if sheet_name not in sheets:
                suggestion = difflib.get_close_matches(sheet_name, sheets, n=1, cutoff=0.6)
                logger.warning(
                    "Sheet '%s' not found. Available sheets=%s%s",
                    sheet_name,
                    sheets,
                    f" | did you mean '{suggestion[0]}'?" if suggestion else "",
                )
        except Exception as e:
            logger.debug("Unable to inspect Excel sheets: %s", e, exc_info=True)

    t0 = time.time()
    df = pd.read_excel(excel_path, sheet_name=sheet_name or 0)
    dt = time.time() - t0

    logger.info(
        "Excel read complete: rows=%d cols=%d time=%.2fs",
        df.shape[0],
        df.shape[1],
        dt,
    )
    logger.debug("Excel columns: %s", list(df.columns))
    return df


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
        "load_texts_from_excel(): sheet=%s expected_columns=%s",
        sheet_name or "<default>",
        text_columns,
    )

    df = _read_excel_with_logs(excel_path, sheet_name)

    if df.empty:
        logger.warning("Excel dataframe is empty (0 rows). Returning empty texts.")
        return []

    available_cols = _resolve_available_columns(df, text_columns)

    if not available_cols:
        suggestions = _suggest_columns(text_columns, list(df.columns))
        logger.error(
            "None of expected text columns found. expected=%s available=%s suggestions=%s",
            text_columns,
            list(df.columns),
            suggestions or "<none>",
        )
        raise ValueError(
            f"None of the expected text columns {text_columns} were found. "
            f"Available columns: {list(df.columns)}. "
            f"Suggestions: {suggestions or 'none'}"
        )

    logger.info("Using text columns: %s", available_cols)

    t0 = time.time()
    texts: List[str] = []
    for col in available_cols:
        # dropna() removes NaNs, but can still include 'nan' strings if converted incorrectly
        col_values: Iterable[str] = df[col].dropna().astype(str).tolist()
        # filter blanks
        cleaned = [v for v in col_values if v and str(v).strip() and str(v).strip().lower() != "nan"]
        texts.extend(cleaned)
        logger.debug("Column '%s': raw=%d cleaned=%d", col, len(col_values), len(cleaned))

    dt = time.time() - t0
    logger.info("Loaded %d text entries from Excel in %.2fs", len(texts), dt)

    if len(texts) == 0:
        logger.warning(
            "No non-empty texts extracted from columns=%s. Check column values / sheet.",
            available_cols,
        )

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
    """
    logger.info(
        "load_structured_data_from_excel(): sheet=%s expected_columns=%s embedding_model=%s",
        sheet_name or "<default>",
        text_columns,
        embedding_model,
    )

    df = _read_excel_with_logs(excel_path, sheet_name)
    file_name = Path(excel_path).name

    if df.empty:
        logger.warning("Excel dataframe is empty (0 rows). Returning empty results.")
        return [], []

    available_cols = _resolve_available_columns(df, text_columns)
    if not available_cols:
        suggestions = _suggest_columns(text_columns, list(df.columns))
        logger.error(
            "None of expected text columns found. expected=%s available=%s suggestions=%s",
            text_columns,
            list(df.columns),
            suggestions or "<none>",
        )
        raise ValueError(
            f"None of the expected text columns {text_columns} were found. "
            f"Available columns: {list(df.columns)}. "
            f"Suggestions: {suggestions or 'none'}"
        )

    logger.info("Using text columns: %s", available_cols)

    texts: List[str] = []
    payloads: List[Dict] = []

    t0 = time.time()
    used_rows = 0
    blank_rows = 0

    # Process each row
    for idx, row in df.iterrows():
        found_text = False
        for col in available_cols:
            text = str(row[col]) if pd.notna(row[col]) else ""
            if text and text.strip() and text.strip().lower() != "nan":
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
                used_rows += 1
                found_text = True
                break  # Only use one text column per row

        if not found_text:
            blank_rows += 1

    dt = time.time() - t0
    logger.info(
        "Structured load complete: rows=%d used_rows=%d blank_rows=%d texts=%d time=%.2fs",
        df.shape[0],
        used_rows,
        blank_rows,
        len(texts),
        dt,
    )

    # Helpful warning if extraction is unexpectedly low
    if df.shape[0] > 0:
        coverage = used_rows / df.shape[0]
        if coverage < 0.2:
            logger.warning(
                "Low extraction coverage (%.1f%%). Check text_columns=%s and sheet=%s.",
                coverage * 100,
                available_cols,
                sheet_name or "<default>",
            )

    return texts, payloads
