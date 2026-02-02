"""
Logging configuration utility.

Writes logs to:
- Console (stdout)
- Rotating log file (logs/sma_ingestion.log by default)
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure root logging. Call once early in main().

    Args:
        level: INFO/DEBUG/WARNING/ERROR
        log_file: path to log file. If None, reads LOG_FILE env or defaults to logs/sma_ingestion.log
        max_bytes: rotate when log reaches this size
        backup_count: number of rotated backups to keep
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Pick log file path
    log_file = log_file or os.getenv("LOG_FILE", "logs/sma_ingestion.log")

    # Ensure directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # IMPORTANT: avoid duplicate handlers if configure_logging() called again
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(fmt)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    root.info("Logging initialized: level=%s log_file=%s", level.upper(), str(log_path))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or __name__)
