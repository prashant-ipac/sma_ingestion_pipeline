"""
Simple logging configuration utility.
"""

import logging
from typing import Optional


def configure_logging(level: str = "INFO") -> None:
    numeric_level: int = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or __name__)



