# """
# Simple logging configuration utility.
# """

# import logging
# from typing import Optional


# def configure_logging(level: str = "INFO") -> None:
#     numeric_level: int = getattr(logging, level.upper(), logging.INFO)
#     logging.basicConfig(
#         level=numeric_level,
#         format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
#     )


# def get_logger(name: Optional[str] = None) -> logging.Logger:
#     return logging.getLogger(name or __name__)



"""
Simple logging configuration utility.

Fixes common "no logs printing" issues by:
- forcing handler configuration (force=True) so basicConfig applies even if handlers already exist
- optionally using RichHandler when rich is installed
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(
    level: str = "INFO",
    *,
    use_rich: bool | None = None,
) -> None:
    """
    Configure root logging.

    Args:
        level: Log level string (DEBUG|INFO|WARNING|ERROR)
        use_rich: If True, use rich.logging.RichHandler for pretty console output.
                  If None, auto-enable when RICH_LOGGING=true in env (default: auto).
    """
    # Resolve level
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level: int = getattr(logging, level, logging.INFO)

    # Decide rich logging
    if use_rich is None:
        use_rich = os.getenv("RICH_LOGGING", "true").lower() in ("true", "1", "yes", "y", "on")

    # Common format (used when not using RichHandler)
    log_format = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    date_format = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

    handlers: list[logging.Handler] | None = None

    if use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            # RichHandler shows time/level/message nicely; avoid double time by omitting %(asctime)s
            handlers = [
                RichHandler(
                    rich_tracebacks=True,
                    tracebacks_show_locals=False,
                    show_time=True,
                    show_level=True,
                    show_path=False,
                )
            ]
            # When using RichHandler, format should not include asctime (Rich prints it)
            log_format = "%(name)s - %(message)s"
        except Exception:
            # If rich isn't available for some reason, fall back to basic logging.
            handlers = None

    # IMPORTANT: force=True ensures config is applied even if some other lib already configured logging.
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    # Optional: reduce noisy third-party logs (tune as needed)
    noisy = os.getenv("LOG_NOISY_LIBS", "urllib3,boto3,botocore,matplotlib,PIL").split(",")
    for name in [n.strip() for n in noisy if n.strip()]:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger. Assumes configure_logging() has been called once in the entrypoint.
    """
    return logging.getLogger(name or __name__)
