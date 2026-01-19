"""
Data formatting utilities to structure social media data according to the vector store schema.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from .logging_utils import get_logger

logger = get_logger(__name__)

# Log only a few sample payloads (to avoid flooding logs)
# Set in env: LOG_SAMPLE_ROWS=5
_SAMPLE_N = int(os.getenv("LOG_SAMPLE_ROWS", "0") or "0")
_sampled = 0


def _is_nan_like(val: Any) -> bool:
    # pandas NaN is float and prints "nan"
    try:
        if val is None:
            return True
        if isinstance(val, float) and str(val).lower() == "nan":
            return True
        if isinstance(val, str) and val.strip().lower() in ("", "nan", "none", "null"):
            return True
    except Exception:
        pass
    return False


def _parse_int(val: Any, field: str, row_number: Optional[int] = None) -> Optional[int]:
    if _is_nan_like(val):
        return None
    try:
        return int(val)
    except Exception:
        logger.debug("Failed int parse: field=%s value=%r row=%s", field, val, row_number)
        return None


def _parse_csv_list(val: Any) -> Optional[List[str]]:
    if _is_nan_like(val):
        return None
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",") if p.strip()]
        return parts
    # unknown type -> ignore
    return None


def _parse_timestamp(created_at: Optional[str], epoch: Optional[int]) -> Dict[str, Any]:
    """
    Returns timestamp dict. Logs warnings only for invalid formats.
    """
    if created_at and not _is_nan_like(created_at):
        try:
            dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            return {
                "created_at": str(created_at),
                "epoch": int(dt.timestamp()),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
            }
        except Exception:
            logger.warning("Invalid created_at timestamp format: %r", created_at)
            return {"created_at": str(created_at)}

    if epoch is not None:
        try:
            dt = datetime.fromtimestamp(int(epoch))
            return {
                "created_at": dt.isoformat() + "Z",
                "epoch": int(epoch),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
            }
        except Exception:
            logger.warning("Invalid epoch timestamp: %r", epoch)
            return {"epoch": epoch}

    # default: now
    now = datetime.utcnow()
    return {
        "created_at": now.isoformat() + "Z",
        "epoch": int(now.timestamp()),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
    }


def create_payload(
    text: str,
    platform: Optional[str] = None,
    platform_post_id: Optional[str] = None,
    author_id: Optional[str] = None,
    author_name: Optional[str] = None,
    language: Optional[str] = None,
    hashtags: Optional[List[str]] = None,
    mentions: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    media_type: str = "text",
    media_urls: Optional[List[str]] = None,
    thumbnail_url: Optional[str] = None,
    likes: Optional[int] = None,
    comments: Optional[int] = None,
    shares: Optional[int] = None,
    views: Optional[int] = None,
    created_at: Optional[str] = None,
    epoch: Optional[int] = None,
    ingested_from: str = "excel",
    file_name: Optional[str] = None,
    row_number: Optional[int] = None,
    content_type: str = "post",
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a payload dictionary according to the vector store schema.
    """
    timestamp_data = _parse_timestamp(created_at=created_at, epoch=epoch)

    payload: Dict[str, Any] = {
        "platform": platform or "unknown",
        "platform_post_id": platform_post_id or "",
        "author_id": author_id or "",
        "author_name": author_name or "",
        "content": {
            "text": text,
            "language": language or "en",
            "hashtags": hashtags or [],
            "mentions": mentions or [],
            "urls": urls or [],
        },
        "media": {
            "type": media_type,
            "media_urls": media_urls or [],
            "thumbnail_url": thumbnail_url,
        },
        "engagement": {
            "likes": likes or 0,
            "comments": comments or 0,
            "shares": shares or 0,
            "views": views or 0,
        },
        "timestamp": timestamp_data,
        "source": {
            "ingested_from": ingested_from,
            "file_name": file_name or "",
            "row_number": row_number or 0,
        },
        "content_type": content_type,
        "embedding_model": embedding_model or "",
    }

    return payload


def create_vector_store_entry(
    vector: List[float],
    payload: Dict[str, Any],
    entry_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a complete vector store entry with ID, vector, and payload.
    """
    if entry_id is None:
        entry_id = str(uuid.uuid4())

    return {
        "id": entry_id,
        "vector": vector,
        "payload": payload,
    }


def format_from_excel_row(
    row: Mapping[str, Any],
    text_column: str,
    file_name: str,
    row_number: int,
    embedding_model: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Format a single Excel row into the vector store payload format.
    """
    if column_mapping is None:
        column_mapping = {}

    def get_value(key: str, default: Any = None) -> Any:
        col_name = column_mapping.get(key, key)

        if col_name in row:
            val = row[col_name]
            if _is_nan_like(val):
                return default
            return val

        if key in row:
            val = row[key]
            if _is_nan_like(val):
                return default
            return val

        return default

    # Extract text
    text = str(get_value(text_column, "") or "")
    if text.strip().lower() == "nan":
        text = ""

    # Extract structured fields
    platform = get_value("platform") or get_value("Platform")
    platform_post_id = get_value("platform_post_id") or get_value("post_id") or get_value("id")
    author_id = get_value("author_id") or get_value("Author ID")
    author_name = get_value("author_name") or get_value("Author") or get_value("author")
    language = get_value("language") or get_value("Language")

    # Lists
    hashtags = _parse_csv_list(get_value("hashtags") or get_value("Hashtags"))
    mentions = _parse_csv_list(get_value("mentions") or get_value("Mentions"))
    urls = _parse_csv_list(get_value("urls") or get_value("URLs"))

    media_type = get_value("media_type") or get_value("Media Type") or "text"
    media_urls = _parse_csv_list(get_value("media_urls") or get_value("Media URLs"))

    # Engagement metrics (log only at DEBUG when parsing fails)
    likes = _parse_int(get_value("likes") or get_value("Likes"), "likes", row_number=row_number)
    comments = _parse_int(get_value("comments") or get_value("Comments"), "comments", row_number=row_number)
    shares = _parse_int(get_value("shares") or get_value("Shares") or get_value("Retweets"), "shares", row_number=row_number)
    views = _parse_int(get_value("views") or get_value("Views"), "views", row_number=row_number)

    # Timestamp
    created_at = get_value("created_at") or get_value("Created At") or get_value("timestamp")
    epoch_raw = get_value("epoch") or get_value("Epoch")
    epoch = _parse_int(epoch_raw, "epoch", row_number=row_number)

    content_type = get_value("content_type") or get_value("Content Type") or "post"

    payload = create_payload(
        text=text,
        platform=str(platform) if platform else None,
        platform_post_id=str(platform_post_id) if platform_post_id else None,
        author_id=str(author_id) if author_id else None,
        author_name=str(author_name) if author_name else None,
        language=str(language) if language else None,
        hashtags=hashtags,
        mentions=mentions,
        urls=urls,
        media_type=str(media_type),
        media_urls=media_urls,
        thumbnail_url=get_value("thumbnail_url") or get_value("Thumbnail URL"),
        likes=likes,
        comments=comments,
        shares=shares,
        views=views,
        created_at=str(created_at) if created_at and not _is_nan_like(created_at) else None,
        epoch=epoch,
        ingested_from="excel",
        file_name=file_name,
        row_number=row_number,
        content_type=str(content_type),
        embedding_model=embedding_model,
    )

    # Sample a few payloads when enabled (helps confirm mappings)
    global _sampled
    if _SAMPLE_N > 0 and _sampled < _SAMPLE_N:
        _sampled += 1
        logger.debug(
            "Sample payload (row=%d): platform=%s author=%s content_type=%s text_len=%d created_at=%s",
            row_number,
            payload.get("platform"),
            payload.get("author_name"),
            payload.get("content_type"),
            len(payload.get("content", {}).get("text", "") or ""),
            payload.get("timestamp", {}).get("created_at"),
        )

    return payload
