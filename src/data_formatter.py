"""
Data formatting utilities to structure social media data according to the vector store schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from .logging_utils import get_logger


logger = get_logger(__name__)


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

    Args:
        text: The main text content
        platform: Social media platform (e.g., "twitter", "facebook", "instagram")
        platform_post_id: Original post ID from the platform
        author_id: Author's unique identifier
        author_name: Author's display name/username
        language: Language code (e.g., "en", "es")
        hashtags: List of hashtags
        mentions: List of mentioned users/accounts
        urls: List of URLs in the content
        media_type: Type of media ("text", "image", "video", etc.)
        media_urls: List of media URLs
        thumbnail_url: Thumbnail image URL
        likes: Number of likes
        comments: Number of comments
        shares: Number of shares/retweets
        views: Number of views
        created_at: ISO format timestamp string
        epoch: Unix epoch timestamp
        ingested_from: Source of ingestion
        file_name: Name of the source file
        row_number: Row number in the source file
        content_type: Type of content ("post", "comment", "reply")
        embedding_model: Model used for embeddings

    Returns:
        Dictionary with the structured payload
    """
    # Parse timestamp if provided
    timestamp_data = {}
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            timestamp_data = {
                "created_at": created_at,
                "epoch": int(dt.timestamp()),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
            }
        except (ValueError, AttributeError):
            logger.warning(f"Invalid timestamp format: {created_at}")
            timestamp_data = {"created_at": created_at}
    elif epoch:
        try:
            dt = datetime.fromtimestamp(epoch)
            timestamp_data = {
                "created_at": dt.isoformat() + "Z",
                "epoch": epoch,
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
            }
        except (ValueError, OSError):
            logger.warning(f"Invalid epoch timestamp: {epoch}")
            timestamp_data = {"epoch": epoch}

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
        "timestamp": timestamp_data if timestamp_data else {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "epoch": int(datetime.utcnow().timestamp()),
            "year": datetime.utcnow().year,
            "month": datetime.utcnow().month,
            "day": datetime.utcnow().day,
            "hour": datetime.utcnow().hour,
        },
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

    Args:
        vector: The embedding vector as a list of floats
        payload: The payload dictionary (from create_payload)
        entry_id: Optional UUID string. If not provided, a new UUID will be generated.

    Returns:
        Dictionary with id, vector, and payload keys
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

    Args:
        row: Dictionary-like object representing an Excel row (e.g., pandas Series)
        text_column: Name of the column containing the text content
        file_name: Name of the Excel file
        row_number: Row number (1-indexed)
        embedding_model: Model used for embeddings
        column_mapping: Optional mapping from standard fields to Excel column names.
            Example: {"platform": "Platform", "author_name": "Author", ...}

    Returns:
        Payload dictionary
    """
    if column_mapping is None:
        column_mapping = {}

    def get_value(key: str, default: Any = None) -> Any:
        """Get value from row using column mapping or direct key lookup."""
        col_name = column_mapping.get(key, key)
        # Try both the mapped name and the original key
        if col_name in row:
            val = row[col_name]
            # Handle NaN/null values
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                return default
            return val
        if key in row:
            val = row[key]
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                return default
            return val
        return default

    # Extract text
    text = str(get_value(text_column, ""))

    # Extract structured fields
    platform = get_value("platform") or get_value("Platform")
    platform_post_id = get_value("platform_post_id") or get_value("post_id") or get_value("id")
    author_id = get_value("author_id") or get_value("Author ID")
    author_name = get_value("author_name") or get_value("Author") or get_value("author")
    language = get_value("language") or get_value("Language")
    
    # Parse lists from strings if needed
    hashtags = get_value("hashtags") or get_value("Hashtags")
    if isinstance(hashtags, str):
        hashtags = [h.strip() for h in hashtags.split(",") if h.strip()]
    
    mentions = get_value("mentions") or get_value("Mentions")
    if isinstance(mentions, str):
        mentions = [m.strip() for m in mentions.split(",") if m.strip()]
    
    urls = get_value("urls") or get_value("URLs")
    if isinstance(urls, str):
        urls = [u.strip() for u in urls.split(",") if u.strip()]

    media_type = get_value("media_type") or get_value("Media Type") or "text"
    media_urls = get_value("media_urls") or get_value("Media URLs")
    if isinstance(media_urls, str):
        media_urls = [u.strip() for u in media_urls.split(",") if u.strip()]

    # Engagement metrics
    likes = get_value("likes") or get_value("Likes")
    if likes is not None:
        try:
            likes = int(likes)
        except (ValueError, TypeError):
            likes = None

    comments = get_value("comments") or get_value("Comments")
    if comments is not None:
        try:
            comments = int(comments)
        except (ValueError, TypeError):
            comments = None

    shares = get_value("shares") or get_value("Shares") or get_value("Retweets")
    if shares is not None:
        try:
            shares = int(shares)
        except (ValueError, TypeError):
            shares = None

    views = get_value("views") or get_value("Views")
    if views is not None:
        try:
            views = int(views)
        except (ValueError, TypeError):
            views = None

    # Timestamp
    created_at = get_value("created_at") or get_value("Created At") or get_value("timestamp")
    epoch = get_value("epoch") or get_value("Epoch")
    if epoch is not None:
        try:
            epoch = int(epoch)
        except (ValueError, TypeError):
            epoch = None

    content_type = get_value("content_type") or get_value("Content Type") or "post"

    return create_payload(
        text=text,
        platform=platform,
        platform_post_id=str(platform_post_id) if platform_post_id else None,
        author_id=str(author_id) if author_id else None,
        author_name=str(author_name) if author_name else None,
        language=language,
        hashtags=hashtags if isinstance(hashtags, list) else None,
        mentions=mentions if isinstance(mentions, list) else None,
        urls=urls if isinstance(urls, list) else None,
        media_type=str(media_type),
        media_urls=media_urls if isinstance(media_urls, list) else None,
        thumbnail_url=get_value("thumbnail_url") or get_value("Thumbnail URL"),
        likes=likes,
        comments=comments,
        shares=shares,
        views=views,
        created_at=str(created_at) if created_at else None,
        epoch=epoch,
        ingested_from="excel",
        file_name=file_name,
        row_number=row_number,
        content_type=str(content_type),
        embedding_model=embedding_model,
    )

