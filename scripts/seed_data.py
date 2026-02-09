import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


def normalize_platform(sheet_name: str) -> str:
    return sheet_name.strip().lower()


def normalize_post_type(raw: Optional[str], platform: str) -> str:
    if not raw or str(raw).strip() == "":
        return "video" if platform == "youtube" else "other"
    v = str(raw).strip().lower()
    if v in {"photo", "image", "img", "picture"}:
        return "image"
    if v in {"video", "reel", "short", "shorts"}:
        return "video"
    if v in {"text"}:
        return "text"
    if v in {"link", "url"}:
        return "link"
    return "other"


def parse_date(val) -> Optional[datetime.date]:
    """
    Handles:
    - 'YYYY-MM-DD' strings
    - 'DD-MM-YYYY' strings
    - pandas Timestamp
    """
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return None

    # pandas Timestamp
    if hasattr(val, "to_pydatetime"):
        dt = val.to_pydatetime()
        return dt.date()

    s = str(val).strip()
    if not s:
        return None

    # Try ISO first
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        pass

    # Try day-first
    try:
        return datetime.strptime(s, "%d-%m-%Y").date()
    except Exception:
        pass

    # Try common alt
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True).date()
    except Exception:
        return None


def build_rows_from_sheet(df: pd.DataFrame, sheet: str, source_file: str, n: int):
    platform = normalize_platform(sheet)
    
    # Define expected columns that are explicitly extracted
    expected_columns = {
        "State", "Date", "Username", "Message", "text", "Post Link", 
        "Video id", "Post Type", "Engagement"
    }

    rows = []
    for i, r in df.iterrows():
        state = r.get("State")
        date_val = r.get("Date")
        posted_date = parse_date(date_val)

        username = r.get("Username")

        # text field differs for Instagram
        if "Message" in df.columns:
            text = r.get("Message")
        else:
            text = r.get("text")

        post_url = r.get("Post Link") if "Post Link" in df.columns else None
        video_id = r.get("Video id") if "Video id" in df.columns else None

        # Build youtube url if missing
        if platform == "youtube" and video_id and (not post_url or str(post_url).strip() == ""):
            post_url = f"https://www.youtube.com/watch?v={str(video_id).strip()}"

        post_type_raw = r.get("Post Type") if "Post Type" in df.columns else ("video" if platform == "youtube" else None)
        post_type_norm = normalize_post_type(post_type_raw, platform)

        engagement = r.get("Engagement") if "Engagement" in df.columns else None
        try:
            engagement = int(engagement) if engagement is not None and str(engagement) != "nan" else None
        except Exception:
            engagement = None

        row_id = str(uuid.uuid4())

        # source_row_number:
        # pandas index 0 corresponds to excel row 2 (row 1 is header)
        source_row_number = int(i) + 2
        
        # Collect all columns not explicitly mapped into extras
        extras_dict = {}
        for col in df.columns:
            if col not in expected_columns:
                val = r.get(col)
                # Only include non-null, non-nan values
                if val is not None and not (isinstance(val, float) and str(val) == "nan"):
                    # Convert pandas types to serializable types
                    if hasattr(val, "to_pydatetime"):
                        extras_dict[col] = val.to_pydatetime().isoformat()
                    else:
                        extras_dict[col] = str(val) if not isinstance(val, (str, int, float, bool)) else val
        
        extras_json = json.dumps(extras_dict)

        rows.append(
            (
                row_id,                 # id
                platform,               # platform
                None,                   # platform_post_id (optional later)
                state,                  # state
                posted_date,            # posted_date
                username,               # username
                post_url,               # post_url
                str(video_id).strip() if video_id is not None and str(video_id) != "nan" else None,  # video_id
                str(post_type_raw) if post_type_raw is not None and str(post_type_raw) != "nan" else None,  # post_type_raw
                post_type_norm,         # post_type_norm
                str(text) if text is not None and str(text) != "nan" else None,  # text
                engagement,             # engagement_total
                None,                   # embedding (NULL for dummy)
                os.getenv("EMBEDDING_MODEL", None),          # embedding_model (optional)
                os.getenv("EMBEDDING_PROVIDER", None),       # embedding_provider (optional)
                extras_json,            # extras jsonb as text
                "excel",                # ingested_from
                source_file,            # source_file
                sheet,                  # source_sheet
                source_row_number,      # source_row_number
            )
        )

    return rows


def main():
    load_dotenv()

    excel_path = os.getenv("EXCEL_PATH", r"src/data/Vector_DB_platform_wise_data_23March_13Dec_2025.xlsx")
    dsn = os.getenv("PGVECTOR_DSN", "postgresql://postgres:postgress@localhost:5433/sma-data")
    table = os.getenv("PGVECTOR_TABLE_NAME", "social_media_posts")
    n_per_sheet = int(os.getenv("SEED_ROWS_PER_SHEET", "5"))

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    source_file = os.path.basename(excel_path)

    sheets = ["Facebook", "Twitter", "Instagram", "Youtube"]
    all_rows = []

    for sh in sheets:
        df = pd.read_excel(excel_path, sheet_name=sh)
        all_rows.extend(build_rows_from_sheet(df, sh, source_file, n_per_sheet))

    print(f"Inserting {len(all_rows)} rows into {table} ...")

    insert_sql = f"""
    INSERT INTO {table} (
      id, platform, platform_post_id,
      state, posted_date, username, post_url, video_id,
      post_type_raw, post_type_norm,
      text, engagement_total,
      embedding, embedding_model, embedding_provider,
      extras, ingested_from, source_file, source_sheet, source_row_number
    ) VALUES %s
    """

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, all_rows, page_size=100)
        conn.commit()

    print("Done ✅")


if __name__ == "__main__":
    main()