"""Upload -> validate -> classify in batches -> enriched table."""
import io
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from langchain_core.runnables import Runnable

from .schemas import ReviewAnalysis

REQUIRED_COLUMNS = ["review_id", "date", "platform", "rating", "review_text"]
OPTIONAL_COLUMNS = ["branch"]  # filled with "Unknown" when the CSV has no such column
INPUT_COLUMNS = ["review_id", "date", "platform", "branch", "rating", "review_text"]
ANALYSIS_COLUMNS = list(ReviewAnalysis.model_fields)
ENRICHED_COLUMNS = INPUT_COLUMNS + ANALYSIS_COLUMNS + ["needs_review", "error"]
MAX_REVIEW_CHARS = 1500


@dataclass
class ValidationResult:
    df: pd.DataFrame
    rows_read: int
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV. Handles UTF-8 (with or without BOM, as Excel saves it)."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
    try:
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read the file as a CSV: {exc}") from exc


def validate_reviews(raw: pd.DataFrame, max_rows: int | None = None) -> ValidationResult:
    """Check columns, drop unusable rows, and report exactly what was dropped."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}."
        )
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[INPUT_COLUMNS].copy()
    rows_read = len(df)
    dropped: dict[str, int] = {}
    warnings: list[str] = []

    for col in ("review_id", "platform", "branch", "review_text"):
        df[col] = df[col].astype(str).str.strip()
    df["branch"] = df["branch"].replace("", "Unknown")

    empty = df["review_text"] == ""
    if empty.any():
        dropped["empty review_text"] = int(empty.sum())
        df = df[~empty]

    blank_id = df["review_id"] == ""
    if blank_id.any():
        dropped["missing review_id"] = int(blank_id.sum())
        df = df[~blank_id]

    dup = df["review_id"].duplicated(keep="first")
    if dup.any():
        dropped["duplicate review_id"] = int(dup.sum())
        df = df[~dup]

    df["platform"] = df["platform"].str.title().replace("", "Unknown")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    if df["date"].isna().any():
        warnings.append(f"{int(df['date'].isna().sum())} row(s) have an unreadable date.")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    bad_rating = df["rating"].notna() & ~df["rating"].between(1, 5)
    df.loc[bad_rating, "rating"] = pd.NA
    if df["rating"].isna().any():
        warnings.append(f"{int(df['rating'].isna().sum())} row(s) have a missing or invalid rating (1-5).")

    too_long = df["review_text"].str.len() > MAX_REVIEW_CHARS
    if too_long.any():
        df.loc[too_long, "review_text"] = df.loc[too_long, "review_text"].str[:MAX_REVIEW_CHARS]
        warnings.append(f"{int(too_long.sum())} very long review(s) were cut to {MAX_REVIEW_CHARS} characters.")

    if max_rows is not None and len(df) > max_rows:
        warnings.append(f"Only the first {max_rows} of {len(df)} valid rows will be analysed.")
        df = df.head(max_rows)

    return ValidationResult(df.reset_index(drop=True), rows_read, dropped, warnings)


def analyze_reviews(
    df: pd.DataFrame,
    chain: Runnable,
    *,
    max_concurrency: int = 8,
    chunk_size: int = 25,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run the review chain over every row with ``chain.batch``.

    A review that still fails after the chain's retries is kept in the table and marked
    ``needs_review=True`` instead of crashing the whole run.
    """
    inputs = [
        {
            "platform": row.platform,
            "rating": "unknown" if pd.isna(row.rating) else f"{int(row.rating)}/5",
            "review_text": row.review_text,
        }
        for row in df.itertuples()
    ]
    results: list = []
    total = len(inputs)
    for start in range(0, total, chunk_size):
        chunk = inputs[start : start + chunk_size]
        results.extend(
            chain.batch(chunk, config={"max_concurrency": max_concurrency}, return_exceptions=True)
        )
        if on_progress:
            on_progress(min(start + chunk_size, total), total)

    rows = []
    for (_, base), result in zip(df.iterrows(), results):
        row = base.to_dict()
        if isinstance(result, ReviewAnalysis):
            row.update(result.model_dump())
            row.update(needs_review=False, error="")
        else:
            row.update({c: pd.NA for c in ANALYSIS_COLUMNS})
            row.update(needs_review=True, error=f"{type(result).__name__}: {result}"[:300])
        rows.append(row)
    return pd.DataFrame(rows, columns=ENRICHED_COLUMNS)
