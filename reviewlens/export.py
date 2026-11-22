"""CSV export that opens correctly in Excel."""
import pandas as pd

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _neutralise(value):
    """Stop customer text such as '=HYPERLINK(...)' from running as an Excel formula."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].map(_neutralise)
    # utf-8-sig adds a BOM so Excel shows Tamil script correctly
    return out.to_csv(index=False).encode("utf-8-sig")
