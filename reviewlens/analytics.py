"""Pure-pandas numbers behind the dashboard (no Streamlit, easy to test)."""
import pandas as pd

from .schemas import CATEGORY_LABELS


def stars(rating) -> str:
    if pd.isna(rating):
        return ""
    n = max(0, min(5, int(round(float(rating)))))
    return "★" * n + "☆" * (5 - n)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Analysed rows only, with clean date / rating types."""
    ok = df[~df["needs_review"].astype(bool)].copy()
    ok["date"] = pd.to_datetime(ok["date"], errors="coerce")
    ok["rating"] = pd.to_numeric(ok["rating"], errors="coerce")
    return ok


def _label(category: str) -> str:
    return CATEGORY_LABELS.get(category, str(category))


def split_weeks(ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(last 7 days up to the newest review, the 7 days before that)."""
    d = ok["date"].dropna()
    if d.empty:
        return ok.iloc[0:0], ok.iloc[0:0]
    latest = d.max()
    this = ok[ok["date"] > latest - pd.Timedelta(days=7)]
    prev = ok[(ok["date"] > latest - pd.Timedelta(days=14)) & (ok["date"] <= latest - pd.Timedelta(days=7))]
    return this, prev


def top_complaint(ok: pd.DataFrame) -> tuple[str, int] | None:
    neg = ok[(ok["sentiment"] == "negative") & (ok["category"] != "other")]
    if neg.empty:
        return None
    cat, n = neg["category"].value_counts().head(1).reset_index().iloc[0]
    return _label(cat), int(n)


def kpis(df: pd.DataFrame) -> dict:
    ok = _prep(df)
    this, prev = split_weeks(ok)
    avg = ok["rating"].mean()
    delta = None
    if this["rating"].notna().any() and prev["rating"].notna().any():
        delta = float(this["rating"].mean() - prev["rating"].mean())
    neg = int((ok["sentiment"] == "negative").sum())
    return {
        "total": len(ok),
        "this_week": len(this),
        "prev_week": len(prev),
        "avg_rating": None if pd.isna(avg) else float(avg),
        "avg_delta": delta,
        "negative_count": neg,
        "negative_share": 100 * neg / len(ok) if len(ok) else 0.0,
        "urgent": int((ok["urgency"] == "high").sum()),
        "top_complaint": top_complaint(ok),
    }


def complaints_by_category(df: pd.DataFrame) -> pd.DataFrame:
    neg = _prep(df)
    neg = neg[neg["sentiment"] == "negative"]
    out = neg["category"].value_counts().rename_axis("category").reset_index(name="reviews")
    out["complaint"] = out["category"].map(_label)
    return out


def sentiment_by_platform(df: pd.DataFrame) -> pd.DataFrame:
    return _prep(df).groupby(["platform", "sentiment"]).size().reset_index(name="reviews")


def platform_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Review counts per platform and sentiment, platforms ordered by total volume."""
    out = sentiment_by_platform(df)
    totals = out.groupby("platform")["reviews"].transform("sum")
    return out.assign(total=totals).sort_values(["total", "platform"], ascending=[False, True]).reset_index(drop=True)


def weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Per week: all reviews (bars) plus positive and negative counts (lines)."""
    ok = _prep(df)
    ok = ok[ok["date"].notna()]
    if ok.empty:
        return pd.DataFrame(columns=["week", "label", "reviews", "positive", "negative"])
    week = ok["date"].dt.to_period("W-SUN").dt.start_time
    g = ok.assign(week=week).groupby("week")
    out = pd.DataFrame({
        "reviews": g.size(),
        "positive": g["sentiment"].apply(lambda s: int((s == "positive").sum())),
        "negative": g["sentiment"].apply(lambda s: int((s == "negative").sum())),
    }).reset_index()
    out["label"] = out["week"].map(lambda d: f"{d.day} {d:%b}")
    return out


def complaint_matrix(weekly: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Every week x complaint cell (zeros included) for the heatmap, plus each cell's share of its week."""
    if weekly.empty:
        return weekly.assign(label=[], share=[])
    weeks = sorted(weekly["week"].unique())
    grid = pd.MultiIndex.from_product([weeks, order], names=["week", "complaint"])
    full = weekly.set_index(["week", "complaint"])["reviews"].reindex(grid, fill_value=0).reset_index()
    full["label"] = full["week"].map(lambda d: f"{d.day} {d:%b}")
    full["share"] = 100 * full["reviews"] / full.groupby("week")["reviews"].transform("sum").replace(0, pd.NA)
    full["share"] = full["share"].fillna(0).astype(float)
    return full


def branch_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per branch, lowest average rating first (the ones needing attention)."""
    ok = _prep(df)
    rows = []
    for branch, g in ok.groupby("branch"):
        neg = g[g["sentiment"] == "negative"]
        top = top_complaint(g)
        rows.append({
            "Branch": branch,
            "Reviews": len(g),
            "Avg rating": g["rating"].mean(),
            "Negative share": 100 * len(neg) / len(g),
            "Urgent": int((g["urgency"] == "high").sum()),
            "Top complaint": top[0] if top else "-",
        })
    cols = ["Branch", "Reviews", "Avg rating", "Negative share", "Urgent", "Top complaint"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("Avg rating", na_position="last", kind="stable").reset_index(drop=True)


def weekly_complaints(df: pd.DataFrame, top_n: int = 4) -> tuple[pd.DataFrame, list[str]]:
    """Negative reviews per week by complaint type; types outside the top N become 'Other'."""
    neg = _prep(df)
    neg = neg[(neg["sentiment"] == "negative") & neg["date"].notna()]
    if neg.empty:
        return pd.DataFrame(columns=["week", "complaint", "reviews"]), []
    ranked = neg[neg["category"] != "other"]["category"].value_counts().head(top_n).index
    neg = neg.assign(
        week=neg["date"].dt.to_period("W-SUN").dt.start_time,
        complaint=neg["category"].map(lambda c: _label(c) if c in ranked else "Other"),
    )
    order = [_label(c) for c in ranked] + (["Other"] if (neg["complaint"] == "Other").any() else [])
    out = neg.groupby(["week", "complaint"]).size().reset_index(name="reviews")
    return out, order
