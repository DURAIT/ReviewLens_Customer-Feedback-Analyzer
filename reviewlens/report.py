"""Weekly report (map-reduce).

Map:    one chain call per review (pipeline.analyze_reviews) -> short English summaries.
Reduce: ONE report chain call that reads pre-computed counts plus a capped sample of
        summaries, so hundreds of reviews fit in a single prompt.

All numbers come from pandas. The LLM only writes the wording around them.
"""
import textwrap
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.runnables import Runnable

from .schemas import CATEGORY_LABELS, SENTIMENTS, WeeklyInsights

_URGENCY_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class ThemeCount:
    category: str
    count: int
    platforms: list[tuple[str, int]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)

    @property
    def platform_text(self) -> str:
        return ", ".join(f"{name} {n}" for name, n in self.platforms)


@dataclass
class WeeklyStats:
    period: str
    total: int
    sentiment_counts: dict[str, int]
    sentiment_pct: dict[str, int]
    complaints: list[ThemeCount]
    praises: list[ThemeCount]
    urgent_count: int
    needs_review: int


@dataclass
class WeeklyReport:
    stats: WeeklyStats
    insights: WeeklyInsights
    text: str


def _percentages(counts: dict[str, int]) -> dict[str, int]:
    """Whole-number percentages that always add up to 100 (largest remainder)."""
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}
    raw = {k: v * 100 / total for k, v in counts.items()}
    floors = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)[: 100 - sum(floors.values())]:
        floors[k] += 1
    return floors


def _period(dates: pd.Series) -> str:
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if d.empty:
        return "date range unknown"
    lo, hi = d.min(), d.max()
    if lo.year == hi.year:
        return f"{lo.day} {lo:%b} to {hi.day} {hi:%b %Y}"
    return f"{lo.day} {lo:%b %Y} to {hi.day} {hi:%b %Y}"


def _themes(ok: pd.DataFrame, sentiment: str, top_n: int) -> list[ThemeCount]:
    sub = ok[(ok["sentiment"] == sentiment) & (ok["category"] != "other")]
    themes = []
    for category, count in sub["category"].value_counts().head(top_n).items():
        platforms = sub[sub["category"] == category]["platform"].value_counts().head(2)
        themes.append(ThemeCount(category, int(count), [(p, int(n)) for p, n in platforms.items()]))
    return themes


def compute_stats(df: pd.DataFrame, top_n: int = 3) -> WeeklyStats:
    ok = df[~df["needs_review"].astype(bool)]
    if ok.empty:
        raise ValueError("No successfully analysed reviews to report on.")
    counts = {s: int((ok["sentiment"] == s).sum()) for s in SENTIMENTS}
    return WeeklyStats(
        period=_period(df["date"]),
        total=len(ok),
        sentiment_counts=counts,
        sentiment_pct=_percentages(counts),
        complaints=_themes(ok, "negative", top_n),
        praises=_themes(ok, "positive", top_n),
        urgent_count=int((ok["urgency"] == "high").sum()),
        needs_review=len(df) - len(ok),
    )


def _top_summaries(sub: pd.DataFrame, n: int) -> list[str]:
    ranked = sub.assign(_r=sub["urgency"].map(_URGENCY_RANK)).sort_values("_r", kind="stable")
    return ranked["summary"].head(n).tolist()


def build_report_input(df: pd.DataFrame, stats: WeeklyStats, per_theme: int = 12, urgent_cap: int = 10) -> str:
    """Compact text the report chain reads: exact counts + a capped sample of summaries."""
    ok = df[~df["needs_review"].astype(bool)]
    lines = [
        f"PERIOD: {stats.period}",
        f"REVIEWS ANALYSED: {stats.total}",
        "SENTIMENT SPLIT: " + ", ".join(f"{s} {stats.sentiment_pct[s]}%" for s in SENTIMENTS),
    ]

    def block(title: str, themes: list[ThemeCount], sentiment: str) -> None:
        lines.append(f"\n{title}")
        if not themes:
            lines.append("(none)")
        for i, t in enumerate(themes, 1):
            lines.append(f"{i}. category={t.category} | {t.count} reviews | {t.platform_text}")
            sub = ok[(ok["sentiment"] == sentiment) & (ok["category"] == t.category)]
            lines.extend(f"   - {s}" for s in _top_summaries(sub, per_theme))

    block("COMPLAINT CATEGORIES (ranked, counts are exact):", stats.complaints, "negative")
    block("PRAISE CATEGORIES (ranked, counts are exact):", stats.praises, "positive")
    lines.append(f"\nURGENT REVIEWS (high urgency): {stats.urgent_count}")
    lines.extend(f"   - {s}" for s in _top_summaries(ok[ok["urgency"] == "high"], urgent_cap))
    return "\n".join(lines)


def render_report(stats: WeeklyStats, insights: WeeklyInsights) -> str:
    notes = {
        "complaints": {n.category: n.insight for n in insights.complaints},
        "praises": {n.category: n.insight for n in insights.praises},
    }

    def wrap(text: str, indent: str) -> str:
        return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)

    p = stats.sentiment_pct
    out = [
        f"WEEKLY FEEDBACK REPORT  |  {stats.period}",
        f"Total reviews: {stats.total}   Positive {p['positive']}%   "
        f"Neutral {p['neutral']}%   Negative {p['negative']}%",
    ]
    if stats.needs_review:
        out.append(f"({stats.needs_review} review(s) could not be analysed and are excluded.)")

    n = len(stats.complaints)
    out.append(f"\nTOP {n} COMPLAINTS" if n > 1 else "\nTOP COMPLAINTS")
    if not stats.complaints:
        out.append("  No complaints this week.")
    for i, t in enumerate(stats.complaints, 1):
        out.append(f"  {i}. {t.label}: {t.count} reviews ({t.platform_text})")
        if t.category in notes["complaints"]:
            out.append(wrap(notes["complaints"][t.category], "     "))

    out.append("\nTOP PRAISE")
    if not stats.praises:
        out.append("  No clear praise themes this week.")
    for t in stats.praises:
        out.append(f"  - {t.label}: {t.count} positive reviews")
        if t.category in notes["praises"]:
            out.append(wrap(notes["praises"][t.category], "    "))

    out.append(f"\nURGENT: {stats.urgent_count} reviews need a reply within 24 hours")
    out.append("\nACTION ITEMS")
    for i, item in enumerate(insights.action_items, 1):
        out.append(textwrap.fill(f"{i}. {item}", width=88, initial_indent="  ", subsequent_indent="     "))
    return "\n".join(out)


def generate_weekly_report(df: pd.DataFrame, report_chain: Runnable) -> WeeklyReport:
    stats = compute_stats(df)
    insights = report_chain.invoke({"report_input": build_report_input(df, stats)})
    return WeeklyReport(stats, insights, render_report(stats, insights))
