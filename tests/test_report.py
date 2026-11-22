import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from reviewlens.export import to_csv_bytes
from reviewlens.pipeline import ENRICHED_COLUMNS
from reviewlens.report import (
    _percentages,
    build_report_input,
    compute_stats,
    generate_weekly_report,
)
from reviewlens.schemas import ThemeNote, WeeklyInsights


def make_df():
    rows = []

    def add(n, sentiment, category, urgency, platform, date="2026-10-01"):
        for i in range(n):
            rows.append({
                "review_id": f"{sentiment}{category}{platform}{len(rows)}", "date": pd.Timestamp(date),
                "platform": platform, "rating": 3, "review_text": "x", "sentiment": sentiment,
                "category": category, "urgency": urgency, "language": "English",
                "summary": f"{category} {sentiment} {i}", "suggested_reply": "ok",
                "needs_review": False, "error": "",
            })

    add(6, "negative", "delivery", "high", "Swiggy", "2026-09-29")
    add(2, "negative", "delivery", "medium", "Zomato")
    add(3, "negative", "price", "medium", "Google")
    add(1, "negative", "other", "low", "Google")
    add(5, "positive", "food_quality", "low", "Zomato")
    add(3, "neutral", "staff", "low", "Google", "2026-10-05")
    rows.append({**rows[0], "review_id": "bad", "needs_review": True, "sentiment": None,
                 "category": None, "urgency": None, "summary": None, "error": "x"})
    return pd.DataFrame(rows, columns=ENRICHED_COLUMNS)


def test_percentages_sum_to_100():
    assert sum(_percentages({"a": 1, "b": 1, "c": 1}).values()) == 100
    assert _percentages({"a": 0, "b": 0}) == {"a": 0, "b": 0}


def test_stats_come_from_the_table():
    stats = compute_stats(make_df())
    assert stats.total == 20 and stats.needs_review == 1
    assert stats.sentiment_counts == {"positive": 5, "neutral": 3, "negative": 12}
    assert sum(stats.sentiment_pct.values()) == 100
    assert [(t.category, t.count) for t in stats.complaints] == [("delivery", 8), ("price", 3)]
    assert stats.complaints[0].platforms == [("Swiggy", 6), ("Zomato", 2)]
    assert [(t.category, t.count) for t in stats.praises] == [("food_quality", 5)]
    assert stats.urgent_count == 6
    assert stats.period == "29 Sep to 5 Oct 2026"


def test_stats_need_at_least_one_analysed_review():
    df = make_df().assign(needs_review=True)
    with pytest.raises(ValueError):
        compute_stats(df)


def test_report_input_is_capped():
    df = make_df()
    text = build_report_input(df, compute_stats(df), per_theme=2, urgent_cap=1)
    assert text.count("delivery negative") == 2 + 1  # 2 under complaints + 1 urgent
    assert "category=delivery | 8 reviews" in text


def test_generate_report_uses_exact_numbers_and_llm_wording():
    insights = WeeklyInsights(
        complaints=[ThemeNote(category="delivery", insight="Late evening deliveries dominate.")],
        praises=[ThemeNote(category="food_quality", insight="Customers love the taste.")],
        action_items=["Add a rider.", "Show charges.", "Reply to urgent reviews.", "extra dropped"],
    )
    seen = {}
    chain = RunnableLambda(lambda inp: seen.update(inp) or insights)
    report = generate_weekly_report(make_df(), chain)

    assert "Delivery: 8 reviews (Swiggy 6, Zomato 2)" in report.text
    assert "Late evening deliveries dominate." in report.text
    assert "URGENT: 6 reviews" in report.text
    assert "3. Reply to urgent reviews." in report.text and "extra dropped" not in report.text
    assert "Price and charges: 3 reviews" in report.text  # no LLM note -> still listed
    assert "report_input" in seen


def test_csv_export_has_bom_and_blocks_formulas():
    df = make_df().head(1).assign(review_text="=HYPERLINK(\"http://x\")", summary="நல்லா இருந்தது")
    data = to_csv_bytes(df)
    assert data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    assert "'=HYPERLINK" in text and "நல்லா இருந்தது" in text
    assert "2026-09-29" in text
