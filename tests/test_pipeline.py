import pandas as pd
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from reviewlens.chains import with_validation_retry
from reviewlens.pipeline import (
    ENRICHED_COLUMNS,
    analyze_reviews,
    read_csv_bytes,
    validate_reviews,
)
from reviewlens.schemas import ReviewAnalysis


def raw(rows):
    return pd.DataFrame(rows, columns=["review_id", "date", "platform", "rating", "review_text"])


def good_analysis(**over):
    data = dict(
        sentiment="negative", category="delivery", urgency="high", language="English",
        summary="Order was late.", suggested_reply="Sorry about that.",
    )
    data.update(over)
    return ReviewAnalysis(**data)


def test_schema_rejects_category_outside_allowed_list():
    with pytest.raises(ValidationError):
        good_analysis(category="weather")


def test_schema_rejects_blank_summary():
    with pytest.raises(ValidationError):
        good_analysis(summary="   ")


def test_missing_columns_raise_clear_error():
    with pytest.raises(ValueError, match="review_text"):
        validate_reviews(pd.DataFrame({"review_id": ["1"], "date": ["2026-10-01"]}))


def test_validation_drops_empty_and_duplicate_rows_and_reports_them():
    df = raw([
        ["1", "2026-10-01", "swiggy", "5", "Great"],
        ["2", "2026-10-01", "zomato", "3", "   "],
        ["1", "2026-10-02", "google", "4", "dup id"],
        ["3", "not a date", "amazon", "9", "Rating out of range"],
    ])
    result = validate_reviews(df)
    assert result.rows_read == 4
    assert result.dropped == {"empty review_text": 1, "duplicate review_id": 1}
    assert list(result.df["review_id"]) == ["1", "3"]
    assert result.df["platform"].tolist() == ["Swiggy", "Amazon"]
    assert pd.isna(result.df.loc[1, "rating"]) and pd.isna(result.df.loc[1, "date"])
    assert len(result.warnings) == 2


def test_validation_is_case_insensitive_on_headers_and_caps_rows():
    df = raw([[str(i), "2026-10-01", "Google", "5", f"r{i}"] for i in range(5)])
    df.columns = [c.upper() for c in df.columns]
    result = validate_reviews(df, max_rows=3)
    assert len(result.df) == 3
    assert any("first 3" in w for w in result.warnings)


def test_read_csv_handles_bom_and_tamil():
    text = "review_id,date,platform,rating,review_text\nR1,2026-10-01,Zomato,5,சாப்பாடு நல்லா இருந்தது\n"
    df = read_csv_bytes(text.encode("utf-8-sig"))
    assert df.loc[0, "review_text"] == "சாப்பாடு நல்லா இருந்தது"
    with pytest.raises(ValueError):
        read_csv_bytes(b"")


def test_analyze_marks_failures_instead_of_crashing():
    def fake(inp):
        if "boom" in inp["review_text"]:
            raise RuntimeError("model unavailable")
        return good_analysis(summary=inp["review_text"])

    df = validate_reviews(raw([
        ["1", "2026-10-01", "Swiggy", "1", "late"],
        ["2", "2026-10-01", "Zomato", "5", "boom"],
        ["3", "2026-10-01", "Google", "2", "cold"],
    ])).df
    seen = []
    out = analyze_reviews(df, RunnableLambda(fake), chunk_size=2, on_progress=lambda d, t: seen.append((d, t)))

    assert list(out.columns) == ENRICHED_COLUMNS
    assert out["needs_review"].tolist() == [False, True, False]
    assert out.loc[1, "error"].startswith("RuntimeError")
    assert out.loc[0, "summary"] == "late"  # order preserved
    assert seen == [(2, 3), (3, 3)]


def test_validation_retry_recovers_from_bad_output():
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OutputParserException("category not allowed")
        return "ok"

    assert with_validation_retry(RunnableLambda(flaky)).invoke({}) == "ok"
    assert calls["n"] == 3


def test_validation_retry_gives_up_after_attempts():
    def always_bad(_):
        raise OutputParserException("nope")

    with pytest.raises(OutputParserException):
        with_validation_retry(RunnableLambda(always_bad), attempts=2).invoke({})
