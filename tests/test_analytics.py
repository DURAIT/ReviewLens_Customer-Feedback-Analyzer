import pandas as pd

from reviewlens import analytics
from reviewlens.pipeline import ENRICHED_COLUMNS, validate_reviews


def make_df():
    def row(i, date, branch, rating, sentiment, category, urgency="low"):
        return {
            "review_id": str(i), "date": pd.Timestamp(date), "platform": "Zomato", "branch": branch,
            "rating": rating, "review_text": "x", "sentiment": sentiment, "category": category,
            "urgency": urgency, "language": "English", "summary": "s", "suggested_reply": "r",
            "needs_review": False, "error": "",
        }

    rows = [
        row(1, "2026-09-22", "OMR", 4, "positive", "food_quality"),      # previous week
        row(2, "2026-09-23", "OMR", 2, "negative", "delivery"),          # previous week
        row(3, "2026-09-29", "OMR", 1, "negative", "delivery", "high"),  # this week
        row(4, "2026-09-30", "OMR", 2, "negative", "price"),
        row(5, "2026-09-30", "Adyar", 5, "positive", "staff"),
        row(6, "2026-10-01", "Adyar", 3, "neutral", "other"),
    ]
    rows.append({**rows[0], "review_id": "bad", "needs_review": True, "sentiment": None, "category": None})
    return pd.DataFrame(rows, columns=ENRICHED_COLUMNS)


def test_stars():
    assert analytics.stars(2) == "★★☆☆☆"
    assert analytics.stars(5.0) == "★★★★★"
    assert analytics.stars(float("nan")) == ""


def test_kpis_compare_this_week_with_previous_week():
    k = analytics.kpis(make_df())
    assert (k["total"], k["this_week"], k["prev_week"]) == (6, 4, 2)
    assert round(k["avg_rating"], 2) == 2.83
    assert round(k["avg_delta"], 2) == round((1 + 2 + 5 + 3) / 4 - 3, 2)
    assert (k["negative_count"], k["urgent"]) == (3, 1)
    assert k["top_complaint"] == ("Delivery", 2)


def test_branch_scorecard_sorted_lowest_rating_first():
    sc = analytics.branch_scorecard(make_df())
    assert sc["Branch"].tolist() == ["OMR", "Adyar"]
    omr = sc.iloc[0]
    assert omr["Reviews"] == 4 and omr["Urgent"] == 1 and omr["Top complaint"] == "Delivery"
    assert round(omr["Negative share"]) == 75
    assert sc.iloc[1]["Top complaint"] == "-"


def test_complaints_and_weekly_trend():
    comp = analytics.complaints_by_category(make_df())
    assert comp[["complaint", "reviews"]].values.tolist() == [["Delivery", 2], ["Price and charges", 1]]
    weekly, order = analytics.weekly_complaints(make_df(), top_n=1)
    assert order == ["Delivery", "Other"]
    assert weekly.groupby("complaint")["reviews"].sum().to_dict() == {"Delivery": 2, "Other": 1}
    assert weekly["week"].nunique() == 2


def test_branch_is_optional_in_uploads():
    raw = pd.DataFrame({
        "review_id": ["1"], "date": ["2026-10-01"], "platform": ["google"],
        "rating": ["5"], "review_text": ["Nice"],
    })
    df = validate_reviews(raw).df
    assert df.loc[0, "branch"] == "Unknown"
    with_branch = validate_reviews(raw.assign(Branch=" T Nagar ")).df
    assert with_branch.loc[0, "branch"] == "T Nagar"


def test_platform_distribution_orders_by_volume():
    dist = analytics.platform_distribution(make_df())
    assert dist["platform"].iloc[0] == "Zomato" and dist["total"].iloc[0] == 6


def test_weekly_trend_counts():
    t = analytics.weekly_trend(make_df())
    assert t["reviews"].tolist() == [2, 4]
    assert t["positive"].tolist() == [1, 1] and t["negative"].tolist() == [1, 2]
    assert t["label"].tolist() == ["21 Sep", "28 Sep"]


def test_complaint_matrix_fills_zero_cells_and_shares():
    weekly, order = analytics.weekly_complaints(make_df(), top_n=2)
    m = analytics.complaint_matrix(weekly, order)
    assert len(m) == weekly["week"].nunique() * len(order)
    assert (m.groupby("week")["share"].sum().round() == 100).all()
