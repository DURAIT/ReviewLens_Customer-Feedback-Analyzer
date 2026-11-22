"""ReviewLens - Streamlit UI.  Run with:  streamlit run app.py"""
import hashlib
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from reviewlens import analytics, chains, config, theme
from reviewlens.export import to_csv_bytes
from reviewlens.pipeline import analyze_reviews, read_csv_bytes, validate_reviews
from reviewlens.report import generate_weekly_report
from reviewlens.schemas import SENTIMENTS, URGENCIES

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "reviews_week40.csv"
URGENCY_RANK = {"high": 0, "medium": 1, "low": 2}

st.set_page_config(page_title="ReviewLens", page_icon=":mag:", layout="wide")
theme.apply_altair_theme()
st.markdown(theme.CSS, unsafe_allow_html=True)
SENTIMENT_COLOR = theme.SENTIMENT_SCALE


def show(chart) -> None:
    """Render with our own chart theme instead of Streamlit's default one."""
    st.altair_chart(chart, theme=None)


@st.cache_data(show_spinner=False)
def load_and_validate(data: bytes, max_rows: int):
    return validate_reviews(read_csv_bytes(data), max_rows=max_rows)


def set_source(name: str, data: bytes) -> None:
    digest = hashlib.sha1(data).hexdigest()
    current = st.session_state.get("source")
    if current and current["digest"] == digest:
        return
    st.session_state.source = {"name": name, "data": data, "digest": digest}
    st.session_state.enriched = None
    st.session_state.report = None


def card(col, label: str, value: str, caption: str = "", delta: str | None = None, **kw) -> None:
    with col.container(border=True):
        st.metric(label, value, delta, **kw)
        if caption:
            st.caption(caption)


# ---------------------------------------------------------------- Overview
def render_overview(enriched: pd.DataFrame) -> None:
    k = analytics.kpis(enriched)
    c = st.columns(5)
    card(c[0], "Reviews", f"{k['total']}", f"{k['this_week']} this week · {k['prev_week']} the week before")
    avg = k["avg_rating"]
    card(
        c[1], "Average rating", "n/a" if avg is None else f"{avg:.2f} / 5",
        delta=None if k["avg_delta"] is None else f"{k['avg_delta']:+.2f} vs last week",
    )
    card(c[2], "Negative share", f"{k['negative_share']:.0f}%", f"{k['negative_count']} negative reviews")
    card(c[3], "Open urgent", f"{k['urgent']}", "Needs action now")
    top = k["top_complaint"]
    card(c[4], "Top complaint", top[0] if top else "None", f"{top[1]} reviews" if top else "")

    left, right = st.columns(2)
    with left:
        st.subheader("What customers complain about most")
        st.caption("Negative reviews by main complaint.")
        data = analytics.complaints_by_category(enriched)
        if data.empty:
            st.info("No negative reviews.")
        else:
            base = alt.Chart(data).encode(
                y=alt.Y("complaint:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220))
            )
            bars = base.mark_bar(color=theme.ARMY, cornerRadiusEnd=3).encode(
                x=alt.X("reviews:Q", title="Negative reviews"), tooltip=["complaint", "reviews"]
            )
            labels = base.mark_text(align="left", dx=4, color=theme.INK).encode(x="reviews:Q", text="reviews:Q")
            show((bars + labels).properties(height=max(120, 34 * len(data))))
    with right:
        st.subheader("Platform distribution")
        st.caption("Review counts per platform, classified by sentiment.")
        dist = analytics.platform_distribution(enriched)
        order = list(dict.fromkeys(dist["platform"]))
        base = alt.Chart(dist).encode(y=alt.Y("platform:N", sort=order, title=None))
        bars = base.mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("reviews:Q", title="Reviews"),
            color=alt.Color("sentiment:N", scale=SENTIMENT_COLOR, title=None, sort=list(SENTIMENTS)),
            tooltip=["platform", "sentiment", "reviews"],
        )
        totals = alt.Chart(dist.drop_duplicates("platform")).mark_text(
            align="left", dx=4, color=theme.INK, fontWeight="bold"
        ).encode(y=alt.Y("platform:N", sort=order), x="total:Q", text="total:Q")
        show((bars + totals).properties(height=max(120, 46 * len(order))))

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment by platform")
        st.caption("Share of reviews on each platform.")
        plat = analytics.sentiment_by_platform(enriched)
        show(
            alt.Chart(plat).mark_bar().encode(
                y=alt.Y("platform:N", title=None),
                x=alt.X("reviews:Q", stack="normalize", axis=alt.Axis(format="%", title=None)),
                color=alt.Color("sentiment:N", scale=SENTIMENT_COLOR, title=None, sort=list(SENTIMENTS)),
                tooltip=["platform", "sentiment", "reviews"],
            ).properties(height=max(120, 46 * plat["platform"].nunique()))
        )
    with right:
        st.subheader("Reviews by category")
        st.caption("All reviews, split by sentiment.")
        cat = enriched[~enriched["needs_review"]].groupby(["category", "sentiment"]).size().reset_index(name="reviews")
        show(
            alt.Chart(cat).mark_bar().encode(
                y=alt.Y("category:N", sort="-x", title=None), x=alt.X("reviews:Q", title="Reviews"),
                color=alt.Color("sentiment:N", scale=SENTIMENT_COLOR, title=None, sort=list(SENTIMENTS)),
                tooltip=["category", "sentiment", "reviews"],
            ).properties(height=max(120, 34 * cat["category"].nunique()))
        )

    scorecard = analytics.branch_scorecard(enriched)
    st.subheader("Branch scorecard")
    st.caption("Sorted by average rating, lowest first: the branches that need attention.")
    if scorecard["Branch"].nunique() == 1 and scorecard.loc[0, "Branch"] == "Unknown":
        st.info("Add a `branch` column to your CSV to see a branch scorecard.")
    else:
        st.dataframe(
            scorecard, hide_index=True,
            column_config={
                "Avg rating": st.column_config.ProgressColumn(
                    "Avg rating", format="%.2f ★", min_value=0, max_value=5),
                "Negative share": st.column_config.ProgressColumn(
                    "Negative share", format="%.1f%%", min_value=0, max_value=100),
                "Urgent": st.column_config.NumberColumn("Urgent", format="%d ⚠"),
            },
        )


# ---------------------------------------------------------------- Trends
def render_trends(enriched: pd.DataFrame) -> None:
    trend = analytics.weekly_trend(enriched)
    if trend.empty:
        st.info("No dated reviews to chart.")
        return
    weeks = trend["label"].tolist()
    if len(weeks) < 2:
        st.caption("Only one week of data so far; upload more history to see a trend.")
    week_x = alt.X("label:O", sort=weeks, title="Week starting", axis=alt.Axis(labelAngle=0))

    st.subheader("Weekly review trend")
    st.caption("Bars: all reviews per week. Lines: positive and negative reviews.")
    bars = alt.Chart(trend).mark_bar(
        color=theme.SAND, stroke=theme.KHAKI, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=60
    ).encode(x=week_x, y=alt.Y("reviews:Q", title="Reviews"),
             tooltip=[alt.Tooltip("label", title="Week"), "reviews"])
    totals = alt.Chart(trend).mark_text(dy=-8, color=theme.INK, fontWeight="bold").encode(
        x=week_x, y="reviews:Q", text="reviews:Q")
    lines_data = trend.melt(id_vars=["label"], value_vars=["positive", "negative"],
                            var_name="sentiment", value_name="count")
    lines = alt.Chart(lines_data).mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=3).encode(
        x=week_x, y=alt.Y("count:Q", title="Reviews"),
        color=alt.Color("sentiment:N", title=None, sort=["positive", "negative"],
                        scale=alt.Scale(domain=["positive", "negative"], range=[theme.POSITIVE, theme.NEGATIVE])),
        tooltip=[alt.Tooltip("label", title="Week"), "sentiment", "count"])
    show((bars + totals + lines).properties(height=340))

    data, order = analytics.weekly_complaints(enriched)
    st.subheader("Weekly complaints")
    st.caption("Top four complaint types per week; the rest are grouped as Other. "
               "Click a legend entry to focus on one type; hover for counts and share.")
    if data.empty:
        st.info("No dated negative reviews to chart.")
        return
    matrix = analytics.complaint_matrix(data, order)
    focus = alt.selection_point(fields=["complaint"], bind="legend")
    tip = [alt.Tooltip("label", title="Week"), "complaint", "reviews",
           alt.Tooltip("share:Q", title="Share of week (%)", format=".0f")]
    stacked = alt.Chart(matrix[matrix["reviews"] > 0]).mark_bar(
        cornerRadius=3, stroke="white", strokeWidth=1.5, size=60
    ).encode(
        x=week_x, y=alt.Y("reviews:Q", title="Negative reviews", stack="zero"),
        color=alt.Color("complaint:N", sort=order, scale=theme.complaint_scale(order), title=None),
        order=alt.Order("complaint:N", sort="ascending"),
        opacity=alt.condition(focus, alt.value(1), alt.value(0.25)),
        tooltip=tip,
    ).add_params(focus)
    week_totals = matrix.groupby("label", as_index=False)["reviews"].sum()
    total_labels = alt.Chart(week_totals).mark_text(dy=-8, color=theme.INK, fontWeight="bold").encode(
        x=week_x, y="reviews:Q", text="reviews:Q")
    show((stacked + total_labels).properties(height=340))

    st.subheader("Complaint intensity")
    st.caption("Reviews per complaint type and week; darker means more.")
    heat = alt.Chart(matrix).encode(
        x=alt.X("label:O", sort=weeks, title="Week starting", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("complaint:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
    )
    rect = heat.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
        color=alt.Color("reviews:Q", title="Reviews",
                        scale=alt.Scale(range=["#EEF0DF", theme.MOSS, theme.OLIVE_DARK])),
        tooltip=tip)
    text = heat.mark_text(fontWeight="bold").encode(
        text="reviews:Q",
        color=alt.condition(alt.datum.reviews > float(matrix["reviews"].max()) / 2,
                            alt.value("white"), alt.value(theme.INK)))
    show((rect + text).properties(height=max(150, 48 * len(order))))


# ---------------------------------------------------------------- Reviews / Urgent
def sorted_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_r=df["urgency"].map(URGENCY_RANK).fillna(3)).sort_values(
        ["_r", "date"], ascending=[True, False], kind="stable"
    ).drop(columns="_r")


def show_table(view: pd.DataFrame) -> None:
    shown = view.copy()
    shown["date"] = pd.to_datetime(shown["date"], errors="coerce")
    shown["rating"] = shown["rating"].map(analytics.stars)
    order = ["date", "platform", "branch", "rating", "sentiment", "category", "urgency",
             "language", "summary", "suggested_reply", "review_text", "review_id", "needs_review"]
    st.dataframe(
        shown, hide_index=True, column_order=order,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "platform": "Platform", "branch": "Branch", "rating": "Rating",
        },
    )


def render_urgent(enriched: pd.DataFrame) -> None:
    urgent = sorted_by_priority(enriched[enriched["urgency"] == "high"])
    st.subheader("Reply to these today")
    st.caption("High-urgency reviews: health, money, abuse or wrong orders. Suggested replies are ready to post.")
    if urgent.empty:
        st.success("No urgent reviews.")
        return
    show_table(urgent)
    st.download_button("Download urgent reviews as CSV", to_csv_bytes(urgent), "reviewlens_urgent.csv", "text/csv")


def render_reviews(enriched: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    picks = {
        "platform": c1.multiselect("Platform", sorted(enriched["platform"].unique())),
        "branch": c2.multiselect("Branch", sorted(enriched["branch"].unique())),
        "sentiment": c3.multiselect("Sentiment", list(SENTIMENTS)),
        "category": c4.multiselect("Category", sorted(enriched["category"].dropna().unique())),
        "urgency": c5.multiselect("Urgency", list(URGENCIES)),
    }
    t1, t2 = st.columns(2)
    high_only = t1.toggle("Show only HIGH urgency", value=False)
    failed_only = t2.toggle("Show only reviews that need manual review", value=False)

    view = enriched
    for col, chosen in picks.items():
        if chosen:
            view = view[view[col].isin(chosen)]
    if high_only:
        view = view[view["urgency"] == "high"]
    if failed_only:
        view = view[view["needs_review"]]
    view = sorted_by_priority(view)

    st.caption(f"{len(view)} of {len(enriched)} reviews")
    show_table(view)
    st.download_button(
        "Download this view as CSV", to_csv_bytes(view), "reviewlens_filtered.csv", "text/csv",
        disabled=view.empty,
    )


# ---------------------------------------------------------------- Weekly report
def render_report(enriched: pd.DataFrame, model: str) -> None:
    if st.button("Generate weekly report", type="primary"):
        try:
            with st.spinner("Writing the weekly report..."):
                chain = chains.build_report_chain(config.build_llm(model))
                st.session_state.report = generate_weekly_report(enriched, chain)
        except Exception as exc:  # show a readable message instead of a stack trace
            st.error(f"Report failed: {type(exc).__name__}: {exc}")
    report = st.session_state.get("report")
    if report:
        st.code(report.text, language=None)
        st.download_button("Download report (.txt)", report.text.encode("utf-8"), "weekly_report.txt")
        st.caption("Counts and percentages are calculated in pandas; the AI only writes the wording.")


def main() -> None:
    for key in ("source", "enriched", "report"):
        st.session_state.setdefault(key, None)

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("OpenAI model", config.get_model_name())
        concurrency = st.slider("Parallel requests", 1, 16, 8)
        max_rows = st.number_input("Max reviews to analyse", 10, 5000, 1000, step=50)
        if config.has_api_key():
            st.success("OPENAI_API_KEY found")
        else:
            st.error("OPENAI_API_KEY missing. Add it to a .env file (see .env.example) and restart.")

    st.markdown(
        '<div class="rl-banner"><h1>ReviewLens</h1>'
        '<p>From customer feedback to smarter business decisions.</p></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a CSV (columns: review_id, date, platform, rating, review_text; optional: branch)",
        type="csv",
    )
    if uploaded is not None:
        set_source(uploaded.name, uploaded.getvalue())
    elif SAMPLE_PATH.exists() and st.button("Or load the sample file (2 weeks, 6 branches)"):
        set_source(SAMPLE_PATH.name, SAMPLE_PATH.read_bytes())

    source = st.session_state.source
    if not source:
        st.info("Upload a CSV or load the sample file to begin.")
        return

    try:
        validation = load_and_validate(source["data"], int(max_rows))
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader(f"1. Validate: {source['name']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows read", validation.rows_read)
    m2.metric("Rows kept", len(validation.df))
    m3.metric("Rows dropped", validation.rows_read - len(validation.df))
    for reason, n in validation.dropped.items():
        st.warning(f"Dropped {n} row(s): {reason}")
    for warning in validation.warnings:
        st.warning(warning)
    if validation.df.empty:
        st.error("No valid reviews left to analyse.")
        return

    st.subheader("2. Analyse")
    if st.button(
        f"Analyse {len(validation.df)} reviews", type="primary", disabled=not config.has_api_key()
    ):
        progress = st.progress(0.0, text="Starting...")
        try:
            chain = chains.build_review_chain(config.build_llm(model))
            st.session_state.enriched = analyze_reviews(
                validation.df, chain, max_concurrency=concurrency,
                on_progress=lambda done, total: progress.progress(
                    done / total, text=f"{done}/{total} reviews analysed"
                ),
            )
            st.session_state.report = None
        except Exception as exc:
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        progress.empty()

    enriched = st.session_state.enriched
    if enriched is None:
        return

    failed = int(enriched["needs_review"].sum())
    if failed:
        st.warning(f"{failed} review(s) could not be analysed and are marked needs_review.")

    st.subheader("3. Results")
    st.download_button(
        "Download enriched CSV", to_csv_bytes(enriched), "reviewlens_enriched.csv", "text/csv",
        type="primary",
    )
    urgent_n = int((enriched["urgency"] == "high").sum())
    tabs = st.tabs([f"Urgent · {urgent_n}", "Overview", "Trends", "Reviews", "Weekly report"])
    with tabs[0]:
        render_urgent(enriched)
    with tabs[1]:
        render_overview(enriched)
    with tabs[2]:
        render_trends(enriched)
    with tabs[3]:
        render_reviews(enriched)
    with tabs[4]:
        render_report(enriched, model)


main()
