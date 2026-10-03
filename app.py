"""Streamlit dashboard for multi-source feedback intelligence."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from feedback_intelligence.analysis import analyze_reviews, get_priority_issues, get_weekly_trends
from feedback_intelligence.reports import build_weekly_report
from feedback_intelligence.sources import (
    SourceError,
    fetch_app_store_reviews,
    fetch_google_play_reviews,
    load_survey_csv,
)


st.set_page_config(page_title="Feedback Intelligence", page_icon="📊", layout="wide")
st.title("Feedback Intelligence")
st.caption("Customer voice across app stores and survey exports")

if "reviews" not in st.session_state:
    sample_path = Path(__file__).parent / "data" / "sample_reviews.csv"
    st.session_state.reviews = pd.read_csv(sample_path)

with st.sidebar:
    st.header("Sources")
    google_app_id = st.text_input("Google Play package ID", "com.spotify.music")
    apple_app_id = st.text_input("App Store app ID", "310633997")
    review_count = st.slider("Reviews per source", min_value=10, max_value=200, value=50, step=10)

    if st.button("Fetch app store reviews", icon=":material/refresh:", width="stretch"):
        fetched = []
        for fetcher, app_id in (
            (fetch_google_play_reviews, google_app_id),
            (fetch_app_store_reviews, apple_app_id),
        ):
            try:
                result = fetcher(app_id, count=review_count)
                if not result.empty:
                    fetched.append(result)
            except (SourceError, ValueError) as exc:
                st.warning(str(exc))
        if fetched:
            combined = pd.concat([st.session_state.reviews, *fetched], ignore_index=True)
            combined = combined.drop_duplicates(subset=["source", "review_id"], keep="last")
            st.session_state.reviews = combined
            st.success(f"Loaded {sum(len(frame) for frame in fetched)} reviews.")
        elif not st.session_state.reviews.empty:
            st.info("No new reviews were returned.")

    survey_file = st.file_uploader("Survey export", type=["csv"])
    if st.button("Import survey CSV", icon=":material/upload_file:", width="stretch", disabled=survey_file is None):
        try:
            survey_reviews = load_survey_csv(survey_file)
            st.session_state.reviews = pd.concat([st.session_state.reviews, survey_reviews], ignore_index=True)
            st.success(f"Imported {len(survey_reviews)} survey responses.")
        except SourceError as exc:
            st.error(str(exc))

raw_reviews = st.session_state.reviews.copy()
analyzed = analyze_reviews(raw_reviews)
if analyzed.empty:
    st.info("No feedback is available yet. Fetch app-store reviews or import a survey CSV.")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("Filters")
    sources = sorted(analyzed["source"].dropna().astype(str).unique())
    selected_sources = st.multiselect("Source", sources, default=sources)
    sentiment_options = ["Positive", "Neutral", "Negative"]
    selected_sentiments = st.multiselect("Sentiment", sentiment_options, default=sentiment_options)

    valid_dates = analyzed["date"].dropna()
    date_bounds = None
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        date_value = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_value, tuple) and len(date_value) == 2:
            date_bounds = date_value

filtered = analyzed[
    analyzed["source"].astype(str).isin(selected_sources)
    & analyzed["sentiment"].isin(selected_sentiments)
].copy()
if date_bounds:
    filtered = filtered[
        filtered["date"].dt.date.between(date_bounds[0], date_bounds[1])
    ]

if filtered.empty:
    st.info("No reviews match these filters.")
    st.stop()

negative_share = filtered["sentiment"].eq("Negative").mean()
average_score = filtered["sentiment_score"].mean()
priorities = get_priority_issues(filtered)
critical_count = int(priorities["critical"].sum()) if not priorities.empty else 0

metric_columns = st.columns(4)
metric_columns[0].metric("Reviews", f"{len(filtered):,}")
metric_columns[1].metric("Negative feedback", f"{negative_share:.0%}")
metric_columns[2].metric("Avg. sentiment", f"{average_score:+.2f}")
metric_columns[3].metric("Critical topics", critical_count)

trend_column, sentiment_column = st.columns([1.5, 1])
with trend_column:
    st.subheader("Sentiment over time")
    trends = get_weekly_trends(filtered)
    if trends.empty:
        st.caption("Dated reviews are needed to show trends.")
    else:
        trend_figure = px.line(
            trends,
            x="week_start",
            y="average_sentiment",
            markers=True,
            labels={"week_start": "Week", "average_sentiment": "Average sentiment"},
        )
        trend_figure.update_yaxes(range=[-1, 1], zeroline=True, zerolinecolor="#aab8b2")
        trend_figure.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(trend_figure, width="stretch")

with sentiment_column:
    st.subheader("Sentiment mix")
    distribution = filtered["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="reviews")
    sentiment_figure = px.bar(
        distribution,
        x="sentiment",
        y="reviews",
        color="sentiment",
        color_discrete_map={"Positive": "#16806b", "Neutral": "#87928e", "Negative": "#d45b4c"},
    )
    sentiment_figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=300)
    st.plotly_chart(sentiment_figure, width="stretch")

issues_column, report_column = st.columns([1.4, 1])
with issues_column:
    st.subheader("Issue priority")
    if priorities.empty:
        st.caption("No issue patterns found in this selection.")
    else:
        display_issues = priorities.copy()
        display_issues["negative_share"] = display_issues["negative_share"].map(lambda value: f"{value:.0%}")
        display_issues["average_sentiment"] = display_issues["average_sentiment"].map(lambda value: f"{value:+.2f}")
        display_issues["priority"] = display_issues["critical"].map({True: "CRITICAL", False: "Monitor"})
        st.dataframe(
            display_issues[["topic", "negative_count", "total_count", "negative_share", "average_sentiment", "priority"]],
            width="stretch",
            hide_index=True,
            column_config={
                "topic": "Topic",
                "negative_count": "Negative",
                "total_count": "Total",
                "negative_share": "Negative share",
                "average_sentiment": "Avg. sentiment",
                "priority": "Priority",
            },
        )

with report_column:
    st.subheader("Weekly report")
    st.caption("Latest seven days in the selected data, with trend history and prioritized topics.")
    pdf_data = build_weekly_report(filtered)
    st.download_button(
        "Download PDF report",
        data=pdf_data,
        file_name="weekly-feedback-report.pdf",
        mime="application/pdf",
        icon=":material/download:",
        width="stretch",
    )

st.subheader("Recent negative feedback")
negative_reviews = filtered[filtered["sentiment"].eq("Negative")].sort_values("date", ascending=False)
if negative_reviews.empty:
    st.caption("No negative feedback in this selection.")
else:
    st.dataframe(
        negative_reviews[["date", "source", "topic", "rating", "confidence", "text"]].head(20),
        width="stretch",
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.0%%"),
            "text": st.column_config.TextColumn("Feedback", width="large"),
        },
    )

with st.expander("About sentiment confidence"):
    st.write("Sentiment uses VADER's compound polarity score. Confidence is an intensity-based estimate mapped from that score; it is not a calibrated probability.")