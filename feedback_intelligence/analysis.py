"""Normalize feedback and derive sentiment, topic, trend, and priority metrics."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ANALYZED_COLUMNS = [
    "review_id",
    "source",
    "text",
    "rating",
    "date",
    "sentiment",
    "sentiment_score",
    "confidence",
    "topic",
]

_TOPIC_KEYWORDS = {
    "Crashes & bugs": {"crash", "crashes", "bug", "bugs", "error", "broken", "freeze", "freezes"},
    "Login & account": {"login", "log in", "password", "account", "sign in", "signin", "verification"},
    "Payments": {"payment", "pay", "paid", "billing", "subscription", "purchase", "refund"},
    "Performance": {"slow", "lag", "loading", "load", "battery", "performance", "hang"},
    "Ads": {"ad", "ads", "advert", "advertisement", "popup"},
    "Feature requests": {"feature", "please add", "would like", "wish", "missing", "request"},
    "Support": {"support", "help", "customer service", "respond", "response"},
    "Usability": {"confusing", "difficult", "interface", "design", "navigation", "hard to use"},
}


def _column(frame: pd.DataFrame, *candidates: str) -> Any:
    columns = {str(name).strip().lower(): name for name in frame.columns}
    return next((columns[name] for name in candidates if name in columns), None)


def _topic_for(text: str) -> str:
    lowered = text.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "Other"


def analyze_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Standardize records and add VADER sentiment and rule-based issue topics."""
    if reviews is None or reviews.empty:
        return pd.DataFrame(columns=ANALYZED_COLUMNS)

    text_col = _column(reviews, "text", "content", "review", "feedback", "comment", "response")
    if text_col is None:
        raise ValueError("Review data must contain a text, content, review, or feedback column.")
    source_col = _column(reviews, "source")
    id_col = _column(reviews, "review_id", "id")
    rating_col = _column(reviews, "rating", "score", "stars")
    date_col = _column(reviews, "date", "at", "created_at", "timestamp")

    frame = pd.DataFrame(
        {
            "review_id": reviews[id_col] if id_col else [f"review-{i + 1}" for i in range(len(reviews))],
            "source": reviews[source_col] if source_col else "Feedback",
            "text": reviews[text_col].fillna("").astype(str).str.strip(),
            "rating": pd.to_numeric(reviews[rating_col], errors="coerce") if rating_col else float("nan"),
            "date": pd.to_datetime(reviews[date_col], errors="coerce", utc=True) if date_col else pd.NaT,
        }
    )
    frame = frame.loc[frame["text"].ne("")].copy()
    if frame.empty:
        return pd.DataFrame(columns=ANALYZED_COLUMNS)

    analyzer = SentimentIntensityAnalyzer()
    scores = frame["text"].map(lambda text: analyzer.polarity_scores(text)["compound"])
    frame["sentiment_score"] = scores
    frame["sentiment"] = scores.map(lambda score: "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral")
    frame["confidence"] = scores.map(
        lambda score: 0.5 + 0.5 * score
        if score >= 0.05
        else 0.5 + 0.5 * abs(score)
        if score <= -0.05
        else 1.0 - abs(score) * 10
    ).clip(0, 1)
    frame["topic"] = frame["text"].map(_topic_for)
    return frame[ANALYZED_COLUMNS].reset_index(drop=True)


def get_weekly_trends(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly sentiment and compare each week with the previous one."""
    columns = ["week_start", "average_sentiment", "review_count", "sentiment_delta", "trend"]
    if reviews is None or reviews.empty or "sentiment_score" not in reviews or "date" not in reviews:
        return pd.DataFrame(columns=columns)
    frame = reviews.dropna(subset=["date", "sentiment_score"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    frame["week_start"] = dates.dt.to_period("W").dt.start_time
    weekly = (
        frame.groupby("week_start", as_index=False)
        .agg(average_sentiment=("sentiment_score", "mean"), review_count=("sentiment_score", "size"))
        .sort_values("week_start")
    )
    weekly["sentiment_delta"] = weekly["average_sentiment"].diff()
    weekly["trend"] = weekly["sentiment_delta"].map(
        lambda delta: "Improving" if delta > 0.05 else "Declining" if delta < -0.05 else "Stable"
    )
    if not weekly.empty:
        weekly.loc[weekly.index[0], "trend"] = "Baseline"
    return weekly[columns].reset_index(drop=True)


def get_priority_issues(reviews: pd.DataFrame, critical_threshold: int = 3) -> pd.DataFrame:
    """Rank topics by negative-review volume and flag recurring critical issues."""
    columns = ["topic", "negative_count", "total_count", "negative_share", "average_sentiment", "critical"]
    if reviews is None or reviews.empty or "topic" not in reviews or "sentiment" not in reviews:
        return pd.DataFrame(columns=columns)
    summary = (
        reviews.assign(is_negative=reviews["sentiment"].eq("Negative"))
        .groupby("topic", as_index=False)
        .agg(
            negative_count=("is_negative", "sum"),
            total_count=("is_negative", "size"),
            average_sentiment=("sentiment_score", "mean") if "sentiment_score" in reviews else ("is_negative", "mean"),
        )
    )
    summary["negative_share"] = summary["negative_count"] / summary["total_count"]
    summary["critical"] = summary["negative_count"] >= max(1, critical_threshold)
    return (
        summary[columns]
        .sort_values(["critical", "negative_count", "negative_share"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def extract_keywords(text: str) -> list[str]:
    """Return basic content words for lightweight fallback topic exploration."""
    stopwords = {"about", "after", "again", "also", "could", "from", "have", "just", "more", "please", "that", "their", "there", "they", "this", "very", "with", "would", "your"}
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [word for word in words if word not in stopwords]