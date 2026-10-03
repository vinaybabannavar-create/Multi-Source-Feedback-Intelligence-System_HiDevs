"""Adapters for app-store reviews and survey CSV exports."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import pandas as pd
import requests


class SourceError(RuntimeError):
    """Raised when an external review source cannot be read."""


REVIEW_COLUMNS = ["review_id", "source", "text", "rating", "date"]


def _as_reviews(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=REVIEW_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    return frame.dropna(subset=["text"]).reset_index(drop=True)


def fetch_google_play_reviews(
    app_id: str,
    count: int = 100,
    country: str = "us",
    language: str = "en",
) -> pd.DataFrame:
    """Fetch recent Google Play reviews using the public store listing."""
    if not app_id.strip():
        raise ValueError("Enter a Google Play package ID, such as com.example.app.")
    try:
        from google_play_scraper import Sort, reviews

        records, _ = reviews(
            app_id,
            lang=language,
            country=country,
            sort=Sort.NEWEST,
            count=max(1, min(int(count), 500)),
        )
    except Exception as exc:
        raise SourceError(f"Google Play could not be reached: {exc}") from exc

    return _as_reviews(
        [
            {
                "review_id": item.get("reviewId"),
                "source": "Google Play",
                "text": item.get("content"),
                "rating": item.get("score"),
                "date": item.get("at"),
            }
            for item in records
        ]
    )


def fetch_app_store_reviews(
    app_id: str,
    count: int = 100,
    country: str = "us",
    timeout: int = 15,
) -> pd.DataFrame:
    """Fetch recent App Store reviews from Apple's customer-review RSS feed."""
    if not app_id.strip():
        raise ValueError("Enter an App Store numeric app ID.")
    url = (
        f"https://itunes.apple.com/{country}/rss/customerreviews/"
        f"id={app_id}/sortBy=mostRecent/json"
    )
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "FeedbackIntelligence/1.0"})
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SourceError(f"App Store reviews could not be fetched: {exc}") from exc

    entries = payload.get("feed", {}).get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]
    records = []
    for entry in entries:
        review_text = entry.get("content", {}).get("label", "")
        if not review_text:
            continue
        records.append(
            {
                "review_id": entry.get("id", {}).get("label"),
                "source": "App Store",
                "text": review_text,
                "rating": entry.get("im:rating", {}).get("label"),
                "date": entry.get("updated", {}).get("label"),
            }
        )
        if len(records) >= max(1, min(int(count), 500)):
            break
    return _as_reviews(records)


def load_survey_csv(file: Any) -> pd.DataFrame:
    """Read an uploaded or local CSV with common survey column names."""
    try:
        frame = pd.read_csv(file)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        raise SourceError(f"The survey CSV could not be read: {exc}") from exc
    if frame.empty:
        return _as_reviews([])

    normalized_names = {str(column).strip().lower(): column for column in frame.columns}

    def find_column(*names: str) -> Any:
        return next((normalized_names[name] for name in names if name in normalized_names), None)

    text_column = find_column("text", "review", "feedback", "comment", "response", "message")
    if text_column is None:
        raise SourceError("CSV needs a text column (for example: feedback, comment, or response).")
    rating_column = find_column("rating", "score", "stars", "satisfaction")
    date_column = find_column("date", "created_at", "timestamp", "submitted_at")
    id_column = find_column("review_id", "id", "response_id")

    records = []
    for index, row in frame.iterrows():
        records.append(
            {
                "review_id": row[id_column] if id_column else f"survey-{index + 1}",
                "source": "Survey CSV",
                "text": row[text_column],
                "rating": row[rating_column] if rating_column else None,
                "date": row[date_column] if date_column else datetime.now(timezone.utc),
            }
        )
    return _as_reviews(records)


def load_survey_csv_bytes(content: bytes) -> pd.DataFrame:
    """Convenience wrapper for raw CSV bytes."""
    try:
        return load_survey_csv(BytesIO(content))
    except UnicodeDecodeError as exc:
        raise SourceError(f"The survey CSV could not be decoded: {exc}") from exc