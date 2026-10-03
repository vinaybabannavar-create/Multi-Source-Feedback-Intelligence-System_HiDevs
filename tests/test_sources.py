import unittest
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import requests

from feedback_intelligence.sources import (
    SourceError,
    fetch_app_store_reviews,
    fetch_google_play_reviews,
    load_survey_csv,
)


class SourceTests(unittest.TestCase):
    @patch("google_play_scraper.reviews")
    def test_google_play_reviews_are_normalized(self, fetch_reviews):
        fetch_reviews.return_value = (
            [{"reviewId": "gp-1", "content": "Works great", "score": 5, "at": "2026-09-01"}],
            None,
        )

        reviews = fetch_google_play_reviews("example.app", count=20)

        self.assertEqual(reviews.loc[0, "source"], "Google Play")
        self.assertEqual(reviews.loc[0, "review_id"], "gp-1")
        self.assertEqual(reviews.loc[0, "rating"], 5)
        fetch_reviews.assert_called_once()

    @patch("feedback_intelligence.sources.requests.get")
    def test_app_store_reviews_are_normalized(self, get):
        response = Mock()
        response.json.return_value = {
            "feed": {
                "entry": [
                    {
                        "id": {"label": "as-1"},
                        "content": {"label": "A very good app"},
                        "im:rating": {"label": "5"},
                        "updated": {"label": "2026-09-02T12:00:00-07:00"},
                    }
                ]
            }
        }
        get.return_value = response

        reviews = fetch_app_store_reviews("123456", count=10)

        self.assertEqual(reviews.loc[0, "source"], "App Store")
        self.assertEqual(reviews.loc[0, "review_id"], "as-1")
        self.assertEqual(reviews.loc[0, "rating"], 5)
        get.assert_called_once()

    @patch("feedback_intelligence.sources.requests.get", side_effect=requests.Timeout("timed out"))
    def test_app_store_network_errors_are_reported(self, get):
        with self.assertRaisesRegex(SourceError, "could not be fetched"):
            fetch_app_store_reviews("123456")

    def test_survey_csv_maps_common_columns(self):
        reviews = load_survey_csv(StringIO("Comment,Stars,Submitted At\nGreat app,5,2026-09-03\n"))

        self.assertEqual(reviews.loc[0, "source"], "Survey CSV")
        self.assertEqual(reviews.loc[0, "text"], "Great app")
        self.assertEqual(reviews.loc[0, "rating"], 5)
        self.assertTrue(pd.notna(reviews.loc[0, "date"]))


if __name__ == "__main__":
    unittest.main()