import unittest
from unittest.mock import patch

import pandas as pd

from feedback_intelligence.analysis import analyze_reviews, get_priority_issues, get_weekly_trends
from feedback_intelligence.reports import build_weekly_report
from feedback_intelligence.sources import SourceError, load_survey_csv


class AnalysisTests(unittest.TestCase):
    def test_sentiment_topic_and_confidence_are_added(self):
        reviews = pd.DataFrame(
            {
                "text": ["This is wonderful and works perfectly", "App crashes and is broken"],
                "source": ["Google Play", "App Store"],
                "date": ["2026-09-01", "2026-09-02"],
            }
        )

        result = analyze_reviews(reviews)

        self.assertEqual(result["sentiment"].tolist(), ["Positive", "Negative"])
        self.assertEqual(result["topic"].tolist(), ["Other", "Crashes & bugs"])
        self.assertTrue(result["confidence"].between(0, 1).all())

    def test_weekly_delta_and_critical_issue_threshold(self):
        reviews = analyze_reviews(
            pd.DataFrame(
                {
                    "text": [
                        "App crashes and this is awful",
                        "App crashes and this is awful",
                        "App crashes and this is awful",
                        "Love the app",
                    ],
                    "date": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-10"],
                }
            )
        )

        issues = get_priority_issues(reviews)
        trends = get_weekly_trends(reviews)

        self.assertTrue(bool(issues.iloc[0]["critical"]))
        self.assertEqual(issues.iloc[0]["negative_count"], 3)
        self.assertEqual(len(trends), 2)
        self.assertGreater(trends.iloc[1]["sentiment_delta"], 0)
        self.assertEqual(trends.iloc[1]["trend"], "Improving")

    def test_survey_csv_requires_a_feedback_column(self):
        with self.assertRaises(SourceError):
            load_survey_csv(pd.io.common.StringIO("department,rating\nsales,4\n"))

    def test_weekly_report_returns_pdf_bytes(self):
        reviews = analyze_reviews(
            pd.DataFrame(
                {
                    "text": ["The app is wonderful", "The app is awful and broken"],
                    "date": ["2026-09-01", "2026-09-03"],
                }
            )
        )

        report = build_weekly_report(reviews)

        self.assertTrue(report.startswith(b"%PDF-"))

    def test_weekly_report_limits_snapshot_to_latest_seven_days(self):
        reviews = analyze_reviews(
            pd.DataFrame(
                {
                    "text": ["The app is awful and broken", "The app is wonderful"],
                    "date": ["2026-09-01", "2026-09-10"],
                }
            )
        )

        with patch("feedback_intelligence.reports.get_priority_issues", wraps=get_priority_issues) as summarize:
            report = build_weekly_report(reviews)

        report_reviews = summarize.call_args.args[0]
        self.assertTrue(report.startswith(b"%PDF-"))
        self.assertEqual(report_reviews["text"].tolist(), ["The app is wonderful"])


if __name__ == "__main__":
    unittest.main()