# Multi-Source Feedback Intelligence System

A Streamlit dashboard for combining Google Play reviews, Apple App Store reviews, and survey CSV exports. It scores sentiment, tracks weekly movement, surfaces recurring negative topics, and exports a stakeholder-ready PDF report.

## Quick start

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens with sample data so it can be explored without network access or credentials. Use the sidebar to fetch public app-store reviews or import a survey CSV. No API keys or secrets are required.

On Windows PowerShell, activate the virtual environment with `.venv\Scripts\Activate.ps1` instead of the `source` command.

## Sources

- **Google Play:** Uses the public store listing through `google-play-scraper`. Enter the app's package ID (for example, `com.spotify.music`). No API key is needed. Store availability and request limits are controlled by Google.
- **Apple App Store:** Reads Apple's customer-review RSS/JSON feed. Enter the numeric App Store app ID (for example, `310633997`). No API key is needed. The feed availability and review count depend on Apple's endpoint.
- **Survey CSV:** Upload a CSV with a feedback column named `text`, `review`, `feedback`, `comment`, `response`, or `message`. Optional columns include `rating`, `date`, and `id`.

Network failures are reported in the dashboard; already loaded feedback remains available. App-store fetching is on demand, not continuous polling.

## Analysis and reports

VADER classifies each review as positive, neutral, or negative using its compound score. The displayed confidence is an intensity-based heuristic derived from that score, **not** a calibrated probability. Topic categories use transparent keyword rules and can be extended in `feedback_intelligence/analysis.py`. A topic is flagged critical when it has at least three negative reviews in the filtered data.

Weekly trend charts compare average VADER sentiment by calendar week. The dashboard filters by date, source, and sentiment. The PDF summarizes the selected reviews, including sentiment metrics, a trend chart, and prioritized topics.

## Project layout

```text
app.py                            Streamlit dashboard
feedback_intelligence/sources.py  Source adapters and CSV normalization
feedback_intelligence/analysis.py Sentiment, topic, trend, and priority logic
feedback_intelligence/reports.py  PDF report generation
data/sample_reviews.csv           Local demo data
tests/test_analysis.py            Focused unit tests
```

## Tests

```sh
python -m unittest discover -s tests -v
```
