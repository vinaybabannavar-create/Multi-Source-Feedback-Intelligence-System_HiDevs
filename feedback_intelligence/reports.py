"""Build stakeholder-ready PDF summaries from analyzed review data."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .analysis import get_priority_issues, get_weekly_trends


def build_weekly_report(reviews: pd.DataFrame, title: str = "Weekly Feedback Intelligence") -> bytes:
    """Return a compact PDF report containing weekly KPIs, trends, and issues."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=title,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_LEFT, textColor=colors.HexColor("#183B35")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#183B35"), spaceBefore=12))

    if reviews is None:
        reviews = pd.DataFrame()
    weekly_reviews = reviews
    period = "All reviews (no dates)"
    if "date" in reviews and reviews["date"].notna().any():
        dates = pd.to_datetime(reviews["date"], errors="coerce", utc=True)
        latest = dates.max()
        latest_day_start = latest.normalize()
        week_start = latest_day_start - pd.Timedelta(days=6)
        week_end = latest_day_start + pd.Timedelta(days=1)
        weekly_reviews = reviews.loc[dates.ge(week_start) & dates.lt(week_end)].copy()
        period = f"{week_start.strftime('%b %d, %Y')} to {latest.strftime('%b %d, %Y')}"

    total = len(weekly_reviews)
    negative_share = weekly_reviews["sentiment"].eq("Negative").mean() if total and "sentiment" in weekly_reviews else 0
    average_score = weekly_reviews["sentiment_score"].mean() if total and "sentiment_score" in weekly_reviews else 0
    issues = get_priority_issues(weekly_reviews)
    critical_count = int(issues["critical"].sum()) if not issues.empty else 0
    trends = get_weekly_trends(reviews)
    story = [
        Paragraph(title, styles["ReportTitle"]),
        Paragraph(f"Reporting window: {period} &nbsp; | &nbsp; Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Weekly snapshot", styles["Section"]),
    ]
    kpis = [
        ["Reviews analyzed", "Negative share", "Avg. sentiment", "Critical topics"],
        [str(total), f"{negative_share:.0%}", f"{average_score:+.2f}", str(critical_count)],
    ]
    kpi_table = Table(kpis, colWidths=[1.8 * inch] * 4)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2EF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183B35")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8D5D0")),
    ]))
    story.extend([kpi_table, Paragraph("Sentiment trend", styles["Section"])])

    if not trends.empty:
        figure, axis = plt.subplots(figsize=(7.0, 2.3))
        axis.plot(trends["week_start"], trends["average_sentiment"], marker="o", color="#147D69", linewidth=2)
        axis.axhline(0, color="#9AA8A3", linewidth=0.8)
        axis.set_ylim(-1, 1)
        axis.set_ylabel("Average sentiment")
        axis.grid(axis="y", color="#E3E9E6", linewidth=0.7)
        figure.autofmt_xdate()
        figure.tight_layout()
        chart = BytesIO()
        figure.savefig(chart, format="png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        chart.seek(0)
        story.append(Image(chart, width=6.9 * inch, height=2.25 * inch))
    else:
        story.append(Paragraph("Not enough dated reviews to chart a trend.", styles["BodyText"]))

    story.append(Paragraph("Top issues", styles["Section"]))
    if issues.empty:
        story.append(Paragraph("No issue data is available.", styles["BodyText"]))
    else:
        issue_rows = [["Topic", "Negative", "Total", "Negative share", "Priority"]]
        for row in issues.head(8).itertuples(index=False):
            issue_rows.append([
                str(row.topic),
                str(int(row.negative_count)),
                str(int(row.total_count)),
                f"{row.negative_share:.0%}",
                "CRITICAL" if row.critical else "Monitor",
            ])
        issue_table = Table(issue_rows, colWidths=[2.5 * inch, 0.8 * inch, 0.8 * inch, 1.2 * inch, 1.0 * inch], repeatRows=1)
        issue_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183B35")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F5")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D5D0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(issue_table)

    story.extend([
        Spacer(1, 12),
        Paragraph("Sentiment confidence is an intensity-based estimate derived from VADER, not a calibrated probability.", styles["Italic"]),
    ])
    document.build(story)
    return buffer.getvalue()