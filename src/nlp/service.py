"""Analyze service behind POST /api/v1/analyze: arbitrary text -> sentiment (D-36).

Event classification and fact extraction are later components: `events` is None and `facts` is
empty until they exist, rather than a made-up "no event". The text is untrusted (R6): it is
only ever classified, never interpreted.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.nlp.sentiment import analyze_text
from src.schemas import AnalyzeRequest, AnalyzeResponse


def analyze(req: AnalyzeRequest, at: datetime | None = None) -> AnalyzeResponse:
    sent = analyze_text(req.text, req.ticker, at or datetime.now(UTC))
    return AnalyzeResponse(sentiment=sent, events=None, facts=[], method=sent.method)
