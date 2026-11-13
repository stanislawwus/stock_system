"""Trading-day clock: the decision cutoff for day D (AGENTS.md R2, decisions D-03)."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
CUTOFF_LOCAL = time(9, 0)
BAR_CLOSE_LOCAL = time(16, 0)


def cutoff_for(day: date) -> datetime:
    """09:00 America/New_York on `day`, as a UTC datetime. Signals for D use data <= this."""
    return datetime.combine(day, CUTOFF_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)


def bar_close_for(day: date) -> datetime:
    """16:00 America/New_York on `day`, as UTC: available_at of that day's stock bar (D-24)."""
    return datetime.combine(day, BAR_CLOSE_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)
