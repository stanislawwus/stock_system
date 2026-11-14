"""Trading-day clock: the decision cutoff for day D (AGENTS.md R2, decisions D-03)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
CUTOFF_LOCAL = time(9, 0)
BAR_CLOSE_LOCAL = time(16, 0)
OPEN_LOCAL = time(9, 30)


def cutoff_for(day: date) -> datetime:
    """09:00 America/New_York on `day`, as a UTC datetime. Signals for D use data <= this."""
    return datetime.combine(day, CUTOFF_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)


def bar_close_for(day: date) -> datetime:
    """16:00 America/New_York on `day`, as UTC: available_at of that day's stock bar (D-24)."""
    return datetime.combine(day, BAR_CLOSE_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)


def to_utc(ts: datetime) -> datetime:
    """Normalise any aware datetime (e.g. a DuckDB value in the session's local zone) to UTC."""
    if ts.tzinfo is None:
        raise ValueError("naive datetime: every stored timestamp must be timezone-aware (R2)")
    return ts.astimezone(UTC)


def decision_day(available_at: datetime, sessions: Sequence[date]) -> date | None:
    """First trading day D whose cutoff (09:00 ET) is >= `available_at` (docs/02, D-28).

    `sessions` are the exchange's trading days, ascending. A row available exactly at the cutoff
    is visible to D (available_at <= as_of). None when no listed session qualifies.
    """
    at = to_utc(available_at)
    for day in sessions:
        if cutoff_for(day) >= at:
            return day
    return None


def market_phase(available_at: datetime) -> str:
    """Where an instant falls relative to the cash session in New York, on any calendar day:
    `pre_open` (before 09:30), `session` (09:30-16:00), `after_close` (from 16:00).
    Weekends and holidays are not distinguished here; use `decision_day` for that."""
    local = to_utc(available_at).astimezone(MARKET_TZ).time()
    if local < OPEN_LOCAL:
        return "pre_open"
    return "session" if local < BAR_CLOSE_LOCAL else "after_close"
