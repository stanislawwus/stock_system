"""UTC normalisation and the conversion of an instant to its decision day (docs/04, D-24)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.marketview.clock import bar_close_for, cutoff_for, decision_day, market_phase, to_utc

NY = ZoneInfo("America/New_York")
KYIV = ZoneInfo("Europe/Kyiv")
# Fri 2026-03-06 (EST), Mon 2026-03-09 (EDT after the 8 March change), Fri 2026-10-30 (EDT),
# Mon 2026-11-02 (EST after the 1 November change)
SESSIONS = [date(2026, 3, 6), date(2026, 3, 9), date(2026, 10, 30), date(2026, 11, 2)]


def test_cutoff_follows_daylight_saving_time() -> None:
    assert cutoff_for(date(2026, 3, 6)) == datetime(2026, 3, 6, 14, 0, tzinfo=UTC)  # EST: UTC-5
    assert cutoff_for(date(2026, 3, 9)) == datetime(2026, 3, 9, 13, 0, tzinfo=UTC)  # EDT: UTC-4
    assert cutoff_for(date(2026, 11, 2)) == datetime(2026, 11, 2, 14, 0, tzinfo=UTC)
    assert bar_close_for(date(2026, 3, 9)) == datetime(2026, 3, 9, 20, 0, tzinfo=UTC)


def test_to_utc_is_independent_of_the_zone_the_database_returns() -> None:
    instant = datetime(2026, 7, 1, 12, 30, tzinfo=UTC)
    for tz in (NY, KYIV, ZoneInfo("Asia/Tokyo")):
        assert to_utc(instant.astimezone(tz)) == instant
        assert to_utc(instant.astimezone(tz)).tzinfo == UTC


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError, match="naive"):
        to_utc(datetime(2026, 7, 1, 12, 30))  # noqa: DTZ001


@pytest.mark.parametrize(
    ("published", "expected_day"),
    [
        # Friday 08:59:59 ET (EST) is visible at Friday's cutoff -> traded at Friday's open
        (datetime(2026, 3, 6, 13, 59, 59, tzinfo=UTC), date(2026, 3, 6)),
        # exactly the cutoff counts as visible (available_at <= as_of)
        (datetime(2026, 3, 6, 14, 0, 0, tzinfo=UTC), date(2026, 3, 6)),
        # one second later: next session, the weekend is skipped
        (datetime(2026, 3, 6, 14, 0, 1, tzinfo=UTC), date(2026, 3, 9)),
        # Saturday news -> Monday
        (datetime(2026, 3, 7, 15, 0, tzinfo=UTC), date(2026, 3, 9)),
        # after the spring change 08:59 ET Monday is 12:59 UTC, still before the cutoff
        (datetime(2026, 3, 9, 12, 59, tzinfo=UTC), date(2026, 3, 9)),
        # 13:30 UTC on that Monday is after the summer-time cutoff (13:00 UTC): next listed day
        (datetime(2026, 3, 9, 13, 30, tzinfo=UTC), date(2026, 10, 30)),
        # the zone the database hands back must not matter
        (datetime(2026, 11, 2, 8, 59, tzinfo=NY), date(2026, 11, 2)),
        (datetime(2026, 11, 2, 9, 0, 1, tzinfo=NY), None),
    ],
)
def test_decision_day(published: datetime, expected_day: date | None) -> None:
    assert decision_day(published, SESSIONS) == expected_day


def test_decision_day_is_none_without_a_later_session() -> None:
    assert decision_day(cutoff_for(SESSIONS[-1]) + timedelta(seconds=1), SESSIONS) is None


@pytest.mark.parametrize(
    ("moment", "phase"),
    [
        (datetime(2026, 6, 2, 9, 29, tzinfo=NY), "pre_open"),
        (datetime(2026, 6, 2, 9, 30, tzinfo=NY), "session"),
        (datetime(2026, 6, 2, 15, 59, tzinfo=NY), "session"),
        (datetime(2026, 6, 2, 16, 0, tzinfo=NY), "after_close"),
        (datetime(2026, 6, 2, 6, 0, tzinfo=NY).astimezone(UTC), "pre_open"),
    ],
)
def test_market_phase(moment: datetime, phase: str) -> None:
    assert market_phase(moment) == phase
