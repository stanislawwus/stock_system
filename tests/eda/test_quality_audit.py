"""Data-quality audit on a small seeded database (docs/04_data_quality.md). Offline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from src import db
from src.collectors.common import Company
from src.eda import quality
from src.eda.data import decision_days, news_frame, open_dev, sessions
from src.marketview.clock import bar_close_for, decision_day

UNIVERSE = [Company("AAPL", "technology", "0000320193"), Company("MSFT", "technology", "1")]
DAYS = [date(2026, 6, 1) + timedelta(days=i) for i in range(14) if i % 7 not in (5, 6)]
HOLIDAY = date(2026, 6, 3)  # SPY has no bar: a weekday missing from the calendar


def _news(c: duckdb.DuckDBPyConnection, nid: str, ticker: str, ts: datetime, text: str) -> None:
    c.execute(
        "INSERT INTO news VALUES (?, ?, ?, ?, '', ?, 'test', ?)",
        [nid, ticker, ts, text, f"http://x/{nid}", ts],
    )


def _filing(
    c: duckdb.DuckDBPyConnection, acc: str, ticker: str, cik: str, path: Path, text: str
) -> None:
    path.write_text("[EX-99.1] x\n\n" + text, encoding="utf-8")
    ts = datetime(2026, 6, 2, 21, 0, tzinfo=UTC)
    c.execute(
        "INSERT INTO filings VALUES (?, ?, ?, '8-K', ['2.02'], ?, ?, 'http://f', ?, ?)",
        [acc, ticker, cik, ts, ts.date(), str(path), ts],
    )


@pytest.fixture()
def con(tmp_path: Path) -> Any:
    path = tmp_path / "q.duckdb"
    with db.writer_session(path) as c:
        for d in DAYS:
            for t in ("SPY", "AAPL", "MSFT"):
                if d == HOLIDAY or (t == "MSFT" and d == date(2026, 6, 5)):
                    continue
                c.execute(
                    "INSERT INTO price_bars VALUES (?, ?, 10, 11, 9, 10, 1000, 'test', ?)",
                    [t, d, bar_close_for(d)],
                )
        h = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)  # 08:00 ET: before the open
        text = "Apple unveils a new chip for the data center"
        _news(c, "a", "AAPL", h, text)
        _news(c, "a", "MSFT", h, text)  # one article, two tickers
        # syndicated copy: same words, other id, 2 h later
        _news(
            c, "b", "AAPL", h + timedelta(hours=2), "APPLE unveils a new chip, for the data center!"
        )
        _news(c, "c", "AAPL", h + timedelta(days=3), "Form 8.3")  # too short
        _news(c, "d", "MSFT", h, "Der Konzern und die Bank melden für das Quartal Gewinn")
        # Friday 2026-06-05 20:30 ET -> decision day is Monday 06-08
        _news(c, "e", "AAPL", datetime(2026, 6, 6, 0, 30, tzinfo=UTC), "Apple prices notes")
        _filing(
            c, "0001", "AAPL", "0000320193", tmp_path / "a.txt", "Item 2.02 Results.\n\nRevenue."
        )
        _filing(c, "0002", "MSFT", "9999999", tmp_path / "b.txt", "Plain press release text.")
        # a test-period row must never be read
        late = datetime(2026, 9, 1, 13, 0, tzinfo=UTC)  # exactly the first test cutoff
        _news(c, "z", "AAPL", late, "Apple test period headline that must not be audited")
    return open_dev(path)


def test_frames_never_contain_test_period_rows(con: duckdb.DuckDBPyConnection) -> None:
    assert "z" not in set(news_frame(con)["id"])
    assert all(d < date(2026, 9, 1) for d in sessions(con))


def test_duplicates_and_multi_ticker(con: duckdb.DuckDBPyConnection) -> None:
    d = quality.duplicates(news_frame(con))
    assert d["rows"] == 6 and d["articles"] == 5 and d["pk_duplicates"] == 0
    assert d["multi_ticker_articles"] == 1
    assert d["syndicated_copies"] == 1  # "b" repeats "a" for AAPL within 24 h


def test_missing_days_found_against_the_spy_calendar(con: duckdb.DuckDBPyConnection) -> None:
    m = quality.missing_days(con, news_frame(con), ["AAPL", "MSFT", "SPY"], "SPY")
    assert "2026-06-03" in m["weekdays_without_session"]
    assert set(m["tickers_below_full"]) == {"MSFT"}
    assert m["tickers_below_full"]["MSFT"]["missing"] == ["2026-06-05"]
    assert m["bad_bars"] == 0


def test_utc_phase_and_decision_day(con: duckdb.DuckDBPyConnection) -> None:
    u = quality.utc_and_open(con, news_frame(con))
    assert u["non_timestamptz_columns"] == 0 and u["returned_utc_offsets"] == ["0:00:00"]
    assert u["phase_share"]["pre_open"] > 0 and u["phase_share"]["after_close"] > 0
    assert 3 in u["decision_lag_days"]  # "e": Friday evening -> Monday, 3 days later
    assert u["no_decision_day"] == 0


def test_vectorised_decision_days_match_the_clock(con: duckdb.DuckDBPyConnection) -> None:
    news = news_frame(con)
    sess = sessions(con)
    vec = decision_days(news["published_at"], sess)
    for ts, got in zip(news["published_at"], vec, strict=True):
        want = decision_day(ts.to_pydatetime(), sess)
        assert (None if pd.isna(got) else got) == want


def test_headline_flags(con: duckdb.DuckDBPyConnection) -> None:
    h = quality.headlines(news_frame(con))
    assert h["problem_counts"]["too_short"] == 1 and h["problem_counts"]["non_english"] == 1
    assert h["unusable"] == 2


def test_ticker_mapping_and_cik(con: duckdb.DuckDBPyConnection) -> None:
    t = quality.tickers_mapping(con, news_frame(con), UNIVERSE)
    assert t["news_tickers_outside_universe"] == [] and t["universe_without_news"] == []
    assert t["filings"] == 2 and t["filings_cik_mismatch"] == 1  # MSFT's CIK is wrong


def test_filing_text_headers_are_stripped(con: duckdb.DuckDBPyConnection) -> None:
    f = quality.filing_text(con)
    assert f["texts"] == 2 and f["with_item_heading"] == 1 and f["with_collector_tag"] == 2
    assert f["headings_left_after_strip"] == 0 and f["item_numbers_left_after_strip"] == 0


def test_render_produces_every_section(con: duckdb.DuckDBPyConnection) -> None:
    text = quality.render(quality.audit(con, UNIVERSE, "SPY"), "abc1234")
    for heading in (
        "## 1. Duplicates",
        "## 2. Missing",
        "## 3. UTC",
        "## 4. Headline",
        "## 7. 8-K",
    ):
        assert heading in text
    assert "2026-09-01T13:00:00+00:00" in text  # the cut is stated
