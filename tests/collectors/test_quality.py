"""Quality checks catch time-rule violations, test-period rows and gaps (add-collector step 6)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from src import db
from src.collectors import quality
from src.collectors.common import Period
from src.marketview.clock import bar_close_for

VAL = Period("validation", date(2026, 6, 1), date(2026, 6, 12))


def _seed(con: duckdb.DuckDBPyConnection, skip_aapl: date | None = None) -> None:
    d = VAL.start
    px = 100.0
    while d <= VAL.end:
        if d.weekday() < 5:
            for t in ("SPY", "AAPL"):
                if t == "AAPL" and d == skip_aapl:
                    continue
                con.execute(
                    "INSERT INTO price_bars VALUES (?, ?, ?, ?, ?, ?, 1, 'test', ?)",
                    [t, d, px, px, px, px, bar_close_for(d)],
                )
        d += timedelta(days=1)
    ts = datetime(2026, 6, 3, 14, 0, tzinfo=UTC)
    con.execute(
        "INSERT INTO news VALUES ('n1', 'AAPL', ?, 'h', '', 'http://a', 'test', ?)", [ts, ts]
    )


@pytest.fixture()
def con(tmp_path: Path):  # noqa: ANN201
    with db.writer_session(tmp_path / "q.duckdb") as c:
        yield c


def test_clean_data_passes(con: duckdb.DuckDBPyConnection) -> None:
    _seed(con)
    rep = quality.run_checks(con, [VAL], ["AAPL"], "SPY")
    assert rep.ok, rep.failures
    assert rep.metrics["price_completeness"]["AAPL"] == 1.0 and not rep.warnings


def test_missing_session_is_a_warning(con: duckdb.DuckDBPyConnection) -> None:
    _seed(con, skip_aapl=date(2026, 6, 4))
    rep = quality.run_checks(con, [VAL], ["AAPL"], "SPY")
    assert rep.ok and any("2026-06-04" in w for w in rep.warnings)
    assert rep.metrics["price_completeness"]["AAPL"] == pytest.approx(9 / 10)


def test_wrong_available_at_fails(con: duckdb.DuckDBPyConnection) -> None:
    _seed(con)
    con.execute("UPDATE news SET available_at = available_at - INTERVAL 1 HOUR")
    con.execute(
        "UPDATE price_bars SET available_at = available_at - INTERVAL 7 HOUR "
        "WHERE bar_date = DATE '2026-06-05'"
    )
    rep = quality.run_checks(con, [VAL], ["AAPL"], "SPY")
    assert any("news" in f for f in rep.failures)
    assert any("price_bars" in f for f in rep.failures)


def test_test_period_row_in_dev_collection_fails(con: duckdb.DuckDBPyConnection) -> None:
    _seed(con)
    ts = datetime(2026, 9, 15, 14, 0, tzinfo=UTC)
    con.execute(
        "INSERT INTO news VALUES ('leak', 'AAPL', ?, 'h', '', 'http://b', 'test', ?)", [ts, ts]
    )
    rep = quality.run_checks(con, [VAL], ["AAPL"], "SPY")
    assert any("test period" in f for f in rep.failures)


def test_outlier_move_is_flagged(con: duckdb.DuckDBPyConnection) -> None:
    _seed(con)
    con.execute(
        "UPDATE price_bars SET close = 150 WHERE ticker = 'AAPL' AND bar_date = DATE '2026-06-05'"
    )
    rep = quality.run_checks(con, [VAL], ["AAPL"], "SPY")
    assert any("+50.0%" in w for w in rep.warnings)
