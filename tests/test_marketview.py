"""Time rules (AGENTS.md R2). Three FAILING tests until MarketView and src.db are implemented:

(a) canary oracle: a planted future record is never returned
(b) random future: for random as_of values no returned record has available_at > as_of
(c) direct data access that bypasses MarketView inside a guarded context raises

They are marked xfail(strict=True, raises=NotImplementedError): any other failure is a real
failure, and once implemented the marker must be removed (strict XPASS fails the suite).
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from src.db import DirectDataAccessError, connect, guarded
from src.marketview import MarketView
from src.marketview.clock import bar_close_for, cutoff_for

SCHEMA = Path(__file__).resolve().parents[1] / "src" / "marketview" / "schema.sql"
AS_OF = cutoff_for(date(2026, 9, 15))  # 09:00 ET on D
CANARY = "CANARY-FUTURE"
design_phase = pytest.mark.xfail(
    strict=True, raises=NotImplementedError, reason="design phase: MarketView not implemented"
)


# ----------------------------------------------------------------------------- clock (passes now)


def test_cutoff_is_0900_new_york_in_utc() -> None:
    assert cutoff_for(date(2026, 11, 13)) == datetime(2026, 11, 13, 14, 0, tzinfo=UTC)  # EST
    assert cutoff_for(date(2026, 7, 1)) == datetime(2026, 7, 1, 13, 0, tzinfo=UTC)  # EDT
    assert bar_close_for(date(2026, 11, 13)) == datetime(2026, 11, 13, 21, 0, tzinfo=UTC)


def test_marketview_requires_aware_as_of() -> None:
    with pytest.raises(ValueError):
        MarketView(datetime(2026, 9, 15, 13, 0))
    assert MarketView.for_trading_day(date(2026, 9, 15)).as_of == AS_OF


# ----------------------------------------------------------------------------- fixture


def _seed(db_path: Path, rng: random.Random) -> None:
    """Rows around AS_OF for AAPL, including planted future rows (CANARY)."""
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA.read_text(encoding="utf-8"))
    day0 = date(2026, 8, 1)
    for i in range(70):
        d = day0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        px = 200 + rng.random() * 10
        con.execute(
            "INSERT INTO price_bars VALUES (?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            ["AAPL", d, px, px + 1, px - 1, px, 1e6, bar_close_for(d)],
        )
        ts = datetime(d.year, d.month, d.day, 12, 0, tzinfo=UTC)
        headline = CANARY if ts > AS_OF else f"news {i}"
        con.execute(
            "INSERT INTO news VALUES (?, 'AAPL', ?, ?, '', 'http://x', 'test', ?)",
            [f"n{i}", ts, headline, ts],
        )
        if i % 10 == 0:
            acc = f"acc-{i}" if ts <= AS_OF else f"{CANARY}-{i}"
            con.execute(
                "INSERT INTO filings VALUES "
                "(?, 'AAPL', '320193', '8-K', ['2.02'], ?, ?, 'http://x', NULL, ?)",
                [acc, ts, d, ts],
            )
            con.execute(
                "INSERT INTO events VALUES (?, 'AAPL', ['2.02'], 0.9, 'index', ?)", [acc, ts]
            )
            con.execute(
                "INSERT INTO sentiment VALUES (?, 'AAPL', 1, 0.6, 'lexicon', ?)", [f"n{i}", ts]
            )
    # XBRL fact filed one day after AS_OF: must be invisible at AS_OF
    con.execute(
        "INSERT INTO facts VALUES ('AAPL', 'revenue', 9.49e10, 'USD', DATE '2026-06-27', "
        "'Q3-2026', 'xbrl', ?, NULL, ?)",
        [CANARY, AS_OF + timedelta(days=1)],
    )
    con.close()


@pytest.fixture()
def seeded_db(tmp_path: Path) -> Path:
    path = tmp_path / "t.duckdb"
    _seed(path, random.Random(0))
    return path


def _all_records(view: MarketView) -> list:
    return [
        *view.prices("AAPL", lookback_days=400),
        *view.news("AAPL", lookback_days=400),
        *view.filings("AAPL", lookback_days=400),
        *view.facts("AAPL"),
        *view.events("AAPL", lookback_days=400),
        *view.sentiment("AAPL", lookback_days=400),
    ]


# ----------------------------------------------------------------------------- the three


@design_phase
def test_a_canary_future_record_is_never_returned(seeded_db: Path) -> None:
    view = MarketView(AS_OF, seeded_db)
    records = _all_records(view)
    assert records, "fixture should yield past records"
    for r in records:
        for value in r.model_dump().values():
            assert CANARY not in str(value), f"future record leaked: {r}"
    # the bar of D itself closes after the cutoff and must be absent; D-1 must be present
    bars = view.prices("AAPL", lookback_days=400)
    assert bars[-1].bar_date < AS_OF.date()
    assert bars[-1].bar_date >= AS_OF.date() - timedelta(days=4)


@design_phase
def test_b_random_as_of_never_returns_the_future(seeded_db: Path) -> None:
    rng = random.Random(1)
    start = datetime(2026, 8, 1, tzinfo=UTC)
    for _ in range(25):
        as_of = start + timedelta(minutes=rng.randint(0, 70 * 24 * 60))
        for r in _all_records(MarketView(as_of, seeded_db)):
            assert r.available_at <= as_of, f"{r} returned at as_of={as_of}"


@design_phase
def test_c_direct_access_inside_guarded_context_raises(seeded_db: Path) -> None:
    with guarded():
        with pytest.raises(DirectDataAccessError):
            connect(seeded_db)
        # the sanctioned path still works inside the guard
        assert MarketView(AS_OF, seeded_db).news("AAPL")
    # outside the guard, read-only access is allowed
    con = connect(seeded_db)
    assert con.execute("SELECT count(*) FROM news").fetchone()[0] > 0
