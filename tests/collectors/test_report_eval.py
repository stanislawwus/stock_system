"""docs/02_data_sources.md renderer and the collectors eval, on a seeded validation slice."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from evals import eval_collectors
from src import db
from src.collectors import report
from src.marketview.clock import bar_close_for


@pytest.fixture()
def seeded(tmp_path: Path) -> Path:
    path = tmp_path / "s.duckdb"
    with db.writer_session(path) as con:
        d = date(2026, 6, 1)
        px = 100.0
        while d <= date(2026, 6, 30):
            if d.weekday() < 5:
                for t in ("SPY", "AAPL"):
                    con.execute(
                        "INSERT INTO price_bars VALUES (?, ?, ?, ?, ?, ?, 1, 'test', ?)",
                        [t, d, px, px, px, px, bar_close_for(d)],
                    )
                ts = datetime(d.year, d.month, d.day, 15, 0, tzinfo=UTC)
                con.execute(
                    "INSERT INTO news VALUES (?, 'AAPL', ?, 'h', '', ?, 'test', ?)",
                    [f"n{d}", ts, f"http://x/{d}", ts],
                )
            d += timedelta(days=1)
        acc = datetime(2026, 6, 10, 20, 30, tzinfo=UTC)
        con.execute(
            "INSERT INTO filings VALUES ('0000320193-26-000099', 'AAPL', '0000320193', '8-K', "
            "['2.02', '9.01'], ?, DATE '2026-06-10', 'http://x', NULL, ?)",
            [acc, acc],
        )
    return path


def test_planted_errors_are_all_caught_and_clean_slice_is_clean(seeded: Path) -> None:
    con = db.connect(seeded)
    try:
        out = eval_collectors.planted_errors(con, "AAPL")
    finally:
        con.close()
    assert out["clean_slice_flags"] == []
    assert out["missed"] == [], out
    assert len(out["caught"]) == 6


def test_report_renders_counts_and_gaps(seeded: Path) -> None:
    con = db.connect(seeded)
    try:
        text = report.build(con)
    finally:
        con.close()
    assert text.startswith("# 02 — Data sources")
    for section in ("## Totals per source and year", "## Rows per ticker and year", "## Gaps",
                    "## Quality checks", "## 8-K item distribution"):  # fmt: skip
        assert section in text
    assert "| AAPL | technology | " in text
    assert "| 2.02 | 1 |" in text
    assert "not been collected" in text
