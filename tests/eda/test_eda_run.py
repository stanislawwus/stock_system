"""`make eda` pieces on a tiny seeded database: figures render, labels use only dev sessions."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src import db
from src.eda import run
from src.eda.data import open_dev
from src.marketview.clock import bar_close_for

# sessions in the validation period, one trading week + the next
DAYS = [date(2026, 6, 1) + timedelta(days=i) for i in range(21) if (i % 7) < 5]


@pytest.fixture()
def con(tmp_path: Path):  # noqa: ANN201
    path = tmp_path / "e.duckdb"
    with db.writer_session(path) as c:
        for k, d in enumerate(DAYS):
            for t, base in (("SPY", 400.0), ("AAPL", 100.0)):
                o = base + k
                c.execute(
                    "INSERT INTO price_bars VALUES (?, ?, ?, ?, ?, ?, 1, 'test', ?)",
                    [t, d, o, o + 1, o - 1, o + 0.5, bar_close_for(d)],
                )
        for i, d in enumerate(DAYS[:-2]):
            ts = datetime(d.year, d.month, d.day, 15, 0, tzinfo=UTC)
            c.execute(
                "INSERT INTO news VALUES (?, 'AAPL', ?, ?, '', ?, 'test', ?)",
                [f"n{i}", ts, f"Apple headline number {i} about the quarter", f"http://x/{i}", ts],
            )
        acc = datetime(2026, 6, 9, 21, 0, tzinfo=UTC)
        c.execute(
            "INSERT INTO filings VALUES ('f1', 'AAPL', '1', '8-K', ['2.02'], ?, ?, 'u', NULL, ?)",
            [acc, acc.date(), acc],
        )
    return open_dev(path)


def test_compute_labels_and_figures(con, tmp_path: Path) -> None:  # noqa: ANN001
    run._style()
    r = run.compute(con)
    assert r["sessions"] == len(DAYS)
    assert not r["labels"].empty and not r["rets"].empty
    assert len(r["by_item"]["2.02"]) == 1  # filed Tuesday evening -> decision day Wednesday
    for fn in (run.fig_news, run.fig_hours, run.fig_returns, run.fig_gap_drift):
        fn(r, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 4
    text = run.render(run.stats(r), "abc1234", "")
    assert "## 4. gap_ar against drift_ar" in text
    assert not any(w in text.lower() for w in ("caused", "drove", "explains"))  # R11
