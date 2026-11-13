"""Probe test from the time-leak-check skill (step 6) and the single-writer lock (R7)."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src import db
from src.marketview import MarketView

AS_OF = datetime(2026, 7, 1, 13, 0, tzinfo=UTC)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "probe.duckdb"
    with db.writer_session(path) as con:
        for i, ts in enumerate([AS_OF - timedelta(seconds=1), AS_OF, AS_OF + timedelta(seconds=1)]):
            con.execute(
                "INSERT INTO news VALUES (?, 'MSFT', ?, ?, '', 'http://x', 'test', ?)",
                [f"n{i}", ts, f"h{i}", ts],
            )
        con.execute(
            "INSERT INTO impact_labels VALUES ('n0', 'MSFT', ?, DATE '2026-07-01', 0.1, 0.2)",
            [AS_OF],
        )
    return path


def test_row_at_as_of_is_served_one_second_later_is_not(db_path: Path) -> None:
    ids = [n.id for n in MarketView(AS_OF, db_path).news("MSFT")]
    assert ids == ["n0", "n1"]


def test_marketview_has_no_accessor_for_impact_labels() -> None:
    public = {m for m in dir(MarketView) if not m.startswith("_")}
    assert not any("label" in m or "impact" in m or "forward" in m for m in public)


def test_marketview_works_inside_guard_and_connect_does_not(db_path: Path) -> None:
    with db.guarded():
        with pytest.raises(db.DirectDataAccessError):
            db.connect(db_path, writer=True)
        assert MarketView.for_trading_day(date(2026, 7, 2), db_path).news("MSFT")


def test_second_writer_process_is_refused(tmp_path: Path) -> None:
    lock = tmp_path / ".duckdb.write.lock"
    db.acquire_writer_lock(lock)
    try:
        code = (
            "import sys; from pathlib import Path; from src import db\n"
            "try:\n    db.acquire_writer_lock(Path(sys.argv[1]))\n"
            "except db.WriterBusyError:\n    sys.exit(3)\n"
            "sys.exit(0)\n"
        )
        root = Path(__file__).resolve().parents[1]
        rc = subprocess.run([sys.executable, "-c", code, str(lock)], cwd=root, check=False)
        assert rc.returncode == 3
    finally:
        db.release_writer_lock()
    # once released, the lock can be taken again
    db.acquire_writer_lock(lock)
    db.release_writer_lock()


def test_read_only_connect_cannot_write(db_path: Path) -> None:
    con = db.connect(db_path)
    with pytest.raises(Exception, match="read-only|read only"):
        con.execute("DELETE FROM news")
    con.close()
