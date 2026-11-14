"""Demo slice (D-48): cut at the cutoff, loadable, no test-period rows, never overwrites a DB."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import duckdb
import pytest

from evals import chat_fixtures as fx
from src import demo
from src.marketview.clock import bar_close_for

CUT = bar_close_for(fx.DAY)  # 20:00 UTC of the fixture day


@pytest.fixture()
def source_db(tmp_path: Path) -> Path:
    return fx.build(tmp_path / "full.duckdb")


def _count(path: Path, table: str) -> int:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def test_slice_excludes_everything_after_the_cutoff(source_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "demo"
    m = demo.build_slice(source_db, out, cutoff=CUT)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    for name in ("news", "price_bars", "sentiment"):
        newest = con.execute(
            f"SELECT max(available_at) FROM read_parquet('{(out / f'{name}.parquet').as_posix()}')"
        ).fetchone()[0]
        assert newest <= CUT, name
    ids = {
        r[0]
        for r in con.execute(f"SELECT id FROM read_parquet('{out / 'news.parquet'}')").fetchall()
    }
    assert "n_late" not in ids and "n_beat" in ids  # n_late: 21:00 UTC, after the cut
    assert m["tables"]["filings"]["rows"] == 1
    demo.check_slice(out, cutoff=CUT)


def test_check_rejects_a_late_row_and_a_tampered_file(source_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "demo"
    demo.build_slice(source_db, out, cutoff=CUT)
    with pytest.raises(demo.DemoDataError, match="after the cutoff"):
        demo.check_slice(out, cutoff=CUT - timedelta(days=30))
    (out / "news.parquet").write_bytes(b"x")
    with pytest.raises(demo.DemoDataError, match="manifest"):
        demo.check_slice(out, cutoff=CUT)


def test_load_roundtrip_and_refuses_to_overwrite(source_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "demo"
    demo.build_slice(source_db, out, cutoff=CUT)
    target = tmp_path / "app.duckdb"
    counts = demo.load_slice(target, out, copy_artifacts=False)
    assert counts["news"] == _count(source_db, "news") - 1  # minus n_late
    assert counts["filings"] == 1 and counts["price_bars"] < _count(
        source_db, "price_bars"
    )  # bars after the cut dropped
    with pytest.raises(demo.DemoDataError, match="exists"):
        demo.load_slice(target, out, copy_artifacts=False)
    assert demo.load_slice(target, out, force=True, copy_artifacts=False)["news"] == counts["news"]


def test_committed_demo_slice_is_intact_and_has_no_test_period_data() -> None:
    m = demo.check_slice()  # manifest hashes + max(available_at) <= close of the last val day
    assert m["cutoff_utc"].startswith("2026-08-31")
    assert set(m["tables"]) == set(demo.TABLES)
    assert not (demo.DEMO_DIR / "test_scores.parquet").exists()
    assert json.loads((demo.DEMO_DIR / "manifest.json").read_text("utf-8"))["tickers"][-1] == "SPY"
