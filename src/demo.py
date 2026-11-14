"""Demo slice: a small, committed copy of the dev data so the app runs without API keys (R15, D-48).

`build_slice` writes one zstd parquet per table to `data/demo/` from the full dev database:
watchlist + benchmark only, every row with `available_at` at or before the 16:00 ET close of the
last validation day (so nothing from the test period can enter; asserted again on every load),
news and its FinBERT scores only for the last `news_days` days. `load_slice` rebuilds a DuckDB
from those files (single writer, R7) and copies the trained models and cached LLM responses to
where the app reads them. CLI: `python -m src.demo build|load`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from src import db
from src.collectors.common import ROOT, load_periods, load_yaml, watchlist_tickers
from src.marketview.clock import bar_close_for

DEMO_DIR = ROOT / "data" / "demo"
MODELS_SRC = ROOT / "data" / "models"
LLM_CACHE_SRC = ROOT / "data" / "cache" / "llm"
TABLES = ("price_bars", "news", "sentiment", "filings", "facts")
NEWS_DAYS = 21  # news (and its sentiment) kept before the cutoff; prices/filings/facts: all


class DemoDataError(RuntimeError):
    pass


def slice_cutoff() -> datetime:
    """16:00 ET close of the last validation day, UTC: the newest row a demo slice may hold."""
    (val,) = load_periods(["validation"])
    return bar_close_for(val.end)


def _tickers() -> list[str]:
    return [*watchlist_tickers(), str(load_yaml("watchlist.yaml")["benchmark"])]


def _queries(cutoff: datetime) -> dict[str, tuple[str, list[Any]]]:
    marks = ",".join("?" * len(_tickers()))
    t = _tickers()
    since = cutoff - timedelta(days=NEWS_DAYS)
    return {
        "price_bars": (
            f"SELECT * FROM price_bars WHERE ticker IN ({marks}) AND available_at <= ? "
            "ORDER BY ticker, bar_date",
            [*t, cutoff],
        ),
        "news": (
            f"SELECT * FROM news WHERE ticker IN ({marks}) AND available_at <= ? "
            "AND available_at > ? ORDER BY ticker, available_at, id",
            [*t, cutoff, since],
        ),
        "sentiment": (
            f"SELECT * FROM sentiment WHERE ticker IN ({marks}) AND available_at <= ? "
            "AND available_at > ? ORDER BY ticker, available_at, source_ref",
            [*t, cutoff, since],
        ),
        "filings": (
            f"SELECT * FROM filings WHERE ticker IN ({marks}) AND available_at <= ? "
            "ORDER BY ticker, available_at, accession_no",
            [*t, cutoff],
        ),
        "facts": (
            f"SELECT * FROM facts WHERE ticker IN ({marks}) AND available_at <= ? "
            "ORDER BY ticker, period_end, name, source_ref",
            [*t, cutoff],
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_slice(
    source_db: Path, out_dir: Path = DEMO_DIR, cutoff: datetime | None = None
) -> dict[str, Any]:
    """Write the parquet files and `manifest.json`; returns the manifest."""
    cutoff = cutoff or slice_cutoff()
    out_dir.mkdir(parents=True, exist_ok=True)
    con = db.connect(source_db)  # read-only, outside any guard (maintainer tooling)
    con.execute("SET TimeZone = 'UTC'")
    tables: dict[str, Any] = {}
    try:
        for name, (sql, params) in _queries(cutoff).items():
            path = out_dir / f"{name}.parquet"
            con.execute(f"CREATE OR REPLACE TEMP TABLE slice AS {sql}", params)
            n = con.execute("SELECT count(*) FROM slice").fetchone()[0]
            con.execute(f"COPY slice TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
            tables[name] = {"rows": n, "sha256": _sha256(path), "bytes": path.stat().st_size}
    finally:
        con.close()
    manifest = {
        "cutoff_utc": cutoff.astimezone(UTC).isoformat(),
        "tickers": _tickers(),
        "news_days": NEWS_DAYS,
        "tables": tables,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _copy_tree(src: Path, dst: Path) -> int:
    n = 0
    if not src.exists():
        return 0
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
        n += 1
    return n


def check_slice(demo_dir: Path = DEMO_DIR, cutoff: datetime | None = None) -> dict[str, Any]:
    """Verify files against the manifest and that no row is newer than the cutoff (R3)."""
    manifest = json.loads((demo_dir / "manifest.json").read_text(encoding="utf-8"))
    cutoff = cutoff or slice_cutoff()
    for name, meta in manifest["tables"].items():
        path = demo_dir / f"{name}.parquet"
        if _sha256(path) != meta["sha256"]:
            raise DemoDataError(f"{path.name} does not match manifest.json")
        newest = pc.max(pq.read_table(path, columns=["available_at"])["available_at"]).as_py()
        if newest is not None and newest > cutoff:
            raise DemoDataError(f"{name}: row available at {newest} is after the cutoff {cutoff}")
    return manifest


def load_slice(
    db_path: Path | None = None,
    demo_dir: Path = DEMO_DIR,
    force: bool = False,
    copy_artifacts: bool = True,
) -> dict[str, int]:
    """Create the DuckDB from the parquet files (refuses to overwrite a database unless forced)."""
    path = Path(db_path) if db_path else db.db_path_default()
    check_slice(demo_dir)
    if path.exists():
        if not force:
            raise DemoDataError(f"{path} exists; use --force to replace it with the demo slice")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with db.writer_session(path) as con:
        for name in TABLES:
            f = (demo_dir / f"{name}.parquet").as_posix()
            con.execute(f"INSERT INTO {name} SELECT * FROM read_parquet('{f}')")
            counts[name] = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    if copy_artifacts:
        counts["models_files"] = _copy_tree(demo_dir / "models", MODELS_SRC)
        counts["llm_cache_files"] = _copy_tree(demo_dir / "llm_cache", LLM_CACHE_SRC)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write data/demo/*.parquet from the full dev database")
    b.add_argument("--source", type=Path, default=None)
    sub.add_parser("check", help="verify data/demo against its manifest and the cutoff")
    ld = sub.add_parser("load", help="create data/stock.duckdb from data/demo")
    ld.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        m = build_slice(a.source or db.db_path_default())
        print(json.dumps({k: v["rows"] for k, v in m["tables"].items()}))
    elif a.cmd == "check":
        check_slice()
        print("demo slice ok")
    else:
        print(json.dumps(load_slice(force=a.force)))


if __name__ == "__main__":
    main()
