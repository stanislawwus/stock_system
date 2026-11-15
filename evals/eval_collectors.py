"""Collectors eval (AGENTS.md R4) -> evals/results/collectors.json.

metric      price_session_completeness: mean over the universe of the share of exchange sessions
            (warmup + train + validation) that have a daily bar
gold set    derived: the exchange calendar = SPY's sessions (checked against the NYSE holiday list
            in docs/02_data_sources.md); dedup keys; the D-24/D-30 availability rules
target      >= 0.99, with 0 duplicates, 0 availability violations, 0 test-period rows
why         a missing bar silently shifts every rolling window and return a strategy computes
planted     six corruptions of a real data slice (future row, early timestamps, a missing session,
            a duplicate, a price spike); the quality checks must flag every one
Never reads the test period (rows at/after the first test cutoff are not in a dev collection).
The planted slices pass `test_period_locked=False`, so 6/6 holds after the final run too (D-49).
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from src.collectors.common import ROOT, load_periods, load_universe, test_period
from src.collectors.quality import QualityReport, price_completeness, run_checks
from src.db import connect, writer_session

DEV = ("warmup", "train", "validation")
TARGET = 0.99
RESULT = ROOT / "evals" / "results" / "collectors.json"


def _git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=False)  # fmt: skip
    return out.stdout.strip() or "unknown"


def _slice(src: duckdb.DuckDBPyConnection, dst: duckdb.DuckDBPyConnection, ticker: str) -> None:
    """Copy one ticker + SPY for the validation period into a scratch database."""
    p = load_periods(["validation"])[0]
    for table, cond in (
        ("price_bars", "ticker IN (?, 'SPY') AND bar_date BETWEEN ? AND ?"),
        ("news", "ticker = ? AND available_at::DATE BETWEEN ? AND ?"),
        ("filings", "ticker = ? AND available_at::DATE BETWEEN ? AND ?"),
    ):
        df = src.execute(f"SELECT * FROM {table} WHERE {cond}", [ticker, p.start, p.end]).df()
        dst.register("slice_df", df)
        dst.execute(f"INSERT INTO {table} SELECT * FROM slice_df")
        dst.unregister("slice_df")


def _planted(ticker: str) -> list[tuple[str, str, Callable[[duckdb.DuckDBPyConnection], None]]]:
    """(name, text the report must contain, corruption)."""
    test_ts = datetime.combine(test_period().start + timedelta(days=1), datetime.min.time(), UTC)
    return [
        ("news row from the test period", "test period",
         lambda c: c.execute("UPDATE news SET available_at = ?, published_at = ? WHERE id = "
                             "(SELECT min(id) FROM news)", [test_ts, test_ts])),
        ("news available before publication", "news:",
         lambda c: c.execute("UPDATE news SET available_at = available_at - INTERVAL 1 HOUR "
                             "WHERE id = (SELECT max(id) FROM news)")),
        ("bar visible before the close", "price_bars:",
         lambda c: c.execute("UPDATE price_bars SET available_at = available_at - INTERVAL 7 HOUR "
                             "WHERE ticker = ? AND bar_date = (SELECT max(bar_date) FROM "
                             "price_bars)", [ticker])),
        ("missing session", f"prices {ticker}:",
         lambda c: c.execute("DELETE FROM price_bars WHERE ticker = ? AND bar_date = (SELECT "
                             "min(bar_date) FROM price_bars)", [ticker])),
        ("duplicate article", "duplicate",
         lambda c: c.execute("INSERT INTO news SELECT 'dup-' || id, ticker, published_at, "
                             "headline, summary, url, source, available_at FROM news "
                             "WHERE id = (SELECT min(id) FROM news)")),
        ("price spike", "day move",
         lambda c: c.execute("UPDATE price_bars SET close = close * 1.5 WHERE ticker = ? AND "
                             "bar_date = (SELECT max(bar_date) FROM price_bars)", [ticker])),
    ]  # fmt: skip


def planted_errors(src: duckdb.DuckDBPyConnection, ticker: str) -> dict[str, Any]:
    val = load_periods(["validation"])
    out: dict[str, Any] = {"ticker": ticker, "caught": [], "missed": []}
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, expect, corrupt) in enumerate([("clean slice", "", lambda c: None),
                                                     *_planted(ticker)]):  # fmt: skip
            with writer_session(Path(tmp) / f"p{i}.duckdb") as dst:
                _slice(src, dst, ticker)
                corrupt(dst)
                # The scratch slice is validation-only, so a test-period row is always an
                # error here, whether or not the repo holds TEST_FINAL.lock (D-49).
                rep: QualityReport = run_checks(dst, val, [ticker], "SPY", test_period_locked=False)
            text = "\n".join(rep.failures + rep.warnings)
            if not expect:
                out["clean_slice_flags"] = rep.failures + rep.warnings
            elif expect in text:
                out["caught"].append(name)
            else:
                out["missed"].append(name)
    return out


def evaluate(db_path: Path | None = None) -> dict[str, Any]:
    periods = load_periods(DEV)
    companies, benchmark = load_universe()
    tickers = [c.ticker for c in companies]
    con = connect(db_path)
    try:
        comp = price_completeness(con, tickers, benchmark, periods)
        rep = run_checks(con, periods, tickers, benchmark)
        planted = planted_errors(con, "AAPL")
        n_bars = con.execute("SELECT count(*) FROM price_bars").fetchall()[0][0]
        text_cov = con.execute(
            "SELECT avg((text_path IS NOT NULL)::INT) FROM filings WHERE form = '8-K'"
        ).fetchall()[0][0]
    finally:
        con.close()
    value = sum(v["completeness"] for v in comp.values()) / len(comp)
    below = sorted(t for t, v in comp.items() if v["completeness"] < TARGET)
    n_planted = len(planted["caught"]) + len(planted["missed"])
    notes = (
        f"{len(below)} ticker(s) below {TARGET}: {below}. " if below else "All tickers at target. "
    ) + (
        f"Quality failures: {rep.failures}. " if rep.failures else "0 quality failures. "
    ) + f"Planted errors caught {len(planted['caught'])}/{n_planted}."  # fmt: skip
    return {
        "component": "collectors",
        "metric": "price_session_completeness",
        "value": round(value, 4),
        "target": TARGET,
        "why_this_metric": "a missing bar silently shifts every rolling window and return a "
        "strategy computes; duplicates and early timestamps are checked alongside",
        "gold_set": "derived: SPY sessions as the exchange calendar; dedup keys; D-24/D-30 rules",
        "n": len(comp),
        "split": "warmup+train+validation",
        "model": "n/a",
        "git_sha": _git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            "price_bars": n_bars,
            "tickers_below_target": below,
            "quality_failures": rep.failures,
            "quality_warnings": len(rep.warnings),
            "news_duplicate_share": rep.metrics.get("news_duplicate_share"),
            "price_outliers": rep.metrics.get("price_outliers"),
            "eightk_text_coverage": round(float(text_cov or 0.0), 4),
            "planted_errors": planted,
        },
    }


if __name__ == "__main__":
    import json

    result = evaluate()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("component", "metric", "value", "target", "notes")}))
