"""Data-quality checks run at the end of every collect (skill add-collector step 6).

Failures (exit non-zero): naive timestamps, available_at that breaks the D-24/D-30 rules, rows of
the test period in a dev collection (D-13), duplicate news. Warnings: price completeness below
target, > outlier_move_pct day moves, possibly truncated news days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import duckdb

from src.collectors.common import ROOT, Period, collectors_config, test_period
from src.marketview.clock import bar_close_for, cutoff_for

TABLES = ("price_bars", "news", "filings", "facts", "sentiment", "events")
FINAL_LOCK = ROOT / "evals" / "results" / "TEST_FINAL.lock"


@dataclass
class QualityReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def _one(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> Any:
    return con.execute(sql, params or []).fetchall()[0][0]


def calendar(con: duckdb.DuckDBPyConnection, benchmark: str, period: Period) -> list[date]:
    """Exchange sessions = the benchmark's bar dates (SPY trades every NYSE session)."""
    rows = con.execute(
        "SELECT bar_date FROM price_bars WHERE ticker = ? AND bar_date BETWEEN ? AND ? "
        "ORDER BY bar_date",
        [benchmark, period.start, period.end],
    ).fetchall()
    return [r[0] for r in rows]


def price_completeness(
    con: duckdb.DuckDBPyConnection, tickers: list[str], benchmark: str, periods: list[Period]
) -> dict[str, dict[str, Any]]:
    """{ticker: {expected, present, completeness, missing: [dates]}} over the given periods."""
    sessions = [d for p in periods for d in calendar(con, benchmark, p)]
    out = {}
    for t in tickers:
        have = {
            r[0]
            for r in con.execute("SELECT bar_date FROM price_bars WHERE ticker = ?", [t]).fetchall()
        }
        missing = [d for d in sessions if d not in have]
        out[t] = {
            "expected": len(sessions),
            "present": len(sessions) - len(missing),
            "completeness": (1 - len(missing) / len(sessions)) if sessions else 0.0,
            "missing": [d.isoformat() for d in missing],
        }
    return out


def run_checks(
    con: duckdb.DuckDBPyConnection,
    periods: list[Period],
    tickers: list[str],
    benchmark: str,
    *,
    test_period_locked: bool | None = None,
) -> QualityReport:
    """Run every check; `test_period_locked` says whether the test period has been used.

    `None` (the collector CLI) reads `TEST_FINAL.lock`: after the final run the dev database
    legitimately holds test-period rows, so check 3 is skipped. Callers that build their own
    slice (the collectors eval, tests) pass the state explicitly, so their result does not
    change because the repo contains the lock (D-49).
    """
    cfg = collectors_config()["prices"]
    rep = QualityReport()
    if test_period_locked is None:
        test_period_locked = FINAL_LOCK.exists()

    # 1. every available_at is TIMESTAMPTZ NOT NULL (R2)
    for table in TABLES:
        dtype = _one(
            con,
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = 'available_at'",
            [table],
        )
        if dtype != "TIMESTAMP WITH TIME ZONE":
            rep.failures.append(f"{table}.available_at has type {dtype}")
        nulls = _one(con, f"SELECT count(*) FROM {table} WHERE available_at IS NULL")
        if nulls:
            rep.failures.append(f"{table}: {nulls} rows with NULL available_at")

    # 2. availability rules (D-24, D-30)
    bad_bars = [
        (t, d)
        for t, d, a in con.execute(
            "SELECT ticker, bar_date, available_at FROM price_bars"
        ).fetchall()
        if a != bar_close_for(d)
    ]
    if bad_bars:
        rep.failures.append(f"price_bars: {len(bad_bars)} rows with available_at != bar close")
    for table, col in (("news", "published_at"), ("filings", "accepted_at")):
        n = _one(con, f"SELECT count(*) FROM {table} WHERE available_at <> {col}")
        if n:
            rep.failures.append(f"{table}: {n} rows with available_at <> {col}")
    n = _one(con, "SELECT count(*) FROM facts WHERE available_at::DATE < period_end")
    if n:
        rep.failures.append(f"facts: {n} rows available before their period ended")
    n = _one(
        con,
        "SELECT count(*) FROM facts f JOIN filings g ON f.source_ref = g.accession_no "
        "WHERE f.method = 'xbrl' AND f.available_at <> g.accepted_at",
    )
    if n:
        rep.failures.append(f"facts: {n} xbrl rows not stamped with their filing's acceptance")

    # 3. a dev collection holds no test-period rows (D-13)
    if "test" not in {p.name for p in periods} and not test_period_locked:
        cut = cutoff_for(test_period().start)
        for table in TABLES:
            n = _one(con, f"SELECT count(*) FROM {table} WHERE available_at >= ?", [cut])
            if n:
                rep.failures.append(f"{table}: {n} rows from the test period in a dev collection")

    # 4. duplicates
    dup = _one(con, "SELECT count(*) - count(DISTINCT (ticker, url)) FROM news")
    total_news = _one(con, "SELECT count(*) FROM news")
    rep.metrics["news_duplicate_share"] = (dup / total_news) if total_news else 0.0
    if dup:
        rep.failures.append(f"news: {dup} duplicate (ticker, url) rows")

    # 5. completeness against the exchange calendar
    comp = price_completeness(con, tickers, benchmark, periods)
    rep.metrics["price_completeness"] = {t: round(v["completeness"], 4) for t, v in comp.items()}
    low = {t: v for t, v in comp.items() if v["completeness"] < float(cfg["min_completeness"])}
    for t, v in low.items():
        rep.warnings.append(
            f"prices {t}: {v['present']}/{v['expected']} sessions, missing {v['missing'][:5]}"
        )

    # 6. outliers
    rows = con.execute(
        "SELECT ticker, bar_date, "
        "close / lag(close) OVER (PARTITION BY ticker ORDER BY bar_date) - 1 AS r "
        "FROM price_bars QUALIFY abs(r) > ?",
        [float(cfg["outlier_move_pct"])],
    ).fetchall()
    rep.metrics["price_outliers"] = [f"{t} {d} {r:+.1%}" for t, d, r in rows]
    for t, d, r in rows:
        rep.warnings.append(f"prices {t} {d}: {r:+.1%} day move (check split/data error)")
    return rep
