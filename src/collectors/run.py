"""Collector CLI, the single DuckDB writer (AGENTS.md R7). Makefile: collect-dev / collect-test.

    uv run python -m src.collectors.run --periods warmup,train,validation
    uv run python -m src.collectors.run --periods test            # only right before test-final
    uv run python -m src.collectors.run --periods train --sources prices --tickers AAPL,MSFT

Order: prices -> news -> filings -> xbrl (facts need filing acceptance times) -> quality checks.
A summary is written to data/raw/collect_summary_<periods>.json for docs/02_data_sources.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.collectors.common import load_periods, load_universe, raw_dir, utc_now, write_json
from src.collectors.edgar import collect_filings
from src.collectors.news import collect_news
from src.collectors.prices import collect_prices
from src.collectors.quality import run_checks
from src.collectors.xbrl import collect_facts
from src.db import writer_session

SOURCES = ("prices", "news", "filings", "xbrl")


def run(
    period_names: list[str],
    sources: list[str],
    tickers: list[str] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    periods = load_periods(period_names)
    companies, benchmark = load_universe()
    if tickers:
        unknown = set(tickers) - {c.ticker for c in companies} - {benchmark}
        if unknown:
            raise SystemExit(f"tickers not in config/universe.yaml: {sorted(unknown)}")
        companies = [c for c in companies if c.ticker in tickers]
    names = [c.ticker for c in companies]
    summary: dict[str, Any] = {
        "periods": [f"{p.name} {p.start}..{p.end}" for p in periods],
        "started_at": utc_now().isoformat(),
        "tickers": len(names),
    }
    with writer_session(db_path) as con:
        if "prices" in sources:
            summary["prices"] = collect_prices(con, [*names, benchmark], periods)
        if "news" in sources:
            summary["news"] = collect_news(con, names, periods)
        if "filings" in sources:
            summary["filings"] = collect_filings(con, companies, periods)
        if "xbrl" in sources:
            summary["xbrl"] = collect_facts(con, companies, periods)
        rep = run_checks(con, periods, names, benchmark)
    summary["quality"] = {"failures": rep.failures, "warnings": rep.warnings, **rep.metrics}
    summary["finished_at"] = utc_now().isoformat()
    write_json(raw_dir() / f"collect_summary_{'_'.join(period_names)}.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--periods", required=True, help="comma list of warmup,train,validation,test")
    ap.add_argument("--sources", default=",".join(SOURCES), help=f"comma list of {SOURCES}")
    ap.add_argument("--tickers", default="", help="comma list (default: whole universe)")
    ap.add_argument("--db", default=None, help="DuckDB path (default DUCKDB_PATH or data/)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sources = [s for s in args.sources.split(",") if s]
    bad = set(sources) - set(SOURCES)
    if bad:
        raise SystemExit(f"unknown sources {sorted(bad)}")
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    summary = run(
        [p.strip() for p in args.periods.split(",")],
        sources,
        tickers or None,
        Path(args.db) if args.db else None,
    )
    q = summary["quality"]
    printable = {k: v for k, v in summary.items() if k != "quality"}
    print(json.dumps(printable, indent=2, default=str))
    print(f"quality: {len(q['failures'])} failure(s), {len(q['warnings'])} warning(s)")
    for f in q["failures"]:
        print("FAIL", f)
    for w in q["warnings"][:30]:
        print("WARN", w)
    return 1 if q["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
