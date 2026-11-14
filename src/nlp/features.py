"""`make features`: precompute news sentiment for the development splits (D-16, D-36).

Scores every news row of the configured tickers that is available inside the requested splits
(plus `lookback_days_before_split` days before the first one, so the first decision day has
input) and writes one `sentiment` row per (news id, ticker, method) with the news item's
`available_at`. Rows already present for the backend's method are skipped, so the table is the
cache: a re-run only scores what is new. Each article is scored once even when it is tagged with
several tickers. Single writer (R7): holds the DuckDB write lock for the whole run.

Only `train` and `validation` are accepted here; the test period belongs to `make test-final`.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

from src import db
from src.collectors.common import load_periods, load_universe, watchlist_tickers
from src.marketview.clock import cutoff_for
from src.nlp.relevance import config as nlp_config
from src.nlp.sentiment import Backend, get_backend, news_text, score_texts

log = logging.getLogger(__name__)
DEV = ("train", "validation")
CHUNK = 2048  # articles per write: progress survives an interrupted run


def feature_tickers(universe: str | None = None) -> list[str]:
    which = universe or nlp_config()["features"]["universe"]
    if which == "watchlist":
        return watchlist_tickers()
    if which == "universe":
        return [c.ticker for c in load_universe()[0]]
    raise ValueError(f"unknown features.universe {which!r}")


def window(splits: list[str]) -> tuple:
    """(lo, hi] of available_at covered by the splits' decisions."""
    bad = [s for s in splits if s not in DEV]
    if bad:
        raise ValueError(f"splits {bad} not allowed here; use {DEV} (AGENTS.md R3)")
    periods = load_periods(splits)
    pad = timedelta(days=int(nlp_config()["features"]["lookback_days_before_split"]))
    return cutoff_for(periods[0].start) - pad, cutoff_for(periods[-1].end)


def pending(
    con: duckdb.DuckDBPyConnection, tickers: list[str], lo, hi, method: str
) -> pd.DataFrame:
    """News rows in (lo, hi] for `tickers` without a sentiment row of `method`."""
    return con.execute(
        "SELECT n.id, n.ticker, n.headline, n.summary, n.available_at FROM news n "
        "LEFT JOIN sentiment s ON s.source_ref = n.id AND s.ticker = n.ticker AND s.method = ? "
        "WHERE s.source_ref IS NULL AND list_contains(?, n.ticker) "
        "AND n.available_at > ? AND n.available_at <= ? ORDER BY n.available_at, n.id",
        [method, tickers, lo, hi],
    ).df()


def write_sentiment(
    con: duckdb.DuckDBPyConnection, rows: pd.DataFrame, backend: Backend, chunk: int = CHUNK
) -> int:
    """Score `rows` (one inference per distinct id) and insert them; returns rows written."""
    written = 0
    ids = rows["id"].drop_duplicates().tolist()
    for start in range(0, len(ids), chunk):
        part_ids = set(ids[start : start + chunk])
        part = rows[rows["id"].isin(part_ids)]
        first = part.drop_duplicates("id")
        pairs = zip(first["headline"], first["summary"], strict=True)
        texts = [news_text(h, s or "") for h, s in pairs]
        scored = dict(zip(first["id"], score_texts(texts, backend), strict=True))
        out = pd.DataFrame(
            {
                "source_ref": part["id"],
                "ticker": part["ticker"],
                "label": [scored[i].label for i in part["id"]],
                "score": [max(-1.0, min(1.0, scored[i].score)) for i in part["id"]],
                "method": backend.method,
                "available_at": part["available_at"],
            }
        )
        con.register("sent_df", out)
        con.execute("INSERT OR IGNORE INTO sentiment SELECT * FROM sent_df")
        con.unregister("sent_df")
        written += len(out)
        log.info("sentiment: %d / %d articles", min(start + chunk, len(ids)), len(ids))
    return written


def build(
    splits: list[str],
    tickers: list[str] | None = None,
    backend: Backend | None = None,
    db_path: Path | None = None,
) -> dict[str, int | str]:
    lo, hi = window(splits)
    b = backend or get_backend()
    tick = tickers or feature_tickers()
    t0 = time.monotonic()
    with db.writer_session(db_path) as con:
        rows = pending(con, tick, lo, hi, b.method)
        n = write_sentiment(con, rows, b) if len(rows) else 0
        total = con.execute(
            "SELECT count(*) FROM sentiment WHERE method = ? AND list_contains(?, ticker)",
            [b.method, tick],
        ).fetchone()[0]
    return {
        "method": b.method,
        "new_rows": n,
        "rows_for_tickers": int(total),
        "seconds": round(time.monotonic() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="train,validation", help="comma list of train,validation")
    ap.add_argument("--tickers", default=None, help="comma list; default features.universe")
    ap.add_argument("--backend", default=None, choices=["auto", "finbert", "lexicon"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tickers = args.tickers.split(",") if args.tickers else None
    backend = get_backend(args.backend) if args.backend else None
    print(build(args.splits.split(","), tickers, backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
