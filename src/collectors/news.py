"""Finnhub company news. available_at = article `datetime` (D-24).

The free plan serves about one year back and returns at most ~250 items per call (newest
first), so a window whose response reaches `truncation_threshold` is split in half until it is a
single day; a single day still at the threshold is reported as possibly truncated (a gap).
Raw cache: `data/raw/finnhub/<period>/<ticker>/<from>_<to>.json` (D-13).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import pandas as pd

from src.collectors.common import (
    Period,
    PoliteClient,
    collectors_config,
    env,
    log,
    raw_dir,
    read_json,
    utc_now,
    write_json,
)
from src.ratelimit import RateLimiter

NEWS_COLS = ["id", "ticker", "published_at", "headline", "summary", "url", "source",
             "available_at"]  # fmt: skip


def news_id(url: str, headline: str) -> str:
    return hashlib.sha1(f"{url}{headline}".encode()).hexdigest()


def parse_items(items: list[dict[str, Any]], ticker: str, start: date, end: date) -> list[dict]:
    """Provider items -> rows; drops items without headline/url/time or outside [start, end] UTC."""
    rows = []
    for it in items:
        try:
            ts = datetime.fromtimestamp(int(it.get("datetime") or 0), UTC)
        except (TypeError, ValueError):
            continue
        headline = str(it.get("headline") or "").strip()
        url = str(it.get("url") or "").strip()
        if not headline or not url or ts.year < 2000 or not (start <= ts.date() <= end):
            continue
        rows.append(
            {
                "id": news_id(url, headline),
                "ticker": ticker,
                "published_at": ts,
                "headline": headline,
                "summary": str(it.get("summary") or "").strip(),
                "url": url,
                "source": "finnhub",
                "available_at": ts,
            }
        )
    return rows


def windows(start: date, end: date, days: int) -> list[tuple[date, date]]:
    out = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        out.append((cur, stop))
        cur = stop + timedelta(days=1)
    return out


class FinnhubNews:
    def __init__(self, client: PoliteClient | None = None) -> None:
        self.cfg = collectors_config()["finnhub"]
        self.client = client
        self.truncated: list[str] = []

    def _client(self) -> PoliteClient:
        if self.client is None:
            self.client = PoliteClient(
                RateLimiter(int(self.cfg["calls_per_minute"]), 60.0),
                max_retries=int(self.cfg["max_retries"]),
                backoff=float(self.cfg["retry_backoff_seconds"]),
            )
        return self.client

    def _get(self, period: Period, ticker: str, start: date, end: date) -> list[dict]:
        path = raw_dir() / "finnhub" / period.name / ticker / f"{start}_{end}.json"
        cached = read_json(path)
        if cached is not None:
            return cached
        resp = self._client().get(
            f"{self.cfg['base_url']}/company-news",
            params={
                "symbol": ticker,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "token": env("FINNHUB_API_KEY"),
            },  # fmt: skip
        )
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError(
                f"finnhub {ticker} {start}..{end}: unexpected payload {str(data)[:80]}"
            )
        write_json(path, data)
        return data

    def fetch(self, period: Period, ticker: str, start: date, end: date) -> list[dict]:
        """Items for [start, end], splitting windows that may be truncated."""
        items = self._get(period, ticker, start, end)
        if len(items) < int(self.cfg["truncation_threshold"]):
            return items
        if start == end:
            self.truncated.append(f"{ticker}:{start}")
            return items
        mid = start + timedelta(days=(end - start).days // 2)
        return self.fetch(period, ticker, start, mid) + self.fetch(
            period, ticker, mid + timedelta(days=1), end
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def free_window_start(today: date | None = None) -> date:
    days = int(collectors_config()["finnhub"]["free_history_days"])
    return (today or utc_now().date()) - timedelta(days=days)


def insert_news(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)[NEWS_COLS].drop_duplicates(subset=["id", "ticker"])
    before = con.execute("SELECT count(*) FROM news").fetchall()[0][0]
    con.register("news_df", df)
    con.execute(f"INSERT OR IGNORE INTO news SELECT {', '.join(NEWS_COLS)} FROM news_df")
    con.unregister("news_df")
    return con.execute("SELECT count(*) FROM news").fetchall()[0][0] - before


def collect_news(
    con: duckdb.DuckDBPyConnection, tickers: list[str], periods: list[Period]
) -> dict[str, Any]:
    fh = FinnhubNews()
    first = free_window_start()
    stats: dict[str, Any] = {"news": 0, "calls": 0, "skipped": [], "truncated_days": []}
    try:
        for p in periods:
            start = max(p.start, first)
            if start > p.end:
                stats["skipped"].append(f"{p.name}: before the free window ({first})")
                continue
            if start > p.start:
                stats["skipped"].append(f"{p.name}: {p.start}..{start - timedelta(days=1)}")
            for n, t in enumerate(tickers, 1):
                rows = []
                for a, b in windows(start, p.end, int(fh.cfg["initial_window_days"])):
                    rows += parse_items(fh.fetch(p, t, a, b), t, a, b)
                stats["news"] += insert_news(con, rows)
                log.info("news %s %d/%d %s: %d rows", p.name, n, len(tickers), t, len(rows))
    finally:
        stats["calls"] = fh.client.calls if fh.client else 0
        stats["truncated_days"] = fh.truncated
        fh.close()
    return stats
