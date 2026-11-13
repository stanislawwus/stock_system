"""Daily stock bars: yfinance, Stooq as fallback. available_at = 16:00 ET bar close (D-24).

Prices are split-adjusted, not dividend-adjusted (yfinance `auto_adjust=False` Close): dividend
back-adjustment rewrites past prices with later dividends, a small look-ahead (D-29).
Raw cache: `data/raw/<provider>/<period>/<ticker>.csv`, one file per ticker and period (D-13).
"""

from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.collectors.common import Period, collectors_config, log, raw_dir
from src.marketview.clock import bar_close_for

OHLCV = ["open", "high", "low", "close", "volume"]
STOOQ_URL = "https://stooq.com/q/d/l/"


def _cache_path(provider: str, period: Period, ticker: str) -> Path:
    return raw_dir() / provider / period.name / f"{ticker}.csv"


def normalise(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Index/column 'date' + OHLCV (any case) -> clean frame inside the period, no NaN closes."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        df = df.reset_index()
        df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[[c for c in ["date", *OHLCV] if c in df.columns]]
    if len(df.columns) != 6:
        raise ValueError(f"missing OHLCV columns: {list(df.columns)}")
    df = df.dropna(subset=["open", "close"])
    df = df[df["date"].map(period.contains)]
    df["volume"] = df["volume"].fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


def yfinance_download(tickers: list[str], period: Period) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    data = yf.download(
        tickers,
        start=period.start.isoformat(),
        end=(period.end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out
    for t in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            if t not in data.columns.get_level_values(0):
                continue
            frame = data[t]
        else:
            frame = data
        frame = frame.dropna(how="all")
        if not frame.empty:
            out[t] = frame.rename_axis("date").reset_index()
    return out


def stooq_download(ticker: str, period: Period) -> pd.DataFrame | None:
    import httpx

    params = {
        "s": f"{ticker.lower()}.us",
        "d1": period.start.strftime("%Y%m%d"),
        "d2": period.end.strftime("%Y%m%d"),
        "i": "d",
    }
    try:
        resp = httpx.get(STOOQ_URL, params=params, timeout=30)
    except httpx.HTTPError as exc:
        log.warning("stooq %s: %s", ticker, exc)
        return None
    text = resp.text.strip()
    if resp.status_code != 200 or not text.lower().startswith("date"):
        log.warning("stooq %s: unusable response (HTTP %s, bot check or no data)", ticker,
                    resp.status_code)  # fmt: skip
        return None
    return pd.read_csv(io.StringIO(text))


def fetch_period(tickers: list[str], period: Period) -> dict[str, tuple[str, pd.DataFrame]]:
    """{ticker: (provider, raw frame)} from cache first, then yfinance in batches, then Stooq."""
    cfg = collectors_config()["prices"]
    out: dict[str, tuple[str, pd.DataFrame]] = {}
    for t in tickers:
        for provider in cfg["providers"]:
            path = _cache_path(provider, period, t)
            if path.exists():
                out[t] = (provider, pd.read_csv(path))
                break
    missing = [t for t in tickers if t not in out]
    if "yfinance" in cfg["providers"]:
        size = int(cfg["yfinance_batch"])
        for i in range(0, len(missing), size):
            batch = missing[i : i + size]
            try:
                got = yfinance_download(batch, period)
            except Exception as exc:  # yfinance raises many types when Yahoo blocks
                log.warning("yfinance batch failed (%s); falling back", exc)
                got = {}
            for t, frame in got.items():
                path = _cache_path("yfinance", period, t)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                out[t] = ("yfinance", frame)
    if "stooq" in cfg["providers"]:
        for t in [t for t in tickers if t not in out]:
            frame = stooq_download(t, period)
            if frame is not None and not frame.empty:
                path = _cache_path("stooq", period, t)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                out[t] = ("stooq", frame)
    for t in tickers:
        if t not in out:
            log.warning("prices: no data for %s in %s from any provider", t, period.name)
    return out


def bar_rows(ticker: str, provider: str, raw: pd.DataFrame, period: Period) -> pd.DataFrame:
    df = normalise(raw, period)
    df.insert(0, "ticker", ticker)
    df = df.rename(columns={"date": "bar_date"})
    df["source"] = provider
    df["available_at"] = pd.to_datetime([bar_close_for(d) for d in df["bar_date"]], utc=True)
    return df[["ticker", "bar_date", *OHLCV, "source", "available_at"]]


def insert_bars(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    con.register("bars_df", df)
    con.execute("INSERT OR REPLACE INTO price_bars SELECT * FROM bars_df")
    con.unregister("bars_df")
    return len(df)


def collect_prices(
    con: duckdb.DuckDBPyConnection, tickers: list[str], periods: list[Period]
) -> dict[str, Any]:
    stats: dict[str, Any] = {"bars": 0, "by_provider": {}, "missing": []}
    for p in periods:
        fetched = fetch_period(tickers, p)
        stats["missing"] += [f"{t}:{p.name}" for t in tickers if t not in fetched]
        for t, (provider, raw) in fetched.items():
            n = insert_bars(con, bar_rows(t, provider, raw, p))
            stats["bars"] += n
            stats["by_provider"][provider] = stats["by_provider"].get(provider, 0) + n
        log.info("prices %s: %d tickers", p.name, len(fetched))
    return stats
