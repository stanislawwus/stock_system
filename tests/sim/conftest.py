"""A small price database for strategy and look-ahead tests (business days, validation dates)."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src import db
from src.marketview.clock import bar_close_for

FIRST = date(2025, 5, 1)
LAST = date(2026, 7, 31)


def business_days(start: date = FIRST, end: date = LAST) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def wave(i: int, ticker: str) -> float:
    """Deterministic, trending and mean-reverting enough to give both signs of every score."""
    k = {"A": 1.0, "B": -0.6, "SPY": 0.3}[ticker]
    return 100 * math.exp(k * i / 400 + 0.08 * math.sin(i / 7 + len(ticker)))


def write_bars(path: Path, rows: list[tuple[str, date, float, float]]) -> None:
    """(ticker, day, open, close) rows; one bulk insert (executemany is ~100x slower)."""
    df = pd.DataFrame(
        [(t, d, o, max(o, c), min(o, c), c, 1e6, "test", bar_close_for(d)) for t, d, o, c in rows],
        columns=["ticker", "bar_date", "open", "high", "low", "close", "volume", "source", "av"],
    )
    with db.writer_session(path) as con:
        con.register("bars_df", df)
        con.execute("INSERT OR REPLACE INTO price_bars SELECT * FROM bars_df")


@pytest.fixture()
def price_db(tmp_path: Path) -> Callable[..., Path]:
    path = tmp_path / "px.duckdb"

    def build(tickers: tuple[str, ...] = ("A", "B", "SPY")) -> Path:
        rows = []
        for t in tickers:
            for i, d in enumerate(business_days()):
                c = wave(i, t)
                rows.append((t, d, c * (1 - 0.002), c))
        write_bars(path, rows)
        return path

    return build
