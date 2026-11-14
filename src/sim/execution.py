"""Execution prices: the open and close of day D, for the simulator only (AGENTS.md R2).

This is the one place that serves open(D). It is loaded once, before the daily loop, for the
split's dates only, and is never handed to a strategy (strategies get a MarketView). The session
calendar is the benchmark's trading days (D-29: SPY sessions = exchange calendar).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

from src.db import connect
from src.marketview.clock import bar_close_for
from src.splits import assert_readable


class ExecutionPrices:
    def __init__(
        self, bars: Mapping[tuple[str, date], tuple[float, float]], sessions: Iterable[date]
    ) -> None:
        """`bars[(ticker, day)] = (open, close)`; `sessions` ascending trading days."""
        self._bars = dict(bars)
        self.sessions: list[date] = sorted(set(sessions))

    @classmethod
    def load(
        cls,
        tickers: Iterable[str],
        start: date,
        end: date,
        calendar_ticker: str,
        db_path: Path | None = None,
    ) -> ExecutionPrices:
        """Bars with start <= bar_date <= end. Refuses an `end` in the locked test period (R3)."""
        assert_readable(bar_close_for(end))
        names = sorted({*tickers, calendar_ticker})
        con = connect(db_path)
        try:
            rows = con.execute(
                "SELECT ticker, bar_date, open, close FROM price_bars "
                "WHERE list_contains(?, ticker) AND bar_date BETWEEN ? AND ? "
                "AND open > 0 AND close > 0",
                [names, start, end],
            ).fetchall()
        finally:
            con.close()
        bars = {(t, d): (float(o), float(c)) for t, d, o, c in rows}
        sessions = [d for t, d, _, _ in rows if t == calendar_ticker]
        if not sessions:
            raise ValueError(f"no {calendar_ticker} sessions between {start} and {end}")
        return cls(bars, sessions)

    def open_price(self, ticker: str, day: date) -> float | None:
        bar = self._bars.get((ticker, day))
        return bar[0] if bar else None

    def close_price(self, ticker: str, day: date) -> float | None:
        bar = self._bars.get((ticker, day))
        return bar[1] if bar else None
