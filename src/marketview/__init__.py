"""MarketView(as_of): the only read path for historical data (AGENTS.md R2).

Every query is filtered by `available_at <= as_of`. Availability rules (D-24, D-30):
price bar = bar close time · news = provider datetime · filing = EDGAR acceptanceDateTime ·
XBRL fact = acceptance of the 10-Q/10-K that carried it · derived feature = its source's
available_at. `impact_labels` (forward-looking) is never served. Tests: tests/test_marketview.py.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from src.db import _connect_for_marketview
from src.marketview.clock import cutoff_for
from src.schemas import Event, Fact, Filing, NewsItem, PriceBar, Sentiment, Span
from src.splits import assert_readable

__all__ = ["MarketView", "cutoff_for"]

DAYS_PER_QUARTER = 92


class MarketView:
    """Read-only, point-in-time view of the database.

    Strategies, features, impact models and chat tools receive a MarketView and nothing else.
    Opening the database any other way inside `src.db.guarded()` raises DirectDataAccessError.
    """

    def __init__(self, as_of: datetime, db_path: Path | None = None) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware (UTC)")
        assert_readable(as_of)
        self.as_of = as_of
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None

    @classmethod
    def for_trading_day(cls, day: date, db_path: Path | None = None) -> MarketView:
        """View as of the 09:00 ET cutoff of `day`."""
        return cls(cutoff_for(day), db_path)

    # --- plumbing ----------------------------------------------------------------------------
    def _rows(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        if self._con is None:
            self._con = _connect_for_marketview(self._db_path)
            self._con.execute("SET TimeZone = 'UTC'")
        cur = self._con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def _since(self, days: int) -> datetime:
        return self.as_of - timedelta(days=days)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> MarketView:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- history, newest last, all with available_at <= as_of -------------------------------
    def prices(self, ticker: str, lookback_days: int = 250) -> list[PriceBar]:
        """The last `lookback_days` bars (trading days) that had closed by `as_of`."""
        rows = self._rows(
            "SELECT * FROM (SELECT ticker, bar_date, open, high, low, close, volume, source, "
            "available_at FROM price_bars WHERE ticker = ? AND available_at <= ? "
            "ORDER BY bar_date DESC LIMIT ?) ORDER BY bar_date",
            [ticker, self.as_of, lookback_days],
        )
        return [PriceBar(**r) for r in rows]

    def news(self, ticker: str, lookback_days: int = 30) -> list[NewsItem]:
        rows = self._rows(
            "SELECT id, ticker, published_at, headline, summary, url, source, available_at "
            "FROM news WHERE ticker = ? AND available_at <= ? AND available_at > ? "
            "ORDER BY available_at, id",
            [ticker, self.as_of, self._since(lookback_days)],
        )
        return [NewsItem(**r) for r in rows]

    def filings(
        self, ticker: str, lookback_days: int = 90, forms: tuple[str, ...] | None = None
    ) -> list[Filing]:
        sql = (
            "SELECT accession_no, ticker, cik, form, items, accepted_at, filed_date, url, "
            "text_path, available_at FROM filings "
            "WHERE ticker = ? AND available_at <= ? AND available_at > ?"
        )
        params: list[Any] = [ticker, self.as_of, self._since(lookback_days)]
        if forms:
            sql += " AND list_contains(?, form)"
            params.append(list(forms))
        rows = self._rows(sql + " ORDER BY available_at, accession_no", params)
        return [Filing(**r) for r in rows]

    def facts(self, ticker: str, lookback_quarters: int = 8) -> list[Fact]:
        """Facts published by `as_of` whose period ended within the last `lookback_quarters`."""
        since = (self.as_of - timedelta(days=DAYS_PER_QUARTER * lookback_quarters)).date()
        rows = self._rows(
            "SELECT ticker, name, value, unit, period_end, fiscal_period, method, source_ref, "
            "grounding, available_at FROM facts "
            "WHERE ticker = ? AND available_at <= ? AND period_end >= ? "
            "ORDER BY period_end, available_at, name",
            [ticker, self.as_of, since],
        )
        out = []
        for r in rows:
            g = r.pop("grounding")
            out.append(Fact(**r, grounding=Span(**json.loads(g)) if g else None))
        return out

    def sentiment(
        self, ticker: str, lookback_days: int = 30, method: str | None = None
    ) -> list[Sentiment]:
        """Sentiment rows of `ticker`; `method` (finbert | lexicon | ...) filters one backend."""
        sql = (
            "SELECT source_ref, ticker, label, score, method, available_at FROM sentiment "
            "WHERE ticker = ? AND available_at <= ? AND available_at > ?"
        )
        params: list[Any] = [ticker, self.as_of, self._since(lookback_days)]
        if method is not None:
            sql += " AND method = ?"
            params.append(method)
        rows = self._rows(sql + " ORDER BY available_at, source_ref", params)
        return [Sentiment(**r) for r in rows]

    def events(self, ticker: str, lookback_days: int = 30) -> list[Event]:
        rows = self._rows(
            "SELECT source_ref, ticker, labels, confidence, method, available_at FROM events "
            "WHERE ticker = ? AND available_at <= ? AND available_at > ? "
            "ORDER BY available_at, source_ref",
            [ticker, self.as_of, self._since(lookback_days)],
        )
        return [Event(**r) for r in rows]

    def latest_close(self, ticker: str) -> float | None:
        bars = self.prices(ticker, lookback_days=1)
        return bars[-1].close if bars else None
