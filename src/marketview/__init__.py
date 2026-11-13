"""MarketView(as_of): the only read path for historical data (AGENTS.md R2).

Every query is filtered by `available_at <= as_of`. Availability rules (D-24):
price bar = bar close time · news = provider datetime · filing = EDGAR acceptanceDateTime ·
XBRL fact = filed date · derived feature = its source's available_at.

Interface only; implementation lands with the collectors. Tests: tests/test_marketview.py.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from src.marketview.clock import cutoff_for
from src.schemas import Event, Fact, Filing, NewsItem, PriceBar, Sentiment

__all__ = ["MarketView", "cutoff_for"]


class MarketView:
    """Read-only, point-in-time view of the database.

    Strategies, features, impact models and chat tools receive a MarketView and nothing else.
    Opening the database any other way inside `src.db.guarded()` raises DirectDataAccessError.
    """

    def __init__(self, as_of: datetime, db_path: Path | None = None) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware (UTC)")
        self.as_of = as_of
        self._db_path = db_path

    @classmethod
    def for_trading_day(cls, day: date, db_path: Path | None = None) -> MarketView:
        """View as of the 09:00 ET cutoff of `day`."""
        return cls(cutoff_for(day), db_path)

    # --- history, newest last, all with available_at <= as_of -------------------------------
    def prices(self, ticker: str, lookback_days: int = 250) -> list[PriceBar]:
        raise NotImplementedError("design phase")

    def news(self, ticker: str, lookback_days: int = 30) -> list[NewsItem]:
        raise NotImplementedError("design phase")

    def filings(
        self, ticker: str, lookback_days: int = 90, forms: tuple[str, ...] | None = None
    ) -> list[Filing]:
        raise NotImplementedError("design phase")

    def facts(self, ticker: str, lookback_quarters: int = 8) -> list[Fact]:
        raise NotImplementedError("design phase")

    def sentiment(self, ticker: str, lookback_days: int = 30) -> list[Sentiment]:
        raise NotImplementedError("design phase")

    def events(self, ticker: str, lookback_days: int = 30) -> list[Event]:
        raise NotImplementedError("design phase")

    def latest_close(self, ticker: str) -> float | None:
        raise NotImplementedError("design phase")
