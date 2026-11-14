"""Read-only services behind GET /watchlist, /brief, /events, /news and /prices (R13).

Everything history-shaped goes through `MarketView(as_of)` inside `guarded()` (R2, D-25). `as_of` is
the 09:00 ET cutoff of a trading day D; the default is the last validation day because the test
period is not servable outside `make test-final` (R3). The watchlist is read from config.
Wording follows R11: items *coincided with* moves; nothing here says a story moved a price.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import yaml

from src.collectors.common import CONFIG_DIR, load_yaml
from src.db import guarded
from src.marketview import MarketView
from src.nlp.daily import window_scores
from src.schemas import (
    Brief,
    BriefItem,
    EventsResponse,
    NewsEntry,
    NewsResponse,
    PriceBar,
    PricesResponse,
    Watchlist,
    WatchlistItem,
)
from src.strategies.registry import build
from src.strategies.service import (
    DataUnavailableError,
    TestPeriodLockedError,
    UnknownTickerError,
    known_tickers,
    resolve_as_of,
)

log = logging.getLogger(__name__)
__all__ = [
    "DataUnavailableError",
    "TestPeriodLockedError",
    "UnknownTickerError",
    "api_config",
    "brief_for",
    "events_for",
    "news_for",
    "prices_for",
    "watchlist",
]


@lru_cache(maxsize=1)
def api_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "api.yaml").read_text(encoding="utf-8"))


def watchlist() -> Watchlist:
    cfg = load_yaml("watchlist.yaml")
    items = [WatchlistItem(ticker=s["ticker"], sector=s.get("sector")) for s in cfg["stocks"]]
    items += [
        WatchlistItem(ticker=c["display"], kind="crypto", sector="crypto")
        for c in cfg.get("crypto", [])
    ]
    return Watchlist(items=items, benchmark=cfg["benchmark"])


@contextmanager
def _session(as_of: date | datetime | None, db_path: Path | None) -> Iterator[MarketView]:
    """A guarded MarketView at the cutoff; database problems become DataUnavailableError."""
    try:
        with MarketView(resolve_as_of(as_of), db_path) as view, guarded():
            yield view
    except FileNotFoundError as exc:
        raise DataUnavailableError(str(exc)) from exc
    except duckdb.IOException as exc:  # a writer holds the file (R7)
        raise DataUnavailableError(f"database busy (a writer holds it): {exc}") from exc


def _known(ticker: str) -> str:
    t = ticker.strip().upper()
    if t not in known_tickers():
        raise UnknownTickerError(f"{t} is not in config/universe.yaml")
    return t


def _brief_strategy() -> Any:
    cfg = api_config()["brief"]
    try:
        return build(cfg["strategy"])
    except FileNotFoundError as exc:  # stacked model files missing before `make train`
        log.warning(
            "brief: %s unavailable (%s); using %s", cfg["strategy"], exc, cfg["fallback_strategy"]
        )
        return build(cfg["fallback_strategy"])


def brief_for(day: date | datetime | None = None, db_path: Path | None = None) -> Brief:
    """One row per watchlist stock: last close, change, sentiment index, events, one signal."""
    cfg = api_config()["brief"]
    strat = _brief_strategy()
    items: list[BriefItem] = []
    with _session(day, db_path) as view:
        for w in watchlist().items:
            if w.kind != "stock":
                continue
            bars = view.prices(w.ticker, lookback_days=int(cfg["price_lookback_days"]))
            last = bars[-1].close if bars else None
            change = (last / bars[-2].close - 1) * 100 if len(bars) >= 2 else None
            scores = [s for _, s in window_scores(w.ticker, view, cfg["sentiment_method"])]
            events = view.events(w.ticker, lookback_days=int(cfg["events_lookback_days"]))
            items.append(
                BriefItem(
                    ticker=w.ticker,
                    last_close=last,
                    change_pct=change,
                    sentiment_index=sum(scores) / len(scores) if scores else None,
                    events=events,
                    signal=strat.signal(w.ticker, view),
                )
            )
        return Brief(as_of=view.as_of, items=items)


def events_for(
    ticker: str,
    lookback_days: int = 90,
    as_of: date | datetime | None = None,
    db_path: Path | None = None,
) -> EventsResponse:
    """8-K filings (and 10-Q/10-K headers) with their event labels and extracted facts."""
    t = _known(ticker)
    with _session(as_of, db_path) as view:
        filings = view.filings(t, lookback_days=lookback_days)
        refs = {f.accession_no for f in filings}
        events = [e for e in view.events(t, lookback_days=lookback_days) if e.source_ref in refs]
        facts = view.facts(
            t, lookback_quarters=int(api_config()["events"]["facts_lookback_quarters"])
        )
    return EventsResponse(ticker=t, filings=filings, events=events, facts=facts)


def news_for(
    ticker: str,
    lookback_days: int | None = None,
    as_of: date | datetime | None = None,
    db_path: Path | None = None,
) -> NewsResponse:
    """Newest news with their sentiment; each entry is a citable record (id, url, available_at)."""
    cfg = api_config()
    t = _known(ticker)
    days = lookback_days or int(cfg["news"]["default_lookback_days"])
    method = cfg["brief"]["sentiment_method"]
    with _session(as_of, db_path) as view:
        news = view.news(t, lookback_days=days)
        sent = {s.source_ref: s for s in view.sentiment(t, lookback_days=days, method=method)}
        as_of_used = view.as_of
    newest = sorted(news, key=lambda n: n.available_at, reverse=True)[
        : int(cfg["news"]["max_items"])
    ]
    return NewsResponse(
        ticker=t,
        as_of=as_of_used,
        items=[NewsEntry(news=n, sentiment=sent[n.id] if n.id in sent else None) for n in newest],
    )


def prices_for(
    ticker: str,
    lookback_days: int | None = None,
    as_of: date | datetime | None = None,
    db_path: Path | None = None,
) -> PricesResponse:
    """Daily bars published by the cutoff (the open of D is execution data: never served)."""
    t = _known(ticker)
    days = lookback_days or int(api_config()["prices"]["default_lookback_days"])
    with _session(as_of, db_path) as view:
        bars: list[PriceBar] = view.prices(t, lookback_days=days)
        return PricesResponse(ticker=t, as_of=view.as_of, bars=bars)
