"""Signals service behind GET /api/v1/signals (AGENTS.md R2, R3, R13).

Every registered strategy is asked for its signal on one `MarketView(as_of)`, inside
`src.db.guarded()`. `as_of` is either a trading day D (then the 09:00 ET cutoff of D, D-03) or an
explicit tz-aware instant. Default: the cutoff of the last validation day, because MarketView
refuses any `as_of` in the test period outside `make test-final` (R3), so "today" is not servable.
No aggregation yet (the aggregator is a later component): `aggregated` is None.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from src.collectors.common import load_periods, load_universe
from src.db import guarded
from src.marketview import MarketView
from src.marketview.clock import cutoff_for
from src.schemas import Signal, SignalsResponse
from src.splits import TestPeriodLockedError
from src.strategies.registry import BUILDERS, build

__all__ = ["TestPeriodLockedError", "UnknownTickerError", "signals_for"]


class UnknownTickerError(ValueError):
    """The ticker is not in config/universe.yaml."""


def known_tickers() -> set[str]:
    companies, benchmark = load_universe()
    return {c.ticker for c in companies} | {benchmark}


def default_as_of() -> datetime:
    (val,) = load_periods(["validation"])
    return cutoff_for(val.end)


def resolve_as_of(as_of: date | datetime | None) -> datetime:
    if as_of is None:
        return default_as_of()
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            raise ValueError("as_of must carry a timezone (e.g. ...Z); naive times are ambiguous")
        return as_of
    return cutoff_for(as_of)


def signals_for(
    ticker: str,
    as_of: date | datetime | None = None,
    db_path: Path | None = None,
    strategies: list[str] | None = None,
) -> SignalsResponse:
    """All strategy signals for `ticker` at `as_of`. Raises UnknownTickerError, ValueError,
    TestPeriodLockedError (test period), FileNotFoundError (no database)."""
    t = ticker.strip().upper()
    if t not in known_tickers():
        raise UnknownTickerError(f"{t} is not in config/universe.yaml")
    cutoff = resolve_as_of(as_of)
    names = strategies or sorted(BUILDERS)
    out: list[Signal] = []
    with MarketView(cutoff, db_path) as view, guarded():
        for name in names:
            out.append(build(name).signal(t, view))
    return SignalsResponse(ticker=t, as_of=cutoff, signals=out, aggregated=None)
