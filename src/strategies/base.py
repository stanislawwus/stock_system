"""Strategy protocol and shared helpers (AGENTS.md R2, R8).

A strategy sees one thing: a `MarketView` built for the 09:00 ET cutoff of the decision day. It
has no `as_of` argument of its own (the view carries it), no database handle and no access to the
open of D, so the only information it can use is what was public before the cutoff. It returns a
`Signal` whose `score <= 0` means cash (long-only, D-08).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from src.marketview import MarketView
from src.schemas import Signal

CONFIG = Path(__file__).resolve().parents[2] / "config" / "strategies.yaml"


@lru_cache(maxsize=1)
def _strategies_yaml() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def strategy_params(name: str) -> dict[str, Any]:
    """Parameters of one strategy from config/strategies.yaml (R14: no magic numbers)."""
    return dict(_strategies_yaml()[name])


def backtest_config() -> dict[str, Any]:
    return dict(_strategies_yaml()["backtest"])


@runtime_checkable
class Strategy(Protocol):
    """`signal(ticker, view)`; `horizon_days` = h: a position opened at the open of D is closed at
    the close of D+h (None = no time exit)."""

    name: str
    horizon_days: int | None

    def signal(self, ticker: str, view: MarketView) -> Signal: ...


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def make_signal(name: str, ticker: str, view: MarketView, score: float, *reasons: str) -> Signal:
    return Signal(
        ticker=ticker, as_of=view.as_of, strategy=name, score=clip(score), reasons=list(reasons)
    )


def closes(view: MarketView, ticker: str, n: int) -> list[float]:
    """The last `n` closes visible at the view's cutoff, oldest first (bar of D-1 is the newest)."""
    return [b.close for b in view.prices(ticker, lookback_days=n)]
