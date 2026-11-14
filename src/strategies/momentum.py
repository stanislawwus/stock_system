"""Simplified 12-1 time-series momentum (D-35).

r_12_1 = close[t - skip] / close[t - lookback] - 1, where t is the last bar visible at the cutoff
(the close of D-1). The most recent month is skipped because one-month returns tend to reverse.
score = clip(r_12_1 / score_scale, -1, 1): long when the past year (minus a month) went up, cash
otherwise. Per-ticker (time-series), not a cross-sectional ranking, so it fits the per-ticker
`Signal` contract and works for a 10-name watchlist.
"""

from __future__ import annotations

from src.marketview import MarketView
from src.schemas import Signal
from src.strategies.base import closes, make_signal, strategy_params


class Momentum:
    name = "momentum"

    def __init__(
        self, lookback_days: int, skip_days: int, score_scale: float, horizon_days: int | None
    ) -> None:
        if not 0 <= skip_days < lookback_days:
            raise ValueError("need 0 <= skip_days < lookback_days")
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.score_scale = score_scale
        self.horizon_days = horizon_days

    @classmethod
    def from_config(cls) -> Momentum:
        return cls(**strategy_params(cls.name))

    def signal(self, ticker: str, view: MarketView) -> Signal:
        c = closes(view, ticker, self.lookback_days + 1)
        if len(c) < self.lookback_days + 1:
            return make_signal(self.name, ticker, view, 0.0, "insufficient_history")
        r = c[-1 - self.skip_days] / c[-1 - self.lookback_days] - 1.0
        reason = f"r_{self.lookback_days}_{self.skip_days}={r:.3f}"
        return make_signal(self.name, ticker, view, r / self.score_scale, reason)
