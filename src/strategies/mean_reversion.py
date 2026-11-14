"""Short-term mean reversion (D-35): fade a large k-day drop.

z = r_k / (sigma_daily * sqrt(k)), with r_k the k-day return to the last visible close (D-1) and
sigma_daily the stdev of the last `vol_window` daily returns. Long when z <= -entry_z;
score = clip((-z - entry_z) / z_scale, -1, 1), so the score turns <= 0 (cash) as soon as the drop
is no longer `entry_z` standard deviations deep.
"""

from __future__ import annotations

import math
import statistics

from src.marketview import MarketView
from src.schemas import Signal
from src.strategies.base import closes, make_signal, strategy_params


class MeanReversion:
    name = "mean_reversion"

    def __init__(
        self,
        lookback_days: int,
        vol_window: int,
        entry_z: float,
        z_scale: float,
        horizon_days: int | None,
    ) -> None:
        if lookback_days < 1 or vol_window < 2:
            raise ValueError("need lookback_days >= 1 and vol_window >= 2")
        self.lookback_days = lookback_days
        self.vol_window = vol_window
        self.entry_z = entry_z
        self.z_scale = z_scale
        self.horizon_days = horizon_days

    @classmethod
    def from_config(cls) -> MeanReversion:
        return cls(**strategy_params(cls.name))

    def zscore(self, c: list[float]) -> float | None:
        rets = [c[i] / c[i - 1] - 1.0 for i in range(len(c) - self.vol_window, len(c))]
        sigma = statistics.stdev(rets)
        if sigma == 0:
            return None
        r_k = c[-1] / c[-1 - self.lookback_days] - 1.0
        return r_k / (sigma * math.sqrt(self.lookback_days))

    def signal(self, ticker: str, view: MarketView) -> Signal:
        need = max(self.vol_window, self.lookback_days) + 1
        c = closes(view, ticker, need)
        if len(c) < need:
            return make_signal(self.name, ticker, view, 0.0, "insufficient_history")
        z = self.zscore(c)
        if z is None:
            return make_signal(self.name, ticker, view, 0.0, "zero_volatility")
        score = (-z - self.entry_z) / self.z_scale
        return make_signal(self.name, ticker, view, score, f"z_{self.lookback_days}d={z:.2f}")
