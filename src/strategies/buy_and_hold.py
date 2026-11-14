"""Buy-and-hold baseline: always long every ticker (equal weight in the simulator)."""

from __future__ import annotations

from src.marketview import MarketView
from src.schemas import Signal
from src.strategies.base import make_signal, strategy_params


class BuyAndHold:
    name = "buy_and_hold"

    def __init__(self, horizon_days: int | None = None) -> None:
        self.horizon_days = horizon_days

    @classmethod
    def from_config(cls) -> BuyAndHold:
        return cls(**strategy_params(cls.name))

    def signal(self, ticker: str, view: MarketView) -> Signal:
        return make_signal(self.name, ticker, view, 1.0, "always_long")
