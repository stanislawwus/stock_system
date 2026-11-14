"""Name -> strategy built from config/strategies.yaml."""

from __future__ import annotations

from collections.abc import Callable

from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum

BUILDERS: dict[str, Callable[[], Strategy]] = {
    BuyAndHold.name: BuyAndHold.from_config,
    Momentum.name: Momentum.from_config,
    MeanReversion.name: MeanReversion.from_config,
}
BENCHMARK = BuyAndHold.name


def build(name: str) -> Strategy:
    if name not in BUILDERS:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[name]()
