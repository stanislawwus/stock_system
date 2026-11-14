"""Strategy maths on hand-checkable closes (src/strategies)."""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum
from src.strategies.registry import BUILDERS, build

AS_OF = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)


class FakeView:
    """Duck-typed MarketView for pure maths tests: closes oldest first, the last = close of D-1."""

    def __init__(self, closes: list[float]) -> None:
        self.as_of = AS_OF
        self._closes = closes

    def prices(self, ticker: str, lookback_days: int = 250) -> list[SimpleNamespace]:
        return [SimpleNamespace(close=c) for c in self._closes[-lookback_days:]]


def test_buy_and_hold_is_always_long() -> None:
    s = BuyAndHold().signal("AAPL", FakeView([]))  # type: ignore[arg-type]
    assert (s.score, s.as_of, s.available_at, s.strategy) == (1.0, AS_OF, AS_OF, "buy_and_hold")


def test_momentum_12_1_uses_close_t_minus_skip_over_close_t_minus_lookback() -> None:
    m = Momentum(lookback_days=4, skip_days=1, score_scale=0.5, horizon_days=2)
    # c[-1-1] = 120 (skip the last close 110), c[-1-4] = 100 -> r = 0.20 -> score 0.4
    s = m.signal("A", FakeView([100, 90, 80, 120, 110]))  # type: ignore[arg-type]
    assert s.score == pytest.approx(0.4)
    s = m.signal("A", FakeView([100, 90, 80, 60, 200]))  # type: ignore[arg-type]
    assert s.score == pytest.approx(-0.8)  # r = 60/100 - 1; the jump to 200 is skipped
    assert m.signal("A", FakeView([1, 2, 3, 4])).score == 0.0  # type: ignore[arg-type]
    big = m.signal("A", FakeView([10, 1, 1, 100, 1]))  # type: ignore[arg-type]
    assert big.score == 1.0  # clipped


def test_mean_reversion_z_score_by_hand() -> None:
    mr = MeanReversion(lookback_days=2, vol_window=4, entry_z=1.0, z_scale=2.0, horizon_days=3)
    c = [100, 101, 100, 101, 95]
    rets = [101 / 100 - 1, 100 / 101 - 1, 101 / 100 - 1, 95 / 101 - 1]
    z = (95 / 100 - 1) / (statistics.stdev(rets) * math.sqrt(2))
    s = mr.signal("A", FakeView(c))  # type: ignore[arg-type]
    assert s.score == pytest.approx(max(-1, min(1, (-z - 1.0) / 2.0)))
    assert s.score > 0  # a sharp drop -> long
    up = mr.signal("A", FakeView([100, 99, 100, 99, 106]))  # type: ignore[arg-type]
    assert up.score <= 0  # a rise -> cash
    assert mr.signal("A", FakeView([1, 2, 3])).score == 0.0  # type: ignore[arg-type]
    assert mr.signal("A", FakeView([5.0] * 6)).reasons == ["zero_volatility"]  # type: ignore[arg-type]


def test_registry_builds_every_strategy_from_config() -> None:
    assert set(BUILDERS) == {"buy_and_hold", "momentum", "mean_reversion"}
    for name in BUILDERS:
        s = build(name)
        assert isinstance(s, Strategy) and s.name == name
    assert build("momentum").lookback_days == 252  # type: ignore[attr-defined]
    with pytest.raises(KeyError):
        build("nope")
