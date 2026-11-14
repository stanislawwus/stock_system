"""No look-ahead in the simulator (AGENTS.md R2, R3; time-leak-check steps 1, 2, 7).

Strategies receive only a MarketView at 09:00 ET of D; the newest bar they can see is D-1; direct
database access inside a strategy raises; signals for D do not change when bars of D and later
change (shift test); the execution accessor and the backtest service refuse the test period.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from src import db
from src.marketview import MarketView
from src.marketview.clock import cutoff_for
from src.schemas import Signal
from src.sim.backtest import resolve_split
from src.sim.engine import SimConfig, collect_signals, default_view_factory, simulate
from src.sim.execution import ExecutionPrices
from src.splits import TestPeriodLockedError
from src.strategies.base import make_signal
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum
from src.strategies.registry import BUILDERS, build
from tests.sim.conftest import business_days, write_bars

DAYS = business_days(date(2026, 7, 1), date(2026, 7, 31))
CFG = SimConfig(initial_capital=10_000, max_position_pct=0.5, cost_per_side=0.0005)


class Spy:
    name = "spy"
    horizon_days = None

    def __init__(self) -> None:
        self.seen: list[tuple[type, object, date, date]] = []

    def signal(self, ticker: str, view: MarketView) -> Signal:
        newest = view.prices(ticker, lookback_days=5)[-1].bar_date
        self.seen.append((type(view), view.as_of, view.as_of.date(), newest))
        return make_signal(self.name, ticker, view, 1.0)


class Peeker:
    """Tries to bypass MarketView."""

    name = "peeker"
    horizon_days = None

    def __init__(self, path: Path) -> None:
        self.path = path

    def signal(self, ticker: str, view: MarketView) -> Signal:
        db.connect(self.path)
        return make_signal(self.name, ticker, view, 1.0)


def _prices(path: Path) -> ExecutionPrices:
    return ExecutionPrices.load(["A", "B"], DAYS[0], DAYS[-1], "SPY", path)


def test_strategies_receive_only_a_marketview_at_the_cutoff(price_db: Callable[..., Path]) -> None:
    path = price_db()
    spy = Spy()
    simulate(spy, ["A", "B"], _prices(path), CFG, default_view_factory(path))
    assert len(spy.seen) == 2 * len(DAYS)
    for kind, as_of, day, newest in spy.seen:
        assert kind is MarketView
        assert as_of == cutoff_for(day)
        assert newest < day  # the bar of D closes at 16:00 ET, after the 09:00 ET cutoff


def test_strategy_signature_has_no_clock_and_no_prices() -> None:
    for name in BUILDERS:
        params = list(inspect.signature(build(name).signal).parameters)
        assert params == ["ticker", "view"], name


def test_direct_database_access_inside_a_strategy_raises(price_db: Callable[..., Path]) -> None:
    path = price_db()
    with pytest.raises(db.DirectDataAccessError):
        simulate(Peeker(path), ["A"], _prices(path), CFG, default_view_factory(path))


@pytest.mark.parametrize(
    "strategy",
    [
        Momentum(lookback_days=60, skip_days=5, score_scale=0.5, horizon_days=5),
        MeanReversion(lookback_days=3, vol_window=20, entry_z=0.5, z_scale=2.0, horizon_days=2),
    ],
    ids=["momentum", "mean_reversion"],
)
def test_signals_for_d_ignore_bars_of_d_and_later(
    strategy: object, price_db: Callable[..., Path]
) -> None:
    """Shift test: corrupt every bar from the pivot day on; signals up to it are unchanged."""
    path = price_db()
    vf = default_view_factory(path)
    before = [collect_signals(strategy, ["A", "B"], d, vf) for d in DAYS]  # type: ignore[arg-type]
    assert {s.score > 0 for sig in before for s in sig.values()} == {True, False}  # both signs
    pivot = 10
    write_bars(
        path,
        [(t, d, 1.0, 1000.0 * (i + 1)) for t in ("A", "B") for i, d in enumerate(DAYS[pivot:])],
    )
    after = [collect_signals(strategy, ["A", "B"], d, vf) for d in DAYS]  # type: ignore[arg-type]
    assert after[: pivot + 1] == before[: pivot + 1]  # D = pivot sees only D-1
    assert after[pivot + 1 :] != before[pivot + 1 :]  # the corruption is visible later, as it must


def test_execution_prices_refuse_the_test_period(price_db: Callable[..., Path]) -> None:
    path = price_db()
    with pytest.raises(TestPeriodLockedError):
        ExecutionPrices.load(["A"], date(2026, 8, 3), date(2026, 9, 2), "SPY", path)


@pytest.mark.parametrize("split", ["test", "warmup", "all"])
def test_backtest_accepts_only_train_and_validation(split: str) -> None:
    with pytest.raises(ValueError):
        resolve_split(split)
    assert resolve_split("val") == resolve_split("validation") == "validation"
    assert resolve_split("train") == "train"
