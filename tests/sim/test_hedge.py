"""Hedge aggregation and the confidence threshold (src/strategies/hedge.py, D-44)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from src.impact.labels import abnormal_returns
from src.marketview import MarketView
from src.marketview.clock import cutoff_for
from src.strategies import hedge
from src.strategies.hedge import AggregateHedge, component_loss, daily_losses, hedge_weights
from tests.sim.conftest import business_days, wave

COMP = ["sentiment", "event", "impact"]


def test_component_loss_is_bounded_and_rewards_the_right_side() -> None:
    assert component_loss(1.0, 0.02) == 0.0
    assert component_loss(-1.0, 0.02) == 1.0
    assert component_loss(0.0, -0.01) == 0.5
    assert component_loss(5.0, -0.01) == 1.0  # scores are clipped to [-1, 1]
    assert component_loss(0.7, 0.0) == 0.5  # no move: no information


def test_daily_losses_average_over_tickers_with_a_label() -> None:
    parts = {"A": {"sentiment": 1.0, "event": 0.0, "impact": -1.0},
             "B": {"sentiment": 1.0, "event": 0.0, "impact": 1.0}}  # fmt: skip
    out = daily_losses(parts, {"A": 0.01, "B": -0.01}, COMP)
    assert out == {"sentiment": 0.5, "event": 0.5, "impact": 1.0}  # impact wrong on both
    assert daily_losses(parts, {"C": 0.01}, COMP) is None


def test_hedge_weights_eta_zero_is_equal_and_losers_lose_weight() -> None:
    rows = [{"sentiment": 1.0, "event": 0.5, "impact": 0.0}] * 3
    assert hedge_weights(rows, COMP, 0.0) == pytest.approx(dict.fromkeys(COMP, 1 / 3))
    w = hedge_weights(rows, COMP, 1.0)
    assert w["impact"] > w["event"] > w["sentiment"] and sum(w.values()) == pytest.approx(1)
    assert hedge_weights([], COMP, 5.0) == pytest.approx(dict.fromkeys(COMP, 1 / 3))
    assert sum(hedge_weights(rows * 1000, COMP, 50.0).values()) == pytest.approx(1)  # stable


@pytest.fixture()
def strat(monkeypatch: pytest.MonkeyPatch) -> Callable[..., AggregateHedge]:
    """No trained artifacts needed: the table and model are stubs, scores come from a dict."""
    monkeypatch.setattr(hedge.AggregateEqual, "__init__", _fake_init)

    def build(parts: dict, eta: float = 5.0, p_min: float | None = None) -> AggregateHedge:
        return AggregateHedge(1, 0.0, eta, p_min, 20, parts_source=parts)

    return build


def _fake_init(self: AggregateHedge, hold_days: int, entry_threshold: float, *a: object) -> None:
    self.horizon_days = self.min_hold_days = hold_days
    self.entry_threshold = entry_threshold
    self.benchmark = "SPY"
    self.components = list(COMP)
    self.parts_log = None


def _wave_db(price_db: Callable[..., Path]) -> Path:
    return price_db()  # tickers A, B, SPY (tests/sim/conftest.py wave prices)


def _true_drift_sign(ticker: str, d: date) -> float:
    """sign(drift_ar) of the fixture, computed the way the label is (D-28)."""
    days = business_days()
    bars = {t: {x: (wave(i, t) * (1 - 0.002), wave(i, t)) for i, x in enumerate(days)}
            for t in (ticker, "SPY")}  # fmt: skip
    ar = abnormal_returns(d, bars[ticker], bars["SPY"], days)
    assert ar is not None
    return 1.0 if ar.drift_ar > 0 else -1.0


def test_weights_move_to_the_component_that_was_right_after_two_sessions(
    strat: Callable[..., AggregateHedge], price_db: Callable[..., Path]
) -> None:
    """'impact' always has the sign of the realised drift_ar, 'sentiment' always the opposite
    (an oracle built from the fixture). Labels of D become usable at D+2 (close of D+1)."""
    db = _wave_db(price_db)
    days = [d for d in business_days() if d >= date(2026, 6, 1)][:8]
    parts = {}
    for d in days:
        for t in ("A", "B"):
            s = _true_drift_sign(t, d)
            parts[(t, d)] = {"sentiment": -s, "event": 0.0, "impact": s, "p": 0.6}
    s = strat(parts)
    seen = []
    for d in days:
        with MarketView(cutoff_for(d), db) as view:
            for t in ("A", "B"):
                s.signal(t, view)
        seen.append(dict(s.last_weights))
    assert seen[0] == pytest.approx(dict.fromkeys(COMP, 1 / 3))
    assert seen[1] == pytest.approx(dict.fromkeys(COMP, 1 / 3))  # day 0 not realised yet
    assert seen[-1]["impact"] > 0.9 and seen[-1]["sentiment"] < 0.01
    # a label is never used before its close: day i's weights use only days <= i - 2
    assert sorted(s.losses)[-1] <= days[-3]


def test_confidence_threshold_turns_low_p_into_cash(
    strat: Callable[..., AggregateHedge], price_db: Callable[..., Path]
) -> None:
    db = _wave_db(price_db)
    d = date(2026, 6, 2)
    parts = {("A", d): {"sentiment": 1.0, "event": 1.0, "impact": 1.0, "p": 0.51}}
    with MarketView(cutoff_for(d), db) as view:
        assert strat(parts, p_min=None).signal("A", view).score > 0
        low = strat(parts, p_min=0.55).signal("A", view)
        assert low.score == 0.0 and low.calibrated_p == 0.51
        assert strat(parts, p_min=0.5).signal("A", view).score > 0
