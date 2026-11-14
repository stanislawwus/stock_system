"""Matched random controls (src/sim/controls.py, D-42) on an in-memory market."""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pytest

from evals.sim_fixtures import Scripted, market
from src.sim.controls import (
    RoundTrip,
    random_controls,
    random_schedule,
    round_trips,
    run_control,
)
from src.sim.engine import SimConfig, SimResult, simulate

TICKERS = ["A", "B", "C", "D"]
CFG = SimConfig(initial_capital=10_000.0, max_position_pct=0.3, cost_per_side=0.0005)


def _days(n: int = 30) -> list[date]:
    out, d = [], date(2026, 6, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = _days()


def _prices():  # noqa: ANN202
    bars = {}
    for k, t in enumerate(TICKERS):
        closes = [100 * math.exp(0.02 * math.sin(i / 3 + k) + 0.001 * k * i) for i in range(30)]
        bars[t] = [(c * (1 - 0.003 * math.cos(i + k)), c) for i, c in enumerate(closes)]
    return market(bars, DAYS)


def _strategy_run() -> SimResult:
    """Score exits (score drops to 0 at an open) and time exits (close of D+2) mixed."""
    scores = {}
    for i, d in enumerate(DAYS):
        scores[d] = {t: 1.0 for k, t in enumerate(TICKERS) if (i + k) % 5 in (0, 1)}
    return simulate(Scripted(scores, horizon_days=2), TICKERS, _prices(), CFG)


def test_round_trips_pair_every_buy_with_its_sell() -> None:
    res = _strategy_run()
    trips = round_trips(res)
    assert len(trips) == len(res.round_trip_pnls) > 10
    assert {tp.at_close for tp in trips} == {True, False}  # both exit types are present
    assert all(tp.length >= 0 for tp in trips)


def test_replaying_the_strategys_own_trips_reproduces_its_equity() -> None:
    """The control path (schedule -> engine) is exact: same trips give the same equity curve."""
    res = _strategy_run()
    ctl = run_control(round_trips(res), TICKERS, _prices(), CFG)
    assert len(ctl.round_trip_pnls) == len(res.round_trip_pnls)
    assert ctl.equity == pytest.approx(res.equity, rel=1e-12)
    assert ctl.exposure == pytest.approx(res.exposure, rel=1e-12)


def _no_overlap(schedule: list[RoundTrip]) -> bool:
    spans: dict[str, list[tuple[int, int]]] = {}
    for tp in schedule:
        for a, b in spans.get(tp.ticker, []):
            if tp.entry <= b and a <= tp.entry + tp.length:
                return False
        spans.setdefault(tp.ticker, []).append((tp.entry, tp.entry + tp.length))
    return True


def test_names_mode_keeps_entry_days_lengths_and_exit_types() -> None:
    trips = round_trips(_strategy_run())
    sched = random_schedule(trips, TICKERS, _prices(), np.random.default_rng(0), "names")
    key = lambda tp: (tp.entry, tp.length, tp.at_close)  # noqa: E731
    assert Counter(map(key, sched)) == Counter(map(key, trips))
    assert _no_overlap(sched)
    assert [tp.ticker for tp in sched] != [tp.ticker for tp in sorted(trips, key=lambda x: x.entry)]


def test_timing_mode_keeps_lengths_and_is_seeded() -> None:
    trips = round_trips(_strategy_run())[:6]
    a = random_schedule(trips, TICKERS, _prices(), np.random.default_rng(1))
    b = random_schedule(trips, TICKERS, _prices(), np.random.default_rng(1))
    assert a == b
    assert sorted(tp.length for tp in a) == sorted(tp.length for tp in trips)
    assert _no_overlap(a)


def test_a_trip_that_cannot_fit_is_dropped_not_overlapped() -> None:
    long = [RoundTrip("A", 0, 25, True)] * 6  # 4 tickers x 30 sessions: only 4 such trips fit
    sched = random_schedule(long, TICKERS, _prices(), np.random.default_rng(0))
    assert len(sched) == 4 and _no_overlap(sched)


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="mode"):
        random_schedule([], TICKERS, _prices(), np.random.default_rng(0), "timing")


@pytest.mark.parametrize("mode", ["names", "timing_and_names"])
def test_random_controls_report_a_percentile_and_the_match(mode: str) -> None:
    res = _strategy_run()
    out = random_controls(res, TICKERS, _prices(), CFG, n=30, seed=3, days_per_year=252, mode=mode)
    assert out["n"] == 30 and out["mode"] == mode
    assert 0.0 <= out["percentile"] <= 100.0
    assert out["random_sharpe_p05"] <= out["random_sharpe_median"] <= out["random_sharpe_p95"]
    assert out["round_trips"] == len(res.round_trip_pnls)
    if mode == "names":
        assert out["random_round_trips_mean"] == pytest.approx(out["round_trips"], abs=1)
        assert out["random_mean_exposure"] == pytest.approx(out["mean_exposure"], abs=0.02)


def test_a_strategy_without_trades_has_no_percentile() -> None:
    res = simulate(Scripted({}), TICKERS, _prices(), CFG)
    out = random_controls(res, TICKERS, _prices(), CFG, n=5, seed=0, days_per_year=252)
    assert out["round_trips"] == 0 and out["percentile"] is None
