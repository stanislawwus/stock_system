"""Backtest metrics (src/sim/metrics.py) on hand-checkable series."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.sim import metrics as m


def test_returns_total_and_annualised() -> None:
    r = m.daily_returns([110, 99, 121], initial=100)
    assert r.tolist() == pytest.approx([0.10, -0.10, 121 / 99 - 1])
    assert m.total_return([110, 99, 121], 100) == pytest.approx(0.21)
    assert m.annualised_return(0.21, 2, 252) == pytest.approx(1.21**126 - 1)
    assert m.annualised_return(0.0, 0, 252) == 0.0


def test_sharpe_is_mean_over_sample_stdev_times_sqrt_252() -> None:
    r = [0.01, -0.01, 0.02]
    sd = math.sqrt(((0.01 - 0.02 / 3) ** 2 + (-0.01 - 0.02 / 3) ** 2 + (0.02 - 0.02 / 3) ** 2) / 2)
    assert m.sharpe(r, 252) == pytest.approx((0.02 / 3) / sd * math.sqrt(252))
    assert m.sharpe([0.01, 0.01], 252) is None  # zero variance
    assert m.sharpe([0.01], 252) is None


def test_max_drawdown_uses_the_initial_capital_as_first_peak() -> None:
    assert m.max_drawdown([120, 90, 130], 100) == pytest.approx(0.25)
    assert m.max_drawdown([80, 90], 100) == pytest.approx(0.20)
    assert m.max_drawdown([101, 102], 100) == 0.0


def test_turnover_and_hit_rate() -> None:
    assert m.turnover(traded_notional=400, equity=[100, 100]) == pytest.approx(2.0)
    assert m.hit_rate([1.0, -0.5, 0.0, 2.0]) == pytest.approx(0.5)  # 0 is not a hit
    assert m.hit_rate([]) is None


def test_block_bootstrap_ci_is_deterministic_and_covers_the_estimate() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 250)
    kw = {"n_resamples": 500, "block_days": 5, "ci": 0.95, "seed": 1}
    lo, hi = m.bootstrap_ci(r, m.mean_stat, **kw)
    assert lo < r.mean() < hi
    assert m.bootstrap_ci(r, m.mean_stat, **kw) == (lo, hi)
    slo, shi = m.bootstrap_ci(r, m.sharpe_stat(252), **kw)
    assert slo < m.sharpe(r, 252) < shi


def test_block_indices_are_contiguous_blocks() -> None:
    idx = m._block_indices(10, 4, 3, np.random.default_rng(0))
    assert idx.shape == (3, 10)
    for row in idx:
        for start in range(0, 8, 4):
            block = row[start : start + 4]
            assert (np.diff(block) == 1).all()


def test_bootstrap_degenerate_inputs() -> None:
    kw = {"n_resamples": 100, "block_days": 5, "ci": 0.95, "seed": 1}
    assert m.bootstrap_ci([0.01], m.mean_stat, **kw) is None
    assert m.bootstrap_ci([0.01] * 20, m.sharpe_stat(252), **kw) is None
    assert m.bootstrap_ci([0.01] * 20, m.mean_stat, **kw) == pytest.approx((0.01, 0.01))
