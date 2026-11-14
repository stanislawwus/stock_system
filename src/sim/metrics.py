"""Backtest metrics and moving-block bootstrap confidence intervals.

Conventions: daily simple returns of close-to-close equity, the first one relative to the initial
capital; risk-free rate 0; Sharpe annualised with sqrt(trading_days_per_year); max drawdown as a
positive fraction of the running peak (the initial capital counts as the first peak).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import NormalDist

import numpy as np

ZERO_SD = 1e-12  # numerical floor: a constant series has float stdev ~1e-18, not 0


def daily_returns(equity: Sequence[float], initial: float) -> np.ndarray:
    e = np.asarray([initial, *equity], dtype=float)
    return e[1:] / e[:-1] - 1.0


def total_return(equity: Sequence[float], initial: float) -> float:
    return float(equity[-1] / initial - 1.0) if len(equity) else 0.0


def annualised_return(total: float, n_days: int, days_per_year: int) -> float:
    if n_days == 0 or total <= -1.0:
        return -1.0 if total <= -1.0 else 0.0
    return float((1.0 + total) ** (days_per_year / n_days) - 1.0)


def sharpe(returns: Sequence[float], days_per_year: int) -> float | None:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return None
    sd = r.std(ddof=1)
    if sd <= ZERO_SD or not math.isfinite(sd):
        return None
    return float(r.mean() / sd * math.sqrt(days_per_year))


def max_drawdown(equity: Sequence[float], initial: float) -> float:
    e = np.asarray([initial, *equity], dtype=float)
    peak = np.maximum.accumulate(e)
    return float(np.max(1.0 - e / peak))


def turnover(traded_notional: float, equity: Sequence[float]) -> float:
    """One-way turnover over the period: (buys + sells) / 2 / mean equity."""
    if not len(equity):
        return 0.0
    return float(traded_notional / 2.0 / np.mean(equity))


def hit_rate(round_trip_pnls: Sequence[float]) -> float | None:
    """Share of closed trades (entry to exit, costs included) with a positive net P&L."""
    if not round_trip_pnls:
        return None
    return float(np.mean([p > 0 for p in round_trip_pnls]))


def _block_indices(n: int, block: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, n))
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_resamples, -1)
    return idx[:, :n]


def bootstrap_ci(
    returns: Sequence[float],
    stat: Callable[[np.ndarray], np.ndarray],
    *,
    n_resamples: int,
    block_days: int,
    ci: float,
    seed: int,
) -> tuple[float, float] | None:
    """Percentile CI of `stat` (vectorised over rows) under a moving-block bootstrap."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = r[_block_indices(r.size, block_days, n_resamples, rng)]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = stat(samples)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant (expected maximum of N normals)


def moments(returns: Sequence[float]) -> tuple[float, float]:
    """(skewness, kurtosis) of daily returns; kurtosis is the plain (non-excess) one, 3 = normal."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=0)
    if r.size < 3 or sd <= ZERO_SD:
        return 0.0, 3.0
    z = (r - r.mean()) / sd
    return float(np.mean(z**3)), float(np.mean(z**4))


def expected_max_sharpe(trial_sharpes: Sequence[float], n_trials: int) -> float:
    """SR0 of Bailey & Lopez de Prado (2014): the Sharpe the best of `n_trials` unskilled
    configurations reaches by luck, given the variance of the observed trial Sharpes (same
    units as the inputs, here per day)."""
    s = np.asarray(trial_sharpes, dtype=float)
    if n_trials < 2 or s.size < 2:
        return 0.0
    nd = NormalDist()
    sd = float(s.std(ddof=1))
    return sd * (
        (1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * nd.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def probabilistic_sharpe(
    sr: float, sr0: float, n_obs: int, skew: float, kurt: float
) -> float | None:
    """P(true Sharpe > sr0) given an observed per-period Sharpe `sr` over `n_obs` returns
    (Bailey & Lopez de Prado 2012), corrected for skewness and fat tails."""
    if n_obs < 2:
        return None
    var = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if var <= 0:
        return None
    return float(NormalDist().cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(var)))


def deflated_sharpe(
    returns: Sequence[float], trial_sharpes: Sequence[float], n_trials: int
) -> dict[str, float | None]:
    """Deflated Sharpe ratio: the probabilistic Sharpe against SR0 = expected max of
    `n_trials` (all configurations tried). `trial_sharpes` are per-day Sharpes of the trials."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    sr = float(r.mean() / sd) if sd > ZERO_SD else 0.0
    skew, kurt = moments(r)
    sr0 = expected_max_sharpe(trial_sharpes, n_trials)
    return {
        "sharpe_daily": sr,
        "sr0_daily": sr0,
        "skew": skew,
        "kurtosis": kurt,
        "dsr": probabilistic_sharpe(sr, sr0, int(r.size), skew, kurt),
        "psr_vs_zero": probabilistic_sharpe(sr, 0.0, int(r.size), skew, kurt),
    }


def mean_stat(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1)


def sharpe_stat(days_per_year: int) -> Callable[[np.ndarray], np.ndarray]:
    def _stat(samples: np.ndarray) -> np.ndarray:
        sd = samples.std(axis=1, ddof=1)
        return np.where(sd > ZERO_SD, samples.mean(axis=1) / sd * math.sqrt(days_per_year), np.nan)

    return _stat
