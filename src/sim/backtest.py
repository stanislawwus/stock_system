"""Backtest service: one strategy on one development split -> BacktestResult (D-35).

Only `train` and `validation` are accepted here; the test period belongs to `make test-final`
(AGENTS.md R3) and is additionally refused by MarketView and ExecutionPrices.
"""

from __future__ import annotations

from pathlib import Path

from src.collectors.common import load_periods, load_universe, load_yaml, watchlist_tickers
from src.marketview.clock import bar_close_for
from src.schemas import BacktestResult
from src.sim import metrics as m
from src.sim.engine import SimConfig, SimResult, default_view_factory, simulate
from src.sim.execution import ExecutionPrices
from src.strategies.base import Strategy, backtest_config
from src.strategies.registry import build

DEV_SPLITS = {"train": "train", "validation": "validation", "val": "validation"}


def resolve_split(name: str) -> str:
    if name not in DEV_SPLITS:
        raise ValueError(f"split {name!r} not allowed; use one of {sorted(DEV_SPLITS)} (R3)")
    return DEV_SPLITS[name]


def backtest_tickers(universe: str) -> tuple[list[str], str]:
    """(tickers, benchmark) for `backtest.universe` = watchlist | universe."""
    companies, benchmark = load_universe()
    if universe == "watchlist":
        return watchlist_tickers(), benchmark
    if universe == "universe":
        return [c.ticker for c in companies], benchmark
    raise ValueError(f"unknown backtest universe {universe!r}")


def summarise(
    res: SimResult, cfg: SimConfig, split: str, tickers: list[str], bt: dict
) -> BacktestResult:
    """Metrics of a finished simulation (definitions in src/sim/metrics.py)."""
    dpy = int(bt["trading_days_per_year"])
    boot = bt["bootstrap"]
    kw = {
        "n_resamples": int(boot["n_resamples"]),
        "block_days": int(boot["block_days"]),
        "ci": float(boot["ci"]),
        "seed": int(boot["seed"]),
    }
    r = m.daily_returns(res.equity, cfg.initial_capital)
    total = m.total_return(res.equity, cfg.initial_capital)
    return BacktestResult(
        available_at=bar_close_for(res.days[-1]),
        strategy=res.strategy,
        split=split,
        start=res.days[0],
        end=res.days[-1],
        initial_capital=cfg.initial_capital,
        final_equity=res.equity[-1],
        total_return=total,
        annualised_return=m.annualised_return(total, len(r), dpy),
        sharpe=m.sharpe(r, dpy),
        sharpe_ci95=m.bootstrap_ci(r, m.sharpe_stat(dpy), **kw),
        mean_daily_return=float(r.mean()),
        mean_daily_return_ci95=m.bootstrap_ci(r, m.mean_stat, **kw),
        max_drawdown=m.max_drawdown(res.equity, cfg.initial_capital),
        turnover=m.turnover(res.traded_notional, res.equity),
        n_trades=len(res.trades),
        hit_rate=m.hit_rate(res.round_trip_pnls),
        n_days=len(r),
        costs_paid=res.costs_paid,
        tickers=tickers,
        equity_curve=list(zip(res.days, res.equity, strict=True)),
    )


def run_backtest(
    strategy: str | Strategy, split: str, db_path: Path | None = None
) -> BacktestResult:
    split = resolve_split(split)
    (period,) = load_periods([split])
    bt = backtest_config()
    tickers, benchmark = backtest_tickers(bt["universe"])
    strat = build(strategy) if isinstance(strategy, str) else strategy
    cfg = SimConfig.from_risk(load_yaml("risk.yaml"))
    prices = ExecutionPrices.load(tickers, period.start, period.end, benchmark, db_path)
    res = simulate(strat, tickers, prices, cfg, default_view_factory(db_path))
    return summarise(res, cfg, split, tickers, bt)
