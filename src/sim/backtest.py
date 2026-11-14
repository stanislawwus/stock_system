"""Backtest service: one strategy on one development split -> BacktestResult (D-35).

Only `train` and `validation` are accepted here; the test period belongs to `make test-final`
(AGENTS.md R3) and is additionally refused by MarketView and ExecutionPrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src import splits
from src.collectors.common import load_periods, load_universe, load_yaml, watchlist_tickers
from src.marketview.clock import bar_close_for
from src.schemas import BacktestResult
from src.sim import metrics as m
from src.sim.engine import SimConfig, SimResult, default_view_factory, simulate
from src.sim.execution import ExecutionPrices
from src.sim.risk import RiskConfig, RiskManager
from src.strategies.base import Strategy, backtest_config
from src.strategies.registry import build

DEV_SPLITS = {"train": "train", "validation": "validation", "val": "validation"}


def resolve_split(name: str) -> str:
    """A development split; the test split only during the final run (R3, D-43)."""
    if name == splits.FINAL_SPLIT and splits.in_final_run():
        return name
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


@dataclass
class Run:
    """A finished simulation with everything needed to replay controls on the same days."""

    sim: SimResult
    result: BacktestResult
    prices: ExecutionPrices
    cfg: SimConfig
    tickers: list[str]
    risk_managed: bool


def run_simulation(
    strategy: str | Strategy,
    split: str,
    db_path: Path | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    risk_managed: bool | None = None,
    prices: ExecutionPrices | None = None,
) -> Run:
    """Like `run_backtest`, but keeps the raw simulation (trades, exposure) and the prices.
    `risk_managed=None` follows `backtest.risk_managed`; `prices` reuses an already loaded
    accessor for the same split/range (the arena runs many strategies on the same days)."""
    split = resolve_split(split)
    (period,) = load_periods([split])
    lo, hi = max(start or period.start, period.start), min(end or period.end, period.end)
    bt = backtest_config()
    tickers, benchmark = backtest_tickers(bt["universe"])
    strat = build(strategy) if isinstance(strategy, str) else strategy
    risk_yaml = load_yaml("risk.yaml")
    cfg = SimConfig.from_risk(risk_yaml)
    if risk_managed is None:
        risk_managed = strat.name in bt.get("risk_managed", [])
    rm = RiskManager(RiskConfig.from_risk(risk_yaml)) if risk_managed else None
    if prices is None:
        prices = ExecutionPrices.load(tickers, lo, hi, benchmark, db_path)
    res = simulate(strat, tickers, prices, cfg, default_view_factory(db_path), risk=rm)
    return Run(res, summarise(res, cfg, split, tickers, bt), prices, cfg, tickers, risk_managed)


def run_backtest(
    strategy: str | Strategy,
    split: str,
    db_path: Path | None = None,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    """`start`/`end` narrow the run to a sub-range of the split (e.g. val_b, D-40). Strategies
    listed in `backtest.risk_managed` run with the risk manager (src/sim/risk.py)."""
    return run_simulation(strategy, split, db_path, start, end).result
