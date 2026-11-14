"""Hold period x entry threshold selection for the signal-stack strategies (D-40).

For each strategy the raw scores are computed once per (ticker, session) of its selection period
(`selection.period`: `event` on train, `impact` and `aggregate_equal` on val_a, R3), then every
grid point is replayed through the simulator with costs and the risk manager. The point with the
highest Sharpe among those with at least `selection.min_trades` round trips is kept; ties go to
the fewer trades. Writes `<model_dir>/selected.json` with the whole grid, so the choice is
auditable. val_b and train-for-val strategies are never looked at here.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.collectors.common import load_periods, load_yaml
from src.marketview import MarketView
from src.schemas import Signal
from src.sim.backtest import backtest_tickers
from src.sim.engine import SimConfig, collect_signals, default_view_factory, simulate
from src.sim.execution import ExecutionPrices
from src.sim.impact_train import val_a_range
from src.sim.metrics import daily_returns, sharpe
from src.sim.risk import RiskConfig, RiskManager
from src.strategies.base import backtest_config, make_signal, strategy_params
from src.strategies.signal_stack import AggregateEqual, EventStrategy, ImpactStrategy, model_dir

STACKED = {s.name: s for s in (EventStrategy, ImpactStrategy, AggregateEqual)}
RAW = -2.0  # an entry threshold below every score: the strategy returns its raw score


def period_range(which: str, db_path: Path | None = None) -> tuple[date, date]:
    if which == "train":
        (p,) = load_periods(["train"])
        return p.start, p.end
    if which == "val_a":
        return val_a_range(db_path)
    raise ValueError(f"selection period {which!r} must be train or val_a (R3)")


class Replay:
    """Replays precomputed raw scores with a given hold period and entry threshold."""

    def __init__(
        self, name: str, raw: dict[tuple[str, date], float], hold: int, thr: float
    ) -> None:
        self.name, self.raw, self.thr = name, raw, thr
        self.horizon_days = self.min_hold_days = hold

    def signal(self, ticker: str, view: MarketView) -> Signal:
        s = self.raw.get((ticker, view.as_of.date()), 0.0)
        return make_signal(self.name, ticker, view, s if s > self.thr else min(s, 0.0))


def raw_scores(
    name: str, tickers: list[str], sessions: list[date], db_path: Path | None = None
) -> dict[tuple[str, date], float]:
    strat = STACKED[name](hold_days=1, entry_threshold=RAW)
    vf = default_view_factory(db_path)
    out = {}
    for d in sessions:
        for t, sig in collect_signals(strat, tickers, d, vf).items():
            out[(t, d)] = sig.score
    return out


def grid(
    name: str, raw: dict, tickers: list[str], prices: ExecutionPrices, risk: dict[str, Any]
) -> list[dict[str, Any]]:
    sel = strategy_params("selection")
    cfg = SimConfig.from_risk(risk)
    rows = []
    for hold in sel["hold_days"]:
        for thr in sel["entry_threshold"]:
            rm = RiskManager(RiskConfig.from_risk(risk))
            res = simulate(Replay(name, raw, int(hold), float(thr)), tickers, prices, cfg, risk=rm)
            dpy = int(backtest_config()["trading_days_per_year"])
            sh = sharpe(daily_returns(res.equity, cfg.initial_capital), dpy)
            rows.append({
                "hold_days": int(hold), "entry_threshold": float(thr),
                "sharpe": None if sh is None else round(sh, 4),
                "total_return": round(res.equity[-1] / cfg.initial_capital - 1, 4),
                "round_trips": len(res.round_trip_pnls), "costs_paid": round(res.costs_paid, 2),
            })  # fmt: skip
    return rows


def choose(rows: list[dict[str, Any]], min_trades: int) -> tuple[dict[str, Any], bool]:
    ok = [r for r in rows if r["round_trips"] >= min_trades and r["sharpe"] is not None]
    pool = ok or [r for r in rows if r["sharpe"] is not None] or rows
    best = max(pool, key=lambda r: (r["sharpe"] or 0.0, -r["round_trips"]))
    return best, bool(ok)


def run(db_path: Path | None = None) -> dict[str, Any]:
    sel = strategy_params("selection")
    risk = load_yaml("risk.yaml")
    tickers, benchmark = backtest_tickers("watchlist")
    out: dict[str, Any] = {}
    for name, which in sel["period"].items():
        start, end = period_range(which, db_path)
        prices = ExecutionPrices.load(tickers, start, end, benchmark, db_path)
        raw = raw_scores(name, tickers, prices.sessions, db_path)
        rows = grid(name, raw, tickers, prices, risk)
        best, eligible = choose(rows, int(sel["min_trades"]))
        out[name] = {
            "period": which, "start": start.isoformat(), "end": end.isoformat(),
            "chosen": {"hold_days": best["hold_days"], "entry_threshold": best["entry_threshold"]},
            "chosen_row": best, "eligible": eligible, "grid": rows,
        }  # fmt: skip
        print(f"{name} [{which}]: {best} eligible={eligible}")
    path = model_dir() / "selected.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run()
