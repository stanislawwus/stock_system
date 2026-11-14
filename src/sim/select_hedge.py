"""Choose Hedge's eta and the confidence threshold p_min on val_a only (D-44, R3).

The component scores (sentiment, event, impact and the calibrated p) of every (ticker, session)
of val_a are computed once with `aggregate_equal`; every grid point of `aggregator.hedge_grid`
then replays them through `AggregateHedge` (weights from realised drift_ar out of the view's
bars, walk-forward) in the simulator with costs and the risk manager, with the hold period and
entry threshold already chosen for `aggregate_equal`. Best Sharpe among points with at least
`selection.min_trades` round trips wins (same rule as `src/sim/select.py`). eta = 0 with
p_min = null is the fixed-equal-weight baseline and is part of the grid, so "no Hedge, no
threshold" can win. Writes `selected.json["aggregate_hedge"]` with the whole grid; val_b is never
looked at here. Run by `make train` after `src.sim.select`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.collectors.common import load_yaml
from src.sim.backtest import backtest_tickers
from src.sim.engine import SimConfig, collect_signals, default_view_factory, simulate
from src.sim.execution import ExecutionPrices
from src.sim.metrics import daily_returns, sharpe
from src.sim.risk import RiskConfig, RiskManager
from src.sim.select import choose, period_range
from src.strategies.base import backtest_config, strategy_params
from src.strategies.hedge import AggregateHedge
from src.strategies.signal_stack import AggregateEqual, model_dir, selected_params

PERIOD = "val_a"


def component_scores(
    tickers: list[str], sessions: list[date], db_path: Path | None = None
) -> dict[tuple[str, date], dict[str, float]]:
    agg = AggregateEqual(hold_days=1, entry_threshold=-2.0)
    agg.parts_log = {}
    vf = default_view_factory(db_path)
    for d in sessions:
        collect_signals(agg, tickers, d, vf)
    return agg.parts_log


def grid(
    parts: dict, tickers: list[str], prices: ExecutionPrices, db_path: Path | None = None
) -> list[dict[str, Any]]:
    g = strategy_params("aggregator")["hedge_grid"]
    base = selected_params("aggregate_equal")
    lookback = int(strategy_params("aggregator")["lookback_days"])
    risk = load_yaml("risk.yaml")
    cfg = SimConfig.from_risk(risk)
    dpy = int(backtest_config()["trading_days_per_year"])
    rows = []
    for eta in g["eta"]:
        for p_min in g["p_min"]:
            strat = AggregateHedge(int(base["hold_days"]), float(base["entry_threshold"]),
                                   float(eta), p_min, lookback, parts_source=parts)  # fmt: skip
            rm = RiskManager(RiskConfig.from_risk(risk))
            res = simulate(strat, tickers, prices, cfg, default_view_factory(db_path), risk=rm)
            sh = sharpe(daily_returns(res.equity, cfg.initial_capital), dpy)
            rows.append({
                "eta": float(eta), "p_min": p_min,
                "sharpe": None if sh is None else round(sh, 4),
                "total_return": round(res.equity[-1] / cfg.initial_capital - 1, 4),
                "round_trips": len(res.round_trip_pnls), "costs_paid": round(res.costs_paid, 2),
                "final_weights": {k: round(v, 3) for k, v in strat.last_weights.items()},
            })  # fmt: skip
            print(f"eta={eta} p_min={p_min}: {rows[-1]['sharpe']} ({rows[-1]['round_trips']})")
    return rows


def run(db_path: Path | None = None) -> dict[str, Any]:
    tickers, benchmark = backtest_tickers("watchlist")
    start, end = period_range(PERIOD, db_path)
    prices = ExecutionPrices.load(tickers, start, end, benchmark, db_path)
    parts = component_scores(tickers, prices.sessions, db_path)
    rows = grid(parts, tickers, prices, db_path)
    best, eligible = choose(rows, int(strategy_params("selection")["min_trades"]))
    equal = next(r for r in rows if r["eta"] == 0.0 and r["p_min"] is None)
    entry = {
        "period": PERIOD, "start": start.isoformat(), "end": end.isoformat(),
        "chosen": {"eta": best["eta"], "p_min": best["p_min"]}, "chosen_row": best,
        "eligible": eligible, "equal_weights_row": equal, "grid": rows,
    }  # fmt: skip
    path = model_dir() / "selected.json"
    sel = json.loads(path.read_text(encoding="utf-8"))
    sel[AggregateHedge.name] = entry
    path.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"{AggregateHedge.name} [{PERIOD}]: {best} eligible={eligible}; equal weights {equal}")
    return entry


if __name__ == "__main__":
    run()
