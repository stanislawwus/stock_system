"""`make backtest STRATEGY=<name|all> SPLIT=<train|val|validation>` (D-35).

Writes `evals/results/backtest_<strategy>_<split>.json` (AGENTS.md section 5 record) and refreshes
the table in docs/evals.md. Metric: annualised Sharpe after costs; target: the Sharpe of
buy-and-hold on the same split and tickers (0 for buy-and-hold itself). Only development splits:
the test period runs once, through `make test-final` (R3).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.collectors.common import ROOT
from src.schemas import DISCLAIMER, BacktestResult
from src.sim.backtest import DEV_SPLITS, resolve_split, run_backtest
from src.strategies.registry import BENCHMARK, BUILDERS

RESULTS = ROOT / "evals" / "results"
WHY = (
    "risk-adjusted return after 5 bps per side; the target is buy-and-hold on the same names and "
    "days, so a strategy has to beat simply holding what it trades"
)


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


@lru_cache(maxsize=4)
def _benchmark(split: str, db_path: Path | None) -> BacktestResult:
    return run_backtest(BENCHMARK, split, db_path)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def record(res: BacktestResult, target: float) -> dict[str, Any]:
    value = res.sharpe if res.sharpe is not None else 0.0
    ci = res.sharpe_ci95
    if ci is None:
        notes = "Sharpe undefined (no variance in daily returns); value set to 0."
    elif ci[0] <= target <= ci[1]:
        notes = (
            f"95% block-bootstrap CI of the Sharpe [{ci[0]:.2f}, {ci[1]:.2f}] over {res.n_days} "
            f"days contains the target {target:.2f}: not distinguishable from it."
        )
    else:
        side = "above" if ci[0] > target else "below"
        notes = f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}] lies entirely {side} the target {target:.2f}."
    details = res.model_dump(mode="json", exclude={"equity_curve"})
    details["equity_curve"] = [[d.isoformat(), round(e, 2)] for d, e in res.equity_curve]
    details["disclaimer"] = DISCLAIMER
    return {
        "component": f"backtest_{res.strategy}_{res.split}",
        "metric": "sharpe",
        "value": round(value, 4),
        "target": round(target, 4),
        "why_this_metric": WHY,
        "gold_set": f"derived: {BENCHMARK} on the same split and tickers (benchmark)",
        "n": res.n_days,
        "split": res.split,
        "model": "rules",
        "git_sha": _git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            k: _r(v) if isinstance(v, float) else v for k, v in details.items()
        },  # fmt: skip
    }


def evaluate(strategy: str, split: str, db_path: Path | None = None) -> dict[str, Any]:
    split = resolve_split(split)
    bench = _benchmark(split, db_path)
    if strategy == BENCHMARK:
        return record(bench, 0.0)
    target = bench.sharpe if bench.sharpe is not None else 0.0
    return record(run_backtest(strategy, split, db_path), target)


def write(rec: dict[str, Any]) -> Path:
    path = RESULTS / f"{rec['component']}.json"
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="all", choices=["all", *sorted(BUILDERS)])
    ap.add_argument("--split", default="validation", choices=sorted(DEV_SPLITS))
    ap.add_argument("--final", action="store_true", help="the single test-period run (D-43)")
    args = ap.parse_args(argv)
    if args.final:
        from src.sim.final import run_final  # late import: final imports this module

        run_final()
        return 0
    names = sorted(BUILDERS) if args.strategy == "all" else [args.strategy]
    for name in names:
        rec = evaluate(name, args.split)
        d = rec["details"]
        print(
            f"{rec['component']}: sharpe {rec['value']} (target {rec['target']}), "
            f"total {d['total_return']}, maxDD {d['max_drawdown']}, trades {d['n_trades']} "
            f"-> {write(rec).relative_to(ROOT)}"
        )
    from evals.run_all import refresh_doc  # late import: evals is a consumer of src

    refresh_doc()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
