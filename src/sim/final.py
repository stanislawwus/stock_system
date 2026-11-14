"""The single test-period run behind `make test-final` (AGENTS.md R3; L-01, D-43).

Both `--final` entry points (`python -m src.strategies.arena --final` and
`python -m src.sim.run --final`) call `run_final()`; `make test-final` calls one of them once.

Order:
1. `preflight()`: the trained artifacts (`data/models`, `make train`) and the test sentiment
   parquet with a complete manifest (`make features-test`, D-38) exist. Only file existence is
   checked; no test-period row is read. A failure here leaves the period unused.
2. `splits.final_run()` writes `evals/results/TEST_FINAL.lock` **before** anything is read
   (exclusive create), so a crash, a retry or a second process can never read the period twice.
3. The arena on the test split (`src/sim/arena.py`: leaderboard with CIs, correlation, ablation,
   random controls, deflated Sharpe against the trials logged during development; the log is
   not extended) and a `backtest_<strategy>_test` record per strategy from the same simulations.
   Sentiment of test days comes from the parquet through `MarketView` (D-38); models, the item
   table and the selected hold/threshold from `data/models`.
Nothing is tuned here or afterwards. Output: `evals/results/arena_test.json`,
`evals/results/backtest_*_test.json`, `docs/06_final_run.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import splits
from src.schemas import DISCLAIMER
from src.sim import arena
from src.sim import run as sim_run
from src.sim.backtest import Run
from src.strategies.registry import BENCHMARK
from src.strategies.signal_stack import model_dir

MODEL_FILES = ("impact.joblib", "event_items.json", "selected.json")


class FinalRunNotReadyError(RuntimeError):
    """A prerequisite of the final run is missing; the test period was not touched."""


def preflight() -> None:
    if splits.FINAL_LOCK.exists():
        raise splits.TestPeriodLockedError(f"{splits.FINAL_LOCK.name} exists: already used once")
    missing = [f for f in MODEL_FILES if not (model_dir() / f).exists()]
    if missing:
        raise FinalRunNotReadyError(f"{missing} missing in {model_dir()}: run `make train` first")
    scores = splits.final_scores_path()
    manifest = scores.with_suffix(".manifest.json")
    if not scores.exists() or not manifest.exists():
        raise FinalRunNotReadyError(f"{scores} missing: run `make features-test` first (D-38)")
    if not json.loads(manifest.read_text(encoding="utf-8")).get("complete"):
        raise FinalRunNotReadyError(f"{manifest} is not complete: rerun `make features-test`")


def run_final(db_path: Path | None = None) -> list[Path]:
    preflight()
    written: list[Path] = []
    with splits.final_run():  # the lock exists from here on, whatever happens next
        runs: dict[str, Run] = {}
        rec = arena.record(arena.run_arena(splits.FINAL_SPLIT, db_path, runs_out=runs))
        written += arena.write(rec)
        bench = runs[BENCHMARK].result.sharpe or 0.0
        for name, r in runs.items():
            brec = sim_run.record(r.result, 0.0 if name == BENCHMARK else bench)
            written.append(sim_run.write(brec))
    from evals.run_all import refresh_doc  # late import: evals is a consumer of src

    refresh_doc()
    for p in written:
        print(f"final run -> {p}")
    print(f"Commit {splits.FINAL_LOCK.name}, evals/results and docs now.")
    print(DISCLAIMER)
    return written
