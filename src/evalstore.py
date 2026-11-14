"""Read-only access to evals/results/*.json for GET /api/v1/evals (AGENTS.md section 5).

The files are written by `make eval` / `make backtest` / `make arena`; this module never runs an
eval. It only lifts two facts the files already contain into typed fields so a client does not have
to know each component's layout: the 95 % CI of the headline value (where stored, paths in
config/api.yaml) and the in-sample flag (backtest `details.in_sample`, or `src.sim.run.FITTED_ON`
for records that predate it; arena headline row `validation_in_sample`; D-47).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.schemas import EvalResult

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"
API_CONFIG = ROOT / "config" / "api.yaml"


def _dig(obj: Any, path: list[str]) -> Any:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def _pair(x: Any) -> tuple[float, float] | None:
    ok = isinstance(x, list | tuple) and len(x) == 2 and all(isinstance(v, int | float) for v in x)
    return (float(x[0]), float(x[1])) if ok else None


def ci_of(rec: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float] | None:
    details = rec.get("details")
    if rec["component"].startswith("backtest_"):
        return _pair(_dig(details, [cfg["eval_ci_backtest_key"]]))
    path = cfg["eval_ci_paths"].get(rec["component"])
    return _pair(_dig(details, path)) if path else None


def in_sample_of(rec: dict[str, Any]) -> str | None:
    """The reason the value is (partly) in-sample, as the record itself states it."""
    details = rec.get("details") or {}
    if isinstance(details.get("in_sample"), str):  # backtest records (src/sim/run.py)
        return details["in_sample"]
    if rec["component"].startswith("backtest_") and "in_sample" not in details:
        # records written before L-11 lack the key; the repo's own mapping is the authority
        from src.sim.run import in_sample_note

        return in_sample_note(details.get("strategy", ""), details.get("split", ""))
    if rec["component"].startswith("arena"):  # headline row of the arena leaderboard
        head = rec["metric"].removeprefix("deflated_sharpe_")
        for row in details.get("leaderboard", []):
            if row.get("strategy") == head and row.get("validation_in_sample"):
                return row.get("choice_period") or "chosen on part of the reported split"
    return None


def load_results(component: str | None = None, results_dir: Path | None = None) -> list[EvalResult]:
    """Every eval record, sorted by component; `component` selects one exactly."""
    cfg = yaml.safe_load(API_CONFIG.read_text(encoding="utf-8"))
    out = []
    for p in sorted((results_dir or RESULTS_DIR).glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        if component is not None and raw.get("component") != component:
            continue
        note = in_sample_of(raw)
        out.append(
            EvalResult.model_validate(
                {
                    **raw,
                    "ci95": ci_of(raw, cfg),
                    "in_sample": note is not None,
                    "in_sample_note": note,
                }
            )
        )
    return out


class NoStoredResult(LookupError):
    """No stored arena record for the requested split (run `make arena`)."""


def stored_arena(split: str, results_dir: Path | None = None) -> dict[str, Any]:
    """The leaderboard `make arena` wrote for `split`, as a job result. Nothing is recomputed."""
    name = "arena" if split == "validation" else f"arena_{split}"
    found = load_results(name, results_dir)
    if not found:
        raise NoStoredResult(f"no stored arena result for split {split!r}: run `make arena`")
    rec = found[0]
    return {
        "recomputed": False,
        "source": f"evals/results/{rec.component}.json",
        "run_at": rec.run_at.isoformat(),
        "split": split,
        "headline": {"metric": rec.metric, "value": rec.value, "target": rec.target},
        "leaderboard": (rec.details or {}).get("leaderboard", []),
        "in_sample": rec.in_sample,
    }
