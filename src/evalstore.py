"""Read-only access to evals/results/*.json for GET /api/v1/evals (AGENTS.md section 5).

The files are written by `make eval` / `make backtest`; this module never runs an eval.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas import EvalResult

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "results"


def load_results(component: str | None = None, results_dir: Path | None = None) -> list[EvalResult]:
    """Every eval record, sorted by component; `component` selects one exactly."""
    out = []
    for p in sorted((results_dir or RESULTS_DIR).glob("*.json")):
        rec = EvalResult.model_validate(json.loads(p.read_text(encoding="utf-8")))
        if component is None or rec.component == component:
            out.append(rec)
    return out
