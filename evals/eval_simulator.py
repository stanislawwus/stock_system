"""Simulator eval (AGENTS.md R4; docs/03_eval_plan.md row "Simulator").

- metric: `hand_fixture_exact_share`, the share of hand-worked fixtures whose final equity matches
  the hand computation to the cent and whose trade count matches.
- gold set: derived, `evals/sim_fixtures.py` (four fixtures: buy-and-hold toy, round-trip costs,
  score <= 0 exit at the open, horizon exit and re-entry); each also checks the identity
  final equity = cash after the end-of-split liquidation.
- target: 1.0.
- why: simulator correctness is binary; one wrong fill or cost rule biases every backtest number.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any

from evals.sim_fixtures import fixtures
from src.collectors.common import ROOT

CENT = 0.005


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def evaluate() -> dict[str, Any]:
    details = {}
    for fx in fixtures():
        res = fx.run()
        exact = abs(res.equity[-1] - fx.expected_final) <= CENT
        ok = exact and len(res.trades) == fx.expected_trades
        details[fx.name] = {
            "expected_final": round(fx.expected_final, 4),
            "simulated_final": round(res.equity[-1], 4),
            "trades": len(res.trades),
            "pass": ok,
        }
    passed = sum(d["pass"] for d in details.values())
    value = passed / len(details)
    return {
        "component": "simulator",
        "metric": "hand_fixture_exact_share",
        "value": value,
        "target": 1.0,
        "why_this_metric": "simulator correctness is binary; one wrong fill or cost rule biases "
        "every backtest number",
        "gold_set": "derived: evals/sim_fixtures.py (4 hand-worked fixtures)",
        "n": len(details),
        "split": "n/a",
        "model": "n/a",
        "git_sha": _git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": "" if value >= 1.0 else f"{len(details) - passed} fixture(s) differ: see details",
        "details": details,
    }
