"""In-sample backtests are flagged and never shown as a pass (L-11)."""

from __future__ import annotations

from evals.run_all import table
from src.sim.run import in_sample_note


def test_fitted_splits_are_in_sample_and_rule_strategies_are_not() -> None:
    assert in_sample_note("impact", "train")
    assert in_sample_note("aggregate_equal", "validation")
    assert in_sample_note("momentum", "train") is None
    assert in_sample_note("event", "validation") is None  # event was fitted on train only


def _rec(component: str, value: float, details: dict | None = None) -> dict:
    return {
        "component": component,
        "metric": "sharpe",
        "value": value,
        "target": 1.76,
        "gold_set": "g",
        "n": 124,
        "model": "rules",
        "git_sha": "x",
        "details": details,
    }


def test_the_table_shows_in_sample_instead_of_a_pass() -> None:
    rows = table(
        [
            _rec("backtest_impact_train", 7.19),  # an old record without the flag
            _rec("backtest_momentum_train", 1.95),
            _rec("backtest_x_train", 9.0, {"in_sample": "fitted on train"}),
        ]
    ).splitlines()[2:]
    assert "| in-sample |" in rows[0] and "| yes |" not in rows[0]
    assert "| yes |" in rows[1]
    assert "| in-sample |" in rows[2]
