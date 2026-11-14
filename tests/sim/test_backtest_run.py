"""Backtest service and the `make backtest` CLI (src/sim/backtest.py, src/sim/run.py)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.collectors.common import load_yaml
from src.schemas import DISCLAIMER, EvalResult
from src.sim import backtest, run


@pytest.fixture()
def fixture_db(price_db: Callable[..., Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(backtest, "backtest_tickers", lambda universe: (["A", "B"], "SPY"))
    run._benchmark.cache_clear()
    return price_db()


def test_backtest_on_validation_is_consistent(fixture_db: Path) -> None:
    res = backtest.run_backtest("buy_and_hold", "val", fixture_db)
    assert res.split == "validation" and res.tickers == ["A", "B"]
    assert res.start.isoformat() == "2026-06-01" and res.end.isoformat() == "2026-07-31"
    assert res.n_days == len(res.equity_curve) == 45
    assert res.n_trades == 4  # two buys on the first day, two sells at the last close
    assert res.final_equity == pytest.approx(res.initial_capital * (1 + res.total_return))
    assert 0 <= res.max_drawdown < 1
    # two names at the 20 % cap: 40 % invested, bought once and sold once -> one-way turnover ~0.4
    cap = load_yaml("risk.yaml")["max_position_pct"]
    assert res.turnover == pytest.approx(2 * cap, abs=0.05)
    lo, hi = res.sharpe_ci95
    assert lo <= hi


@pytest.mark.parametrize("strategy", ["buy_and_hold", "momentum", "mean_reversion"])
def test_eval_record_matches_the_schema(fixture_db: Path, strategy: str) -> None:
    rec = run.evaluate(strategy, "validation", fixture_db)
    EvalResult(**rec)  # AGENTS.md section 5 fields
    assert rec["component"] == f"backtest_{strategy}_validation"
    assert rec["metric"] == "sharpe" and rec["n"] == 45
    bench = run.evaluate("buy_and_hold", "validation", fixture_db)
    assert rec["target"] == (0.0 if strategy == "buy_and_hold" else bench["value"])
    assert rec["details"]["disclaimer"] == DISCLAIMER
    assert rec["notes"]


def test_cli_refuses_the_test_split(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        run.main(["--split", "test"])
    assert "invalid choice" in capsys.readouterr().err
