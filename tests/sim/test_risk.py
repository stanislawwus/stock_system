"""Risk manager: unit rules and engine integration on planted scenarios (D-40)."""

from __future__ import annotations

from datetime import date

import pytest

from evals.eval_risk import DAYS, RISK, evaluate, scenarios
from evals.sim_fixtures import Scripted, market
from src.collectors.common import load_yaml
from src.sim.engine import SimConfig, simulate
from src.sim.risk import RiskConfig, RiskManager


def test_config_has_every_rule() -> None:
    cfg = RiskConfig.from_risk(load_yaml("risk.yaml"))
    assert 0 < cfg.max_position_pct <= 1 and cfg.stop_loss_pct > 0
    assert cfg.max_drawdown_halt_pct > 0 and cfg.cooldown_days >= 0


def test_drawdown_halt_uses_running_peak() -> None:
    rm = RiskManager(RISK)
    rm.mark(100.0)
    rm.mark(120.0)
    rm.mark(103.0)  # 14.2 % below the peak of 120
    assert not rm.halted()
    rm.mark(102.0)  # 15 %
    assert rm.halted()
    assert rm.veto_entry("X", 0) == "drawdown_halt"


def test_cooldown_only_after_a_loss_and_expires() -> None:
    rm = RiskManager(RISK)
    rm.record_exit("A", 3, pnl=+5.0)
    assert rm.veto_entry("A", 4) is None
    rm.record_exit("A", 3, pnl=-5.0)
    assert rm.veto_entry("A", 3 + RISK.cooldown_days) == "cooldown"
    assert rm.veto_entry("A", 4 + RISK.cooldown_days) is None
    assert rm.veto_entry("B", 4) is None


def test_stop_and_cap_arithmetic() -> None:
    rm = RiskManager(RiskConfig(0.2, 0.08, 0.15, 5))
    assert rm.stop_hit(100.0, 92.0) and not rm.stop_hit(100.0, 92.01)
    assert rm.cap_entry(50_000.0, 100_000.0) == pytest.approx(20_000.0)


@pytest.mark.parametrize("sc", scenarios(), ids=lambda s: s.name)
def test_planted_violation_is_real_and_caught(sc) -> None:
    assert not sc.holds(sc.run(with_risk=False)), "scenario does not plant a violation"
    assert sc.holds(sc.run(with_risk=True))


def test_eval_record() -> None:
    rec = evaluate()
    assert rec["value"] == 1.0 and rec["details"]["violations_after_veto"] == 0


def test_min_hold_blocks_score_exit() -> None:
    """A position entered on day 1 with min_hold 2 survives score 0 on days 2-3, exits on day 4."""
    strat = Scripted({DAYS[0]: {"A": 1.0}})
    strat.min_hold_days = 2  # type: ignore[attr-defined]
    prices = market({"A": [(100.0, 100.0)] * len(DAYS)}, DAYS)
    res = simulate(strat, ["A"], prices, SimConfig(100_000.0, 1.0, 0.0))  # type: ignore[arg-type]
    sells = [t for t in res.trades if t.side == "sell"]
    assert [t.day for t in sells] == [date(2026, 6, 4)] and sells[0].reason == "score<=0"


def test_engine_without_risk_is_unchanged() -> None:
    """risk=None keeps the old behaviour: the falling stock is held to the end."""
    sc = scenarios()[0]
    res = sc.run(with_risk=False)
    assert [t.reason for t in res.trades if t.side == "sell"] == ["end_of_split"]
