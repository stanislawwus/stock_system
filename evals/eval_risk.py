"""Risk-manager eval (AGENTS.md R4; docs/03_eval_plan.md row "Risk manager"; D-40).

- metric: `planted_caught_share`, the share of planted scenarios where (a) without the risk
  manager the rule is violated (the plant is real) and (b) with it the rule holds.
- gold set: derived, `scenarios()` below: four tiny hand-built markets (stop-loss, drawdown halt,
  cooldown after a loss, position cap with a sizing bug planted in the simulator config).
- target: 1.0 (and 0 violations after veto, reported in details).
- why: the risk layer is a safety property; one missed rule is a bug, so the score is binary.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from evals.sim_fixtures import Scripted, market
from src.collectors.common import ROOT
from src.sim.engine import SimConfig, SimResult, simulate
from src.sim.risk import RiskConfig, RiskManager

# eight validation sessions without holidays (Mon 2026-06-01 .. Wed 2026-06-10)
DAYS = [date(2026, 6, d) for d in (1, 2, 3, 4, 5, 8, 9, 10)]
CAPITAL = 100_000.0
RISK = RiskConfig(
    max_position_pct=1.0, stop_loss_pct=0.08, max_drawdown_halt_pct=0.15, cooldown_days=5
)


@dataclass(frozen=True)
class Scenario:
    name: str
    strategy: Scripted
    tickers: list[str]
    bars: dict[str, list[tuple[float, float]]]
    sim: SimConfig
    risk: RiskConfig
    holds: Callable[[SimResult], bool]  # True when the rule is respected

    def run(self, with_risk: bool) -> SimResult:
        rm = RiskManager(self.risk) if with_risk else None
        return simulate(self.strategy, self.tickers, market(self.bars, DAYS), self.sim, risk=rm)


def _flat(px: float) -> list[tuple[float, float]]:
    return [(px, px)] * len(DAYS)


def _buys(res: SimResult, ticker: str) -> list[date]:
    return [t.day for t in res.trades if t.side == "buy" and t.ticker == ticker]


def scenarios() -> list[Scenario]:
    full = SimConfig(CAPITAL, max_position_pct=1.0, cost_per_side=0.0)
    every_day = {d: {"A": 1.0} for d in DAYS}

    # 1. stop-loss: A closes day 1 at -10 % (stop 8 %) and keeps falling; the stop sells at the
    #    open of day 2, so no close after day 1 is held.
    falling = [(100.0, 90.0), (89.0, 85.0)] + [(80.0, 80.0)] * 6
    stop = Scenario(
        "stop_loss", Scripted(every_day), ["A"], {"A": falling}, full, RISK,
        lambda r: any(t.reason == "stop_loss" and t.day == DAYS[1] for t in r.trades),
    )  # fmt: skip

    # 2. drawdown halt: A bought day 1 at 100 with all equity, sold at the close of day 2 at 80
    #    (-20 % > 15 % halt); B signals from day 3 on and must not be bought.
    halt = Scenario(
        "drawdown_halt",
        Scripted({DAYS[0]: {"A": 1.0}, DAYS[1]: {"A": 1.0}, **{d: {"B": 1.0} for d in DAYS[2:]}},
                 horizon_days=1),
        ["A", "B"], {"A": [(100.0, 95.0), (90.0, 80.0)] + [(80.0, 80.0)] * 6, "B": _flat(50.0)},
        full, RISK, lambda r: not _buys(r, "B"),
    )  # fmt: skip

    # 3. cooldown: A loses 2 % on its first round trip (closed day 2, session 1); signals again on
    #    day 3; re-entry allowed only from session 1 + 5 + 1 = 7 (day 8 of the list).
    small_loss = [(100.0, 99.0), (99.0, 98.0)] + [(98.0, 98.0)] * 6
    cool = Scenario(
        "cooldown", Scripted(every_day, horizon_days=1), ["A"], {"A": small_loss}, full, RISK,
        lambda r: _buys(r, "A") == [DAYS[0], DAYS[7]],
    )  # fmt: skip

    # 4. position cap: the simulator config is planted with a 100 % cap; the risk cap is 20 %.
    capped = RiskConfig(0.20, RISK.stop_loss_pct, RISK.max_drawdown_halt_pct, RISK.cooldown_days)
    cap = Scenario(
        "position_cap", Scripted({DAYS[0]: {"A": 1.0}}), ["A"], {"A": _flat(100.0)}, full, capped,
        lambda r: all(t.qty * t.price <= 0.2 * CAPITAL + 1e-6 for t in r.trades if t.side == "buy"),
    )  # fmt: skip
    return [stop, halt, cool, cap]


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def evaluate() -> dict[str, Any]:
    details: dict[str, Any] = {}
    for sc in scenarios():
        planted = not sc.holds(sc.run(with_risk=False))
        held = sc.holds(sc.run(with_risk=True))
        details[sc.name] = {"violation_without_risk": planted, "rule_holds_with_risk": held}
    caught = sum(
        d["violation_without_risk"] and d["rule_holds_with_risk"] for d in details.values()
    )
    violations = sum(not d["rule_holds_with_risk"] for d in details.values())
    value = caught / len(details)
    details["violations_after_veto"] = violations
    return {
        "component": "risk",
        "metric": "planted_caught_share",
        "value": value,
        "target": 1.0,
        "why_this_metric": "the risk layer is a safety property; one missed rule is a bug",
        "gold_set": "derived: evals/eval_risk.py scenarios() (4 planted violations)",
        "n": len(details) - 1,
        "split": "n/a",
        "model": "rules",
        "git_sha": _git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": "" if value >= 1.0 else "a planted violation was not caught: see details",
        "details": details,
    }
