"""Risk manager (AGENTS.md R8; config/risk.yaml; D-19, D-40).

Rules, all read from `config/risk.yaml` (no thresholds in code):

- **position cap**: an entry may not exceed `max_position_pct` of equity (the engine sizes to it;
  `cap_entry` is the veto that guarantees it);
- **stop-loss**: a position whose close is at or below `entry price * (1 - stop_loss_pct)` is sold
  at the next open (the stop is only *observed* at a close, so it fills at the following open);
- **drawdown halt (kill switch)**: no new positions while equity at the last close is at least
  `max_drawdown_halt_pct` below its running peak; existing positions keep their exits;
- **cooldown**: after a losing round trip in a ticker, no re-entry in it for `cooldown_days`
  sessions.

The manager holds state for one simulation run and never reads prices itself: the engine passes
the marks it already has, so the manager cannot see anything the engine does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float
    stop_loss_pct: float
    max_drawdown_halt_pct: float
    cooldown_days: int

    @classmethod
    def from_risk(cls, risk: dict[str, Any]) -> RiskConfig:
        return cls(
            max_position_pct=float(risk["max_position_pct"]),
            stop_loss_pct=float(risk["stop_loss_pct"]),
            max_drawdown_halt_pct=float(risk["max_drawdown_halt_pct"]),
            cooldown_days=int(risk["cooldown_days"]),
        )


@dataclass
class RiskManager:
    cfg: RiskConfig
    peak: float | None = None
    last_equity: float | None = None
    last_loss_index: dict[str, int] = field(default_factory=dict)
    vetoes: list[tuple[int, str, str]] = field(default_factory=list)  # (session, ticker, reason)

    # --- state updates ---------------------------------------------------------------------
    def mark(self, equity: float) -> None:
        """Equity at a close: updates the running peak used by the drawdown halt."""
        self.last_equity = equity
        self.peak = equity if self.peak is None else max(self.peak, equity)

    def record_exit(self, ticker: str, session: int, pnl: float) -> None:
        if pnl < 0:
            self.last_loss_index[ticker] = session

    # --- checks ----------------------------------------------------------------------------
    def drawdown(self) -> float:
        if not self.peak or self.last_equity is None:
            return 0.0
        return 1.0 - self.last_equity / self.peak

    def halted(self) -> bool:
        return self.drawdown() >= self.cfg.max_drawdown_halt_pct

    def veto_entry(self, ticker: str, session: int) -> str | None:
        """Reason to refuse a new position in `ticker` on `session`, or None."""
        reason = None
        if self.halted():
            reason = "drawdown_halt"
        elif (
            ticker in self.last_loss_index
            and session - self.last_loss_index[ticker] <= self.cfg.cooldown_days
        ):
            reason = "cooldown"
        if reason:
            self.vetoes.append((session, ticker, reason))
        return reason

    def cap_entry(self, budget: float, equity: float) -> float:
        """The largest allowed entry notional: never above the position cap."""
        return min(budget, self.cfg.max_position_pct * equity)

    def stop_hit(self, entry_price: float, close: float) -> bool:
        return close <= entry_price * (1.0 - self.cfg.stop_loss_pct)
