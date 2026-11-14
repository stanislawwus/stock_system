"""Daily long-only simulator (AGENTS.md R2, R8; decisions D-03, D-08, D-35).

For every session D of the split, in this order:

1. **Decide** at `as_of = 09:00 ET of D`: each ticker's signal comes from
   `strategy.signal(ticker, MarketView(as_of))`, called inside `src.db.guarded()`, so the strategy
   can reach data only through the view (newest visible bar = close of D-1).
2. **Open of D**: held positions whose score is <= 0 are sold (cash). Tickers with a score above
   the entry threshold that are not held are bought, equal weight: each gets
   `min(1 / N, max_position_pct)` of the equity marked at the open, where N is the number of
   tickers that will be held; existing positions are not resized. No leverage: a buy never spends
   more cash than there is. Costs: `cost_per_side` of the traded notional on every buy and sell.
3. **Close of D**: positions opened on session D-h are sold at the close (time exit after h
   sessions; h = None means no time exit). On the last session everything left is sold at the
   close, so every trade is a closed round trip and the final equity is cash.
4. Equity is marked at the close of D (last known close if a ticker has no bar on D).

Optional (D-42): a strategy may define `horizon_for(ticker, day) -> int | None`, the time exit of
a position entered on `day` (overrides `horizon_days`; used by the arena's random controls).

Optional (D-40): a strategy's `min_hold_days` (default 0) blocks score exits at the open while
`session <= entry session + min_hold_days`; a `RiskManager` (src/sim/risk.py) vetoes entries
(drawdown halt, cooldown after a loss), caps each entry at `max_position_pct`, and sells at the
next open any position whose close hit the stop-loss. Without one the engine behaves as before.

Fills: buy qty = budget / (open * (1 + c)), so cash out = budget exactly; sell proceeds =
qty * price * (1 - c). A ticker without a bar on D cannot be traded on D.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from src.db import guarded
from src.marketview import MarketView
from src.marketview.clock import MARKET_TZ, OPEN_LOCAL, bar_close_for, cutoff_for
from src.schemas import Signal, Trade
from src.sim.execution import ExecutionPrices
from src.sim.risk import RiskManager
from src.strategies.base import Strategy

MIN_TRADE_USD = 1e-6  # below this a buy is a rounding artefact, not a trade

ViewFactory = Callable[[datetime], MarketView]


@dataclass(frozen=True)
class SimConfig:
    initial_capital: float
    max_position_pct: float
    cost_per_side: float
    entry_threshold: float = 0.0  # enter only when score > this; exit when score <= 0

    @classmethod
    def from_risk(cls, risk: dict) -> SimConfig:
        """config/risk.yaml: cost per side = commission + slippage (D-35)."""
        threshold = risk.get("confidence_threshold")
        return cls(
            initial_capital=float(risk["initial_capital_usd"]),
            max_position_pct=float(risk["max_position_pct"]),
            cost_per_side=float(risk["commission_pct_per_side"]) + float(risk["slippage_pct"]),
            entry_threshold=float(threshold) if threshold is not None else 0.0,
        )


@dataclass
class Position:
    qty: float
    entry_day: date
    entry_cost: float  # cash paid, costs included
    exit_index: int | None  # session index whose close ends the holding period
    entry_price: float = 0.0
    hold_until: int = -1  # score exits are blocked at opens of sessions <= this index


@dataclass
class SimResult:
    strategy: str
    days: list[date]
    equity: list[float]  # at the close of each day
    trades: list[Trade]
    round_trip_pnls: list[float]
    traded_notional: float
    costs_paid: float
    exposure: list[float] = field(default_factory=list)  # invested share of equity at the close


def _fill_time(day: date, at_open: bool) -> datetime:
    if at_open:
        return datetime.combine(day, OPEN_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)
    return bar_close_for(day)


def default_view_factory(db_path: Path | None = None) -> ViewFactory:
    return lambda as_of: MarketView(as_of, db_path)


def collect_signals(
    strategy: Strategy, tickers: list[str], day: date, view_factory: ViewFactory
) -> dict[str, Signal]:
    """Signals for D from a MarketView at the 09:00 ET cutoff of D; direct DB access raises."""
    as_of = cutoff_for(day)
    out: dict[str, Signal] = {}
    with guarded(), view_factory(as_of) as view:
        for t in tickers:
            s = strategy.signal(t, view)
            if s.ticker != t or s.as_of != as_of:
                raise ValueError(
                    f"{strategy.name}: signal for {s.ticker}@{s.as_of}, asked {t}@{as_of}"
                )
            out[t] = s
    return out


def simulate(
    strategy: Strategy,
    tickers: list[str],
    prices: ExecutionPrices,
    cfg: SimConfig,
    view_factory: ViewFactory | None = None,
    risk: RiskManager | None = None,
) -> SimResult:
    view_factory = view_factory or default_view_factory()
    min_hold = int(getattr(strategy, "min_hold_days", 0) or 0)
    horizon_for = getattr(strategy, "horizon_for", None)  # optional per-entry horizon
    stop_pending: set[str] = set()
    if risk is not None:
        risk.mark(cfg.initial_capital)  # the drawdown peak starts at the initial capital
    c = cfg.cost_per_side
    cash = cfg.initial_capital
    pos: dict[str, Position] = {}
    last_px: dict[str, float] = {}
    res = SimResult(strategy.name, [], [], [], [], 0.0, 0.0)

    def sell(t: str, i: int, px: float, at_open: bool, reason: str) -> None:
        nonlocal cash
        day = sessions[i]
        p = pos.pop(t)
        gross = p.qty * px
        fee = gross * c
        cash += gross - fee
        res.traded_notional += gross
        res.costs_paid += fee
        res.round_trip_pnls.append(gross - fee - p.entry_cost)
        if risk is not None:
            risk.record_exit(t, i, gross - fee - p.entry_cost)
        res.trades.append(
            Trade(available_at=_fill_time(day, at_open), ticker=t, day=day, side="sell",
                  qty=p.qty, price=px, commission=fee, strategy=strategy.name, reason=reason)
        )  # fmt: skip

    sessions = prices.sessions
    for i, day in enumerate(sessions):
        signals = collect_signals(strategy, tickers, day, view_factory)

        # --- open of D: exits on score <= 0, then equal-weight entries -----------------------
        for t in sorted(pos):
            px = prices.open_price(t, day)
            if px is None:
                continue
            if t in stop_pending:
                sell(t, i, px, at_open=True, reason="stop_loss")
            elif signals[t].score <= 0 and i > pos[t].hold_until:
                sell(t, i, px, at_open=True, reason="score<=0")
        stop_pending &= set(pos)  # a stop without an open today fills at the next one
        entries = [
            t for t in tickers
            if t not in pos and signals[t].score > cfg.entry_threshold
            and prices.open_price(t, day) is not None
            and (risk is None or risk.veto_entry(t, i) is None)
        ]  # fmt: skip
        n_target = len(pos) + len(entries)
        if entries:
            marks = {t: prices.open_price(t, day) or last_px[t] for t in pos}
            equity_open = cash + sum(p.qty * marks[t] for t, p in pos.items())
            weight = min(1.0 / n_target, cfg.max_position_pct)
            for t in entries:
                budget = min(weight * equity_open, cash)
                if risk is not None:
                    budget = risk.cap_entry(budget, equity_open)
                if budget <= MIN_TRADE_USD:
                    continue
                px = prices.open_price(t, day)
                assert px is not None
                qty = budget / (px * (1.0 + c))
                fee = qty * px * c
                cash -= budget
                res.traded_notional += qty * px
                res.costs_paid += fee
                h = horizon_for(t, day) if horizon_for else strategy.horizon_days
                exit_index = None if h is None else i + h
                pos[t] = Position(qty, day, budget, exit_index, px, i + min_hold)
                last_px[t] = px
                res.trades.append(
                    Trade(available_at=_fill_time(day, True), ticker=t, day=day, side="buy",
                          qty=qty, price=px, commission=fee, strategy=strategy.name,
                          reason=f"score={signals[t].score:.3f}")
                )  # fmt: skip

        # --- close of D: time exits, end of split, mark to market ----------------------------
        last = i == len(sessions) - 1
        for t in sorted(pos):
            px = prices.close_price(t, day)
            if px is None and not last:
                continue
            px = last_px[t] = px if px is not None else last_px[t]
            p = pos[t]
            if last:
                sell(t, i, px, at_open=False, reason="end_of_split")
            elif p.exit_index is not None and p.exit_index <= i:
                sell(t, i, px, at_open=False, reason=f"horizon_{strategy.horizon_days}d")
            elif risk is not None and risk.stop_hit(p.entry_price, px):
                stop_pending.add(t)
        invested = sum(p.qty * last_px[t] for t, p in pos.items())
        equity = cash + invested
        res.days.append(day)
        res.equity.append(equity)
        res.exposure.append(invested / equity if equity > 0 else 0.0)
        if risk is not None:
            risk.mark(equity)
    return res
