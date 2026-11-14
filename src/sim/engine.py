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
) -> SimResult:
    view_factory = view_factory or default_view_factory()
    c = cfg.cost_per_side
    cash = cfg.initial_capital
    pos: dict[str, Position] = {}
    last_px: dict[str, float] = {}
    res = SimResult(strategy.name, [], [], [], [], 0.0, 0.0)

    def sell(t: str, day: date, px: float, at_open: bool, reason: str) -> None:
        nonlocal cash
        p = pos.pop(t)
        gross = p.qty * px
        fee = gross * c
        cash += gross - fee
        res.traded_notional += gross
        res.costs_paid += fee
        res.round_trip_pnls.append(gross - fee - p.entry_cost)
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
            if signals[t].score <= 0 and px is not None:
                sell(t, day, px, at_open=True, reason="score<=0")
        entries = [
            t for t in tickers
            if t not in pos and signals[t].score > cfg.entry_threshold
            and prices.open_price(t, day) is not None
        ]  # fmt: skip
        n_target = len(pos) + len(entries)
        if entries:
            marks = {t: prices.open_price(t, day) or last_px[t] for t in pos}
            equity_open = cash + sum(p.qty * marks[t] for t, p in pos.items())
            weight = min(1.0 / n_target, cfg.max_position_pct)
            for t in entries:
                budget = min(weight * equity_open, cash)
                if budget <= MIN_TRADE_USD:
                    continue
                px = prices.open_price(t, day)
                assert px is not None
                qty = budget / (px * (1.0 + c))
                fee = qty * px * c
                cash -= budget
                res.traded_notional += qty * px
                res.costs_paid += fee
                exit_index = None if strategy.horizon_days is None else i + strategy.horizon_days
                pos[t] = Position(qty, day, budget, exit_index)
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
                sell(t, day, px, at_open=False, reason="end_of_split")
            elif p.exit_index is not None and p.exit_index <= i:
                sell(t, day, px, at_open=False, reason=f"horizon_{strategy.horizon_days}d")
        invested = sum(p.qty * last_px[t] for t, p in pos.items())
        equity = cash + invested
        res.days.append(day)
        res.equity.append(equity)
        res.exposure.append(invested / equity if equity > 0 else 0.0)
    return res
