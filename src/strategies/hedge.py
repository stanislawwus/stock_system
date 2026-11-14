"""Hedge aggregation with a confidence threshold (D-17, D-44): strategy `aggregate_hedge`.

Same components as `aggregate_equal` (sentiment, event, impact), weighted by multiplicative
weights (Hedge) that are updated every session on the *realised* `drift_ar`:

- `drift_ar` of a past session d' (open of d' -> close of d'+1, abnormal vs the benchmark, D-28)
  is computed from the view's own bars with `src.impact.labels.abnormal_returns`; at the cutoff
  of D the newest bar is D-1, so the newest realised label is d' = D-2. No label table, no
  execution prices (R2): only bars that were public before the cutoff.
- loss of component k on (ticker, d') = (1 - s_k * sign(drift_ar)) / 2 in [0, 1]: 0 for a full
  score on the right side, 1 on the wrong side, 0.5 for no opinion; averaged over the tickers
  seen that day.
- weights over the last `lookback_days` realised sessions: w_k ~ exp(-eta * sum of losses);
  eta = 0 is equal weights. Without any realised session yet the weights are equal.
- confidence threshold: the suggestion is cash unless the calibrated probability of the impact
  model is >= `p_min` (None = off), then the usual entry threshold on the score (R8).

The strategy walks forward: it remembers the component scores it computed on earlier days of
the same run (a fresh object starts with equal weights). eta and p_min are chosen on val_a
only (`src/sim/select_hedge.py`, R3); hold and entry threshold are those of `aggregate_equal`.
Signals are suggestions with a confidence, not predictions (R11).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from src.impact.labels import abnormal_returns
from src.marketview import MarketView
from src.schemas import Signal
from src.strategies.signal_stack import AggregateEqual, selected_params

Parts = dict[str, float]  # component -> score


def component_loss(score: float, drift_ar: float) -> float:
    """(1 - s * sign(drift_ar)) / 2, in [0, 1]."""
    sign = 1.0 if drift_ar > 0 else (-1.0 if drift_ar < 0 else 0.0)
    return (1.0 - max(-1.0, min(1.0, score)) * sign) / 2.0


def daily_losses(
    parts: Mapping[str, Parts], drifts: Mapping[str, float], components: Sequence[str]
) -> dict[str, float] | None:
    """Mean loss per component over the tickers that have both scores and a realised label."""
    names = [t for t in parts if t in drifts]
    if not names:
        return None
    return {
        k: sum(component_loss(parts[t][k], drifts[t]) for t in names) / len(names)
        for k in components
    }


def hedge_weights(
    losses: Sequence[Mapping[str, float]], components: Sequence[str], eta: float
) -> dict[str, float]:
    """Multiplicative weights from the cumulative losses (normalised, numerically stable)."""
    cum = {k: sum(row[k] for row in losses) for k in components}
    lo = min(cum.values()) if cum else 0.0
    raw = {k: math.exp(-eta * (cum[k] - lo)) for k in components}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


class AggregateHedge(AggregateEqual):
    name = "aggregate_hedge"

    def __init__(
        self,
        hold_days: int,
        entry_threshold: float,
        eta: float,
        p_min: float | None,
        lookback_days: int,
        table: dict | None = None,
        model: Any = None,
        parts_source: Mapping[tuple[str, date], Parts] | None = None,
    ) -> None:
        super().__init__(hold_days, entry_threshold, table, model)
        self.eta, self.p_min, self.lookback = float(eta), p_min, int(lookback_days)
        self.parts_source = parts_source  # precomputed scores (selection replay); else features
        self.seen: dict[date, dict[str, Parts]] = {}  # decision day -> ticker -> parts
        self.losses: dict[date, dict[str, float] | None] = {}  # realised day -> mean losses
        self.last_weights: dict[str, float] = {}

    @classmethod
    def from_config(cls) -> AggregateHedge:
        from src.strategies.base import strategy_params

        base = selected_params("aggregate_equal")
        chosen = selected_params(cls.name)
        return cls(
            int(base["hold_days"]), float(base["entry_threshold"]), float(chosen["eta"]),
            chosen.get("p_min"), int(strategy_params("aggregator")["lookback_days"]),
        )  # fmt: skip

    # -- realised labels from public bars --------------------------------------------------

    def _realised_drift(self, ticker: str, day: date, view: MarketView) -> float | None:
        need = self.lookback + 5
        stock = {b.bar_date: (b.open, b.close) for b in view.prices(ticker, need)}
        market = {b.bar_date: (b.open, b.close) for b in view.prices(self.benchmark, need)}
        ar = abnormal_returns(day, stock, market, sorted(market))
        return None if ar is None else ar.drift_ar

    def _update_losses(self, today: date, view: MarketView) -> None:
        """Score every remembered day whose drift_ar is realised by now (d' <= D-2)."""
        sessions = sorted(b.bar_date for b in view.prices(self.benchmark, self.lookback + 5))
        if len(sessions) < 2:
            return
        realised_until = sessions[-2]  # drift of d' needs the close of d'+1 <= D-1
        for d, by_ticker in self.seen.items():
            if d in self.losses or d > realised_until:
                continue
            drifts = {}
            for t in by_ticker:
                dr = self._realised_drift(t, d, view)
                if dr is not None:
                    drifts[t] = dr
            self.losses[d] = daily_losses(by_ticker, drifts, self.components)

    def weights(self, today: date, view: MarketView) -> dict[str, float]:
        self._update_losses(today, view)
        window = sorted(d for d, v in self.losses.items() if v is not None and d < today)
        rows = [self.losses[d] for d in window[-self.lookback :]]
        return hedge_weights(rows, self.components, self.eta)  # type: ignore[arg-type]

    # -- the signal --------------------------------------------------------------------------

    def signal(self, ticker: str, view: MarketView) -> Signal:
        today = view.as_of.date()
        if self.parts_source is not None:
            empty = dict.fromkeys(self.components, 0.0)
            parts = dict(self.parts_source.get((ticker, today), empty))
            p = parts.pop("p", None)
        else:
            full = self.parts(self._features(ticker, view))
            parts, p = {k: v[0] for k, v in full.items()}, full["impact"][1]
        self.seen.setdefault(today, {})[ticker] = parts
        if not self.last_weights or self._weights_day != today:
            self.last_weights, self._weights_day = self.weights(today, view), today
        w = self.last_weights
        score = sum(w[k] * parts[k] for k in self.components)
        reasons = [f"w_{k}={w[k]:.2f}" for k in self.components]
        if self.p_min is not None and (p is None or p < self.p_min):
            reasons.append(f"p<{self.p_min}")
            score = min(score, 0.0)  # below the confidence threshold: cash (R8)
        return self._signal(ticker, view, score, p, *reasons)

    _weights_day: date | None = None
