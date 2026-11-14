"""Impact-model features per (ticker, decision day D), from `MarketView(cutoff(D))` only (R2, D-40).

Every value is computed from rows with `available_at <= 09:00 ET of D`; the same function serves
training (one view per session) and the live strategy, so there is no train/serve difference.

- `sent_mean/min/max/n`: FinBERT scores of the relevant, deduplicated news in D's window
  (`src.nlp.daily`); 0 when there is no news, `sent_n` says how many.
- `ev_<item>`: 1 if an 8-K of the ticker accepted in D's window carries that item (D-10 labels,
  9.01 dropped, others -> other_item); `ev_any` = any 8-K label.
- `growth_rev`, `growth_eps`: **growth_momentum**, the YoY growth of the latest XBRL revenue /
  diluted EPS against the same fiscal period a year earlier (`available_at` = the carrying
  10-Q/10-K, D-30). Not an earnings *surprise*: we have no consensus estimates. 0 when missing.
- `prior_ar`: abnormal return of D-1 (close(D-2) -> close(D-1), minus SPY).
- `vol`: stdev of the last `vol_window` daily close-to-close returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.marketview import MarketView
from src.nlp.daily import previous_cutoff, window_scores
from src.nlp.events import NONE, items_to_labels, labels
from src.schemas import Fact
from src.strategies.base import strategy_params

GROWTH_FACTS = {"growth_rev": "revenue", "growth_eps": "eps_diluted"}
DAYS_PER_YEAR = 365


def config() -> dict[str, Any]:
    return strategy_params("impact")


def event_columns() -> list[str]:
    return [f"ev_{lab.replace('.', '_')}" for lab in labels()]


def feature_names() -> list[str]:
    return [
        "sent_mean", "sent_min", "sent_max", "sent_n",
        *event_columns(), "ev_any",
        *GROWTH_FACTS, "prior_ar", "vol",
    ]  # fmt: skip


def growth_momentum(facts: list[Fact], name: str, tol_days: int, clip: float) -> float | None:
    """YoY growth of the newest `name` fact vs the same fiscal period about a year earlier."""
    rows = [f for f in facts if f.name == name and f.method == "xbrl"]
    if not rows:
        return None
    latest = max(rows, key=lambda f: (f.period_end, f.available_at))
    kind = latest.fiscal_period.split("-")[0]
    prior = [
        f
        for f in rows
        if f.fiscal_period.split("-")[0] == kind
        and abs((latest.period_end - f.period_end).days - DAYS_PER_YEAR) <= tol_days
    ]
    if not prior:
        return None
    base = max(prior, key=lambda f: f.available_at).value
    if base == 0:
        return None
    return float(np.clip((latest.value - base) / abs(base), -clip, clip))


def _returns(closes: list[float]) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    return c[1:] / c[:-1] - 1.0 if len(c) > 1 else np.empty(0)


def ticker_features(ticker: str, view: MarketView, benchmark: str) -> dict[str, float]:
    cfg = config()
    method = strategy_params("finbert_v0")["method"]
    out = dict.fromkeys(feature_names(), 0.0)

    scores = [s for _, s in window_scores(ticker, view, method)]
    if scores:
        out.update(
            sent_mean=float(np.mean(scores)), sent_min=min(scores), sent_max=max(scores),
            sent_n=float(len(scores)),
        )  # fmt: skip

    start = previous_cutoff(view.as_of)
    lookback = (view.as_of - start).days + 1
    for f in view.filings(ticker, lookback_days=lookback, forms=("8-K",)):
        if f.available_at <= start:
            continue
        for lab in items_to_labels(f.items) - {NONE}:
            out[f"ev_{lab.replace('.', '_')}"] = 1.0
            out["ev_any"] = 1.0

    facts = view.facts(ticker)
    for col, name in GROWTH_FACTS.items():
        g = growth_momentum(
            facts, name, int(cfg["growth_tolerance_days"]), float(cfg["growth_clip"])
        )
        out[col] = 0.0 if g is None else g

    n = int(cfg["vol_window"]) + 1
    bars = view.prices(ticker, lookback_days=n)
    spy = view.prices(benchmark, lookback_days=2)
    r = _returns([b.close for b in bars])
    if len(r) >= 2:
        out["vol"] = float(np.std(r, ddof=1))
    if len(r) and len(spy) == 2 and bars[-1].bar_date == spy[-1].bar_date:
        if bars[-2].bar_date == spy[-2].bar_date:
            out["prior_ar"] = float(r[-1] - (spy[-1].close / spy[-2].close - 1.0))
    return out
