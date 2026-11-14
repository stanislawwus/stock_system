"""Signal-stack strategies (D-40): `event`, `impact`, `aggregate_equal`.

All three read only the MarketView (features from `src.impact.features`) plus artifacts fitted
by `make train` on dev data: the 8-K item table (train), the impact model (train) and its
calibrator (val_a), and the selected hold period / entry threshold (`selected.json`, written by
`src.sim.select`). Each holds a position for exactly `hold_days` sessions (minimum holding
period = time exit), which keeps one-day signals from paying a round trip every day.

- `event`: 8-K items in D's window; score = mean over items with at least `min_label_n` train
  events of (train mean drift_ar of the item / `score_scale`), clipped to [-1, 1]; 0 without one.
- `impact`: calibrated P(drift_ar > 0) from the impact model; score = 2p - 1, `calibrated_p` = p.
- `aggregate_equal`: equal-weight mean of the sentiment (mean FinBERT of D's window), event and
  impact scores. Signals are suggestions with a confidence, not predictions (R11).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.collectors.common import ROOT, load_universe
from src.impact.features import ticker_features
from src.impact.model import ImpactModel
from src.marketview import MarketView
from src.schemas import Signal
from src.strategies.base import clip, make_signal, strategy_params


def model_dir() -> Path:
    return ROOT / str(strategy_params("impact")["model_dir"])


def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make train` first")
    return path


@lru_cache(maxsize=1)
def load_item_table() -> dict[str, dict[str, float]]:
    return json.loads(_need(model_dir() / "event_items.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_impact_model() -> ImpactModel:
    return joblib.load(_need(model_dir() / "impact.joblib"))


def selected_params(name: str) -> dict[str, Any]:
    """{hold_days, entry_threshold} chosen by `src.sim.select` for strategy `name`."""
    sel = json.loads(_need(model_dir() / "selected.json").read_text(encoding="utf-8"))
    return dict(sel[name]["chosen"])


def event_score(features: dict[str, float], table: dict[str, dict[str, float]]) -> float:
    cfg = strategy_params("event")
    vals = [
        v["mean_drift_ar"]
        for lab, v in table.items()
        if features.get(f"ev_{lab.replace('.', '_')}", 0.0) > 0 and v["n"] >= cfg["min_label_n"]
    ]
    return clip(float(np.mean(vals)) / float(cfg["score_scale"])) if vals else 0.0


class _Stacked:
    """Shared plumbing: hold period, features of D from the view, benchmark."""

    name = "stacked"

    def __init__(self, hold_days: int, entry_threshold: float) -> None:
        self.horizon_days = int(hold_days)
        self.min_hold_days = int(hold_days)
        self.entry_threshold = float(entry_threshold)
        self.benchmark = load_universe()[1]

    @classmethod
    def from_config(cls) -> _Stacked:
        return cls(**selected_params(cls.name))

    def _features(self, ticker: str, view: MarketView) -> dict[str, float]:
        return ticker_features(ticker, view, self.benchmark)

    def _signal(self, ticker: str, view: MarketView, score: float, p: float | None,
                *reasons: str) -> Signal:  # fmt: skip
        # below the entry threshold the suggestion is "cash" (R8: never trade below it)
        s = score if score > self.entry_threshold else min(score, 0.0)
        sig = make_signal(self.name, ticker, view, s, f"raw_score={score:.3f}", *reasons)
        return sig.model_copy(update={"calibrated_p": p}) if p is not None else sig


class EventStrategy(_Stacked):
    name = "event"

    def __init__(self, hold_days: int, entry_threshold: float, table: dict | None = None) -> None:
        super().__init__(hold_days, entry_threshold)
        self.table = table if table is not None else load_item_table()

    def signal(self, ticker: str, view: MarketView) -> Signal:
        f = self._features(ticker, view)
        items = [k for k, v in f.items() if k.startswith("ev_") and k != "ev_any" and v > 0]
        return self._signal(ticker, view, event_score(f, self.table), None, *items)


class ImpactStrategy(_Stacked):
    name = "impact"

    def __init__(
        self, hold_days: int, entry_threshold: float, model: ImpactModel | None = None
    ) -> None:
        super().__init__(hold_days, entry_threshold)
        self.model = model or load_impact_model()

    def probability(self, f: dict[str, float]) -> float:
        x = np.array([[f[n] for n in self.model.features]], dtype=float)
        return float(self.model.predict(x)[0])

    def signal(self, ticker: str, view: MarketView) -> Signal:
        p = self.probability(self._features(ticker, view))
        return self._signal(ticker, view, 2 * p - 1, p, f"p={p:.3f}", self.model.model_name)


class AggregateEqual(_Stacked):
    name = "aggregate_equal"

    def __init__(
        self,
        hold_days: int,
        entry_threshold: float,
        table: dict | None = None,
        model: ImpactModel | None = None,
    ) -> None:
        super().__init__(hold_days, entry_threshold)
        self.table = table if table is not None else load_item_table()
        self.impact = ImpactStrategy(hold_days, entry_threshold, model)
        self.components = list(strategy_params(self.name)["components"])

    def parts(self, f: dict[str, float]) -> dict[str, tuple[float, float | None]]:
        p = self.impact.probability(f)
        return {
            "sentiment": (f["sent_mean"], None),
            "event": (event_score(f, self.table), None),
            "impact": (2 * p - 1, p),
        }

    def signal(self, ticker: str, view: MarketView) -> Signal:
        parts = self.parts(self._features(ticker, view))
        score = float(np.mean([parts[c][0] for c in self.components]))
        reasons = [f"{c}={parts[c][0]:.3f}" for c in self.components]
        return self._signal(ticker, view, score, parts["impact"][1], *reasons)
