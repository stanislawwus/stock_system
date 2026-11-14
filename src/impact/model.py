"""Impact model: P(drift_ar > 0) from the features of `src.impact.features` (D-28, D-40).

Pure functions on arrays; nothing here reads data. Two classifiers fitted on `train` (time order
kept, no shuffling): a logistic-regression baseline (standardised features) and sklearn's
HistGradientBoosting. A calibrator (Platt or isotonic) maps the raw probability to a calibrated
one and is fitted on validation only (val_a, the first half; val_b is kept for reporting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


class Classifier(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


def fit_models(x: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> dict[str, Classifier]:
    lr = cfg["logreg"]
    hgb = cfg["hgb"]
    logreg = make_pipeline(
        StandardScaler(), LogisticRegression(C=float(lr["C"]), max_iter=int(lr["max_iter"]))
    )
    gb = HistGradientBoostingClassifier(
        max_depth=int(hgb["max_depth"]),
        learning_rate=float(hgb["learning_rate"]),
        max_iter=int(hgb["max_iter"]),
        min_samples_leaf=int(hgb["min_samples_leaf"]),
        l2_regularization=float(hgb["l2_regularization"]),
        early_stopping=False,  # early stopping would draw a random (shuffled) holdout
        random_state=0,
    )
    return {"logreg": logreg.fit(x, y), "hgb": gb.fit(x, y)}


# --- calibration ---------------------------------------------------------------------------
class Calibrator(Protocol):
    def transform(self, p: np.ndarray) -> np.ndarray: ...


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class Platt:
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(C=1e6))

    def fit(self, p: np.ndarray, y: np.ndarray) -> Platt:
        self.model.fit(_logit(p), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(p))[:, 1]


@dataclass
class Isotonic:
    model: IsotonicRegression = field(
        default_factory=lambda: IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    )

    def fit(self, p: np.ndarray, y: np.ndarray) -> Isotonic:
        self.model.fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(p), dtype=float)


CALIBRATORS = {"platt": Platt, "isotonic": Isotonic}


def fit_calibrator(name: str, p: np.ndarray, y: np.ndarray) -> Calibrator:
    return CALIBRATORS[name]().fit(p, y)


# --- metrics -------------------------------------------------------------------------------
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int) -> list[dict[str, float]]:
    """Equal-width bins: mean predicted p, observed frequency, count (empty bins dropped)."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            p_mean, freq = float(p[m].mean()), float(y[m].mean())
            out.append({"p_mean": p_mean, "freq": freq, "n": int(m.sum())})
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    rows = reliability(p, y, bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["p_mean"] - r["freq"]) for r in rows) / n) if n else 0.0


def scores(p: np.ndarray, y: np.ndarray, majority: int, bins: int) -> dict[str, float]:
    """AUC, accuracy at 0.5, accuracy of always predicting the train majority class, Brier,
    Brier of the constant base rate of this sample (skill reference), ECE."""
    auc = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
    base = float(y.mean())
    return {
        "n": int(len(y)),
        "auc": auc,
        "accuracy": float(np.mean((p > 0.5).astype(int) == y)),
        "majority_accuracy": float(np.mean(y == majority)),
        "brier": brier(p, y),
        "brier_base_rate": brier(np.full(len(y), base), y),
        "ece": ece(p, y, bins),
        "pos_rate": base,
    }


@dataclass
class ImpactModel:
    """The fitted model the strategies use: raw classifier + calibrator + feature order."""

    features: list[str]
    model_name: str
    model: Any
    calibrator_name: str
    calibrator: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator.transform(self.predict_raw(x)), 0.0, 1.0)
