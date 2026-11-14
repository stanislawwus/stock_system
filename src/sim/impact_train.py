"""`make train`: impact features, labels, models, calibrator and the 8-K item table (D-40).

Dev splits only (R3). Steps:

1. feature panel per (watchlist ticker, session) from `MarketView(cutoff(D))`, cached as parquet
   under `impact.feature_dir` (keyed by split and a hash of the feature config);
2. `drift_ar` labels from `src.sim.labels` (execution prices; labels never reach a feature);
3. logistic regression and HistGradientBoosting fitted on `train`;
4. validation is cut in time order into val_a (first `val_fit_fraction` of sessions) and val_b;
   the model with the higher val_a AUC is kept and the calibrator (`impact.calibrator`) is fitted
   on val_a. val_b is never used for a choice, so its numbers are out-of-sample;
5. the 8-K item table: mean `drift_ar` per item over all universe names' 8-Ks on `train`.

Artifacts: `<model_dir>/impact.joblib`, `<model_dir>/event_items.json`, and a JSON summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.collectors.common import ROOT, load_periods, load_universe, watchlist_tickers
from src.db import guarded
from src.impact.features import config, feature_names, ticker_features
from src.impact.model import ImpactModel, fit_calibrator, fit_models, scores
from src.marketview import MarketView
from src.marketview.clock import decision_day
from src.nlp.events import NONE, items_to_labels
from src.sim.execution import ExecutionPrices
from src.sim.labels import drift_labels

log = logging.getLogger(__name__)
DEV = ("train", "validation")


def model_dir() -> Path:
    return ROOT / str(config()["model_dir"])


def _feature_hash() -> str:
    from src.strategies.base import strategy_params

    blob = json.dumps(
        [feature_names(), config(), strategy_params("finbert_v0")], sort_keys=True, default=str
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def split_prices(split: str, tickers: list[str], db_path: Path | None = None) -> ExecutionPrices:
    if split not in DEV:
        raise ValueError(f"split {split!r} not allowed here; use {DEV} (R3)")
    (period,) = load_periods([split])
    _, benchmark = load_universe()
    return ExecutionPrices.load(tickers, period.start, period.end, benchmark, db_path)


def build_features(
    days: list[date], tickers: list[str], benchmark: str, db_path: Path | None = None
) -> pd.DataFrame:
    rows = []
    for d in days:
        with guarded(), MarketView.for_trading_day(d, db_path) as view:
            for t in tickers:
                rows.append({"ticker": t, "day": d, **ticker_features(t, view, benchmark)})
    return pd.DataFrame(rows)


def panel(split: str, db_path: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Features joined with labels for the watchlist on `split`, cached as parquet."""
    tickers = watchlist_tickers()
    _, benchmark = load_universe()
    cache = ROOT / str(config()["feature_dir"]) / f"impact_{split}_{_feature_hash()}.parquet"
    prices = split_prices(split, tickers, db_path)
    labels = drift_labels(prices, tickers, benchmark)
    if cache.exists() and not refresh and db_path is None:
        feats = pd.read_parquet(cache)
    else:
        t0 = time.monotonic()
        feats = build_features(prices.sessions, tickers, benchmark, db_path)
        log.info("features %s: %d rows in %.0f s", split, len(feats), time.monotonic() - t0)
        if db_path is None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            feats.to_parquet(cache, index=False)
    feats["day"] = pd.to_datetime(feats["day"]).dt.date
    out = feats.merge(labels, on=["ticker", "day"], how="inner")
    return out.sort_values(["day", "ticker"]).reset_index(drop=True)


def split_validation(val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered cut of validation into val_a (fit) and val_b (report)."""
    days = sorted(val["day"].unique())
    k = int(len(days) * float(config()["val_fit_fraction"]))
    cut = days[k]
    return val[val["day"] < cut], val[val["day"] >= cut]


def val_a_range(db_path: Path | None = None) -> tuple[date, date]:
    """First and last session of val_a (also the selection period of D-40)."""
    sessions = split_prices("validation", [], db_path).sessions
    k = int(len(sessions) * float(config()["val_fit_fraction"]))
    return sessions[0], sessions[k - 1]


def val_b_range(db_path: Path | None = None) -> tuple[date, date]:
    sessions = split_prices("validation", [], db_path).sessions
    k = int(len(sessions) * float(config()["val_fit_fraction"]))
    return sessions[k], sessions[-1]


def fit_impact(train: pd.DataFrame, val: pd.DataFrame) -> tuple[ImpactModel, dict[str, Any]]:
    cfg = config()
    names = feature_names()
    bins = int(cfg["ece_bins"])
    xt, yt = train[names].to_numpy(float), train["y"].to_numpy(int)
    majority = int(yt.mean() >= 0.5)
    models = fit_models(xt, yt, cfg)
    val_a, val_b = split_validation(val)
    report: dict[str, Any] = {"n_train": len(train), "n_val_a": len(val_a), "n_val_b": len(val_b)}
    for name, m in models.items():
        report[name] = {
            part: scores(m.predict_proba(df[names].to_numpy(float))[:, 1], df["y"].to_numpy(int),
                         majority, bins)
            for part, df in (("train", train), ("validation", val), ("val_a", val_a),
                             ("val_b", val_b))
        }  # fmt: skip
    best = max(models, key=lambda k: np.nan_to_num(report[k]["val_a"]["auc"]))
    xa, ya = val_a[names].to_numpy(float), val_a["y"].to_numpy(int)
    xb, yb = val_b[names].to_numpy(float), val_b["y"].to_numpy(int)
    raw_a = models[best].predict_proba(xa)[:, 1]
    raw_b = models[best].predict_proba(xb)[:, 1]
    report["calibration"] = {"model": best, "raw_val_b": scores(raw_b, yb, majority, bins)}
    for cal in ("platt", "isotonic"):
        c = fit_calibrator(cal, raw_a, ya)
        report["calibration"][cal] = {
            "val_a_in_sample": scores(c.transform(raw_a), ya, majority, bins),
            "val_b": scores(c.transform(raw_b), yb, majority, bins),
        }
    chosen = str(cfg["calibrator"])
    model = ImpactModel(
        names, best, models[best], chosen, fit_calibrator(chosen, raw_a, ya),
        meta={"majority_class": majority, "feature_hash": _feature_hash()},
    )  # fmt: skip
    report["chosen"] = {"model": best, "calibrator": chosen}
    return model, report


def event_item_table(db_path: Path | None = None) -> dict[str, dict[str, float]]:
    """Mean drift_ar of D per 8-K item over the universe's 8-Ks decided inside `train`."""
    companies, benchmark = load_universe()
    tickers = [c.ticker for c in companies]
    prices = split_prices("train", tickers, db_path)
    labels = drift_labels(prices, tickers, benchmark).set_index(["ticker", "day"])["drift_ar"]
    sessions = prices.sessions
    span = (sessions[-1] - sessions[0]).days + 1
    acc: dict[str, list[float]] = {}
    with guarded(), MarketView.for_trading_day(sessions[-1], db_path) as view:
        for t in tickers:
            for f in view.filings(t, lookback_days=span, forms=("8-K",)):
                d = decision_day(f.available_at, sessions)
                if d is None or (t, d) not in labels.index:
                    continue
                for lab in items_to_labels(f.items) - {NONE}:
                    acc.setdefault(lab, []).append(float(labels[(t, d)]))
    return {
        lab: {"mean_drift_ar": float(np.mean(v)), "n": len(v), "std": float(np.std(v, ddof=1))}
        for lab, v in sorted(acc.items())
        if len(v) > 1
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="rebuild the cached feature panels")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = model_dir()
    out.mkdir(parents=True, exist_ok=True)
    train, val = panel("train", refresh=args.refresh), panel("validation", refresh=args.refresh)
    model, report = fit_impact(train, val)
    joblib.dump(model, out / "impact.joblib")
    items = event_item_table()
    (out / "event_items.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    (out / "impact_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["chosen"]), {k: (v["n"], round(v["mean_drift_ar"], 4))
                                         for k, v in items.items()})  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
