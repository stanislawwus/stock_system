"""Impact, calibration and aggregator evals (AGENTS.md R4; docs/03_eval_plan.md; D-40).

impact
- metric: AUC of P(drift_ar > 0) on validation (all 62 sessions x 10 names), model chosen on
  val_a; details: logistic regression vs HistGradientBoosting, accuracy vs the majority class,
  Brier vs the base rate, on train / val_a / val_b.
- gold set: derived, `drift_ar` sign from prices (src/sim/labels.py), D-28.
- target: 0.55. why: AUC is threshold-free with chance at 0.5.

calibration
- metric: Brier skill score of the calibrated p on val_b, 1 - Brier / Brier(base rate of val_b);
  the calibrator is fitted on val_a only, so val_b is out-of-sample. Details: ECE, reliability
  bins, Platt vs isotonic, raw. Plot: docs/figs/reliability_impact.png.
- target: > 0 (better than always saying the base rate). why: Brier checks the probability itself.

aggregator
- metric: Sharpe after costs of `aggregate_equal` on val_b (never used for any choice).
- target: buy-and-hold Sharpe on the same days; details: event, impact, finbert_v0 on val_b.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any

import numpy as np

from evals.metrics import cluster_bootstrap_ci
from src.collectors.common import ROOT
from src.impact.features import config, feature_names
from src.impact.model import fit_calibrator, reliability
from src.schemas import DISCLAIMER
from src.sim.backtest import run_backtest
from src.sim.impact_train import fit_impact, panel, split_validation, val_b_range

FIG = ROOT / "docs" / "figs" / "reliability_impact.png"


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def _record(component: str, metric: str, value: float, target: float, **kw: Any) -> dict:
    return {
        "component": component, "metric": metric, "value": round(float(value), 4),
        "target": target, "git_sha": _git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"), **kw,
    }  # fmt: skip


def _round(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _round(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_round(v) for v in d]
    return round(d, 4) if isinstance(d, float) else d


def _fit() -> tuple[Any, dict[str, Any], Any, Any]:
    train, val = panel("train"), panel("validation")
    model, report = fit_impact(train, val)
    return model, report, train, val


def _day_groups(frame: Any) -> list[list[int]]:
    """Row indices per session: the 10 names of one day share the market move (cluster)."""
    days = frame["day"].to_numpy()
    return [list(np.flatnonzero(days == d)) for d in sorted(set(days))]


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def _brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    base = float(np.mean((y.mean() - y) ** 2))
    return 1 - float(np.mean((p - y) ** 2)) / base if base > 0 else float("nan")


def evaluate_impact() -> dict[str, Any]:
    model, report, train, val = _fit()
    v = report[model.model_name]["validation"]
    pv = model.predict(val[feature_names()].to_numpy(float))
    yv = val["y"].to_numpy(int)
    auc_ci = cluster_bootstrap_ci(_day_groups(val), lambda i: _auc(yv[i], pv[i]))
    notes = (
        f"{model.model_name} (higher val_a AUC) scores AUC {v['auc']:.3f} on validation; "
        f"logreg {report['logreg']['validation']['auc']:.3f}. Out-of-sample accuracy "
        f"{v['accuracy']:.3f} vs majority {v['majority_accuracy']:.3f}; Brier {v['brier']:.4f} vs "
        f"base rate {v['brier_base_rate']:.4f}. HGB train AUC "
        f"{report['hgb']['train']['auc']:.3f} shows it memorises train; the features carry no "
        f"detectable next-day signal in this sample. 95% CI of the AUC (bootstrap over "
        f"{len(set(val['day']))} days) {auc_ci}."
    )
    return _record(
        "impact", "auc", v["auc"], 0.55,
        why_this_metric="AUC is threshold-free with chance at 0.5; Brier and majority-class "
        "accuracy are in details",
        gold_set="derived: sign(drift_ar) from prices (src/sim/labels.py, D-28)",
        n=int(v["n"]), split="validation", model=f"{model.model_name} (fit on train)",
        notes=notes, details=_round({**report, "auc_ci95_day_bootstrap": auc_ci}),
        disclaimer=DISCLAIMER,
    )  # fmt: skip


def _plot(rows: dict[str, list[dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect")
    for (name, bins), color in zip(rows.items(), ("#c0504d", "#4f81bd", "#9bbb59"), strict=False):
        ax.plot([b["p_mean"] for b in bins], [b["freq"] for b in bins], "o-", color=color,
                label=name, ms=4)  # fmt: skip
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel("predicted P(drift_ar > 0)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Impact model reliability, val_b (out-of-sample)")
    ax.legend(loc="upper left", frameon=False)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    plt.close(fig)


def evaluate_calibration() -> dict[str, Any]:
    model, report, _, val = _fit()
    bins = int(config()["ece_bins"])
    names = feature_names()
    val_a, val_b = split_validation(val)
    raw_a = model.predict_raw(val_a[names].to_numpy(float))
    raw_b = model.predict_raw(val_b[names].to_numpy(float))
    yb = val_b["y"].to_numpy(int)
    ya = val_a["y"].to_numpy(int)
    rel = {"raw": reliability(raw_b, yb, bins)}
    for cal in ("platt", "isotonic"):
        rel[cal] = reliability(fit_calibrator(cal, raw_a, ya).transform(raw_b), yb, bins)
    _plot(rel)
    cal = report["calibration"]
    chosen = cal[model.calibrator_name]["val_b"]
    skill = 1 - chosen["brier"] / chosen["brier_base_rate"]
    p_b = model.predict(val_b[names].to_numpy(float))
    skill_ci = cluster_bootstrap_ci(_day_groups(val_b), lambda i: _brier_skill(yb[i], p_b[i]))
    notes = (
        f"{model.calibrator_name} on val_a: val_b Brier {chosen['brier']:.4f} vs base rate "
        f"{chosen['brier_base_rate']:.4f} (raw {cal['raw_val_b']['brier']:.4f}), ECE "
        f"{chosen['ece']:.3f} (raw {cal['raw_val_b']['ece']:.3f}). Calibration shrinks the "
        "overconfident raw probabilities toward the base rate but cannot add skill the model "
        "lacks; the base rate itself shifted between val_a and val_b. 95% CI of the Brier "
        f"skill (bootstrap over val_b days) {skill_ci}."
    )
    return _record(
        "calibration", "brier_skill_val_b", skill, 0.0,
        why_this_metric="Brier scores the probability itself; skill > 0 means better than "
        "always predicting the base rate",
        gold_set="derived: sign(drift_ar) on val_b; calibrator fitted on val_a only",
        n=int(chosen["n"]), split="validation",
        model=f"{model.model_name} + {model.calibrator_name}", notes=notes,
        details=_round({**cal, "reliability_val_b": rel, "figure": str(FIG.relative_to(ROOT)),
                        "brier_skill_ci95_day_bootstrap": skill_ci,
                        "p_range_val_b": [float(np.min(p_b)), float(np.max(p_b))]}),
    )  # fmt: skip


def evaluate_aggregator() -> dict[str, Any]:
    start, end = val_b_range()
    names = ["aggregate_equal", "impact", "event", "finbert_v0", "buy_and_hold"]
    res = {n: run_backtest(n, "validation", start=start, end=end) for n in names}
    row = {
        n: {"sharpe": r.sharpe, "total_return": r.total_return, "max_drawdown": r.max_drawdown,
            "n_trades": r.n_trades, "costs_paid": r.costs_paid, "sharpe_ci95": r.sharpe_ci95}
        for n, r in res.items()
    }  # fmt: skip
    agg, bh = res["aggregate_equal"], res["buy_and_hold"]
    target = bh.sharpe or 0.0
    ci = agg.sharpe_ci95
    notes = (
        f"val_b {start}..{end} ({agg.n_days} days), never used for a choice. aggregate_equal "
        f"Sharpe {agg.sharpe or 0:.2f} (95% CI {ci}) vs buy-and-hold {target:.2f}; costs "
        f"{agg.costs_paid:.0f} USD over {agg.n_trades} trades. Equal weights include an impact "
        "component with no out-of-sample skill (see impact)."
    )
    return _record(
        "aggregator", "sharpe_val_b", agg.sharpe or 0.0, round(target, 4),
        why_this_metric="risk-adjusted return after costs on days no choice has seen; the "
        "target is simply holding the same names",
        gold_set="derived: buy_and_hold on val_b, same tickers",
        n=agg.n_days, split="validation", model="equal weights (sentiment, event, impact)",
        notes=notes, details=_round({**row, "val_b": [str(start), str(end)]}),
        disclaimer=DISCLAIMER,
    )  # fmt: skip
