"""Shared classification metrics for the evals: per-class P/R/F1, confusion matrix, i.i.d.
bootstrap CI of a metric over items, multi-label micro/macro F1. Pure numpy, no I/O apart from
`eval_config()`."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Hashable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import yaml

from src.collectors.common import CONFIG_DIR, ROOT


@lru_cache(maxsize=1)
def eval_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "evals.yaml").read_text(encoding="utf-8"))


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def macro_f1(
    gold: Sequence[Hashable], pred: Sequence[Hashable], labels: Sequence[Hashable]
) -> float:
    g, p = np.asarray(gold), np.asarray(pred)
    f1s = []
    for c in labels:
        tp = np.sum((g == c) & (p == c))
        f1s.append(_prf(tp, np.sum((g != c) & (p == c)), np.sum((g == c) & (p != c)))[2])
    return float(np.mean(f1s))


def classification_report(
    gold: Sequence[Hashable], pred: Sequence[Hashable], labels: Sequence[Hashable]
) -> dict[str, Any]:
    """Macro-F1, accuracy, per-class precision/recall/F1/support, confusion[gold][pred]."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    g, p = np.asarray(gold), np.asarray(pred)
    per: dict[str, dict[str, float]] = {}
    for c in labels:
        tp = float(np.sum((g == c) & (p == c)))
        fp = float(np.sum((g != c) & (p == c)))
        fn = float(np.sum((g == c) & (p != c)))
        prec, rec, f1 = _prf(tp, fp, fn)
        per[str(c)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(g == c)),
        }
    confusion = {str(a): {str(b): int(np.sum((g == a) & (p == b))) for b in labels} for a in labels}
    return {
        "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
        "accuracy": round(float(np.mean(g == p)), 4) if len(g) else 0.0,
        "per_class": per,
        "confusion_gold_rows_pred_cols": confusion,
    }


def bootstrap_ci(
    n: int,
    stat: Callable[[np.ndarray], float],
    *,
    n_resamples: int | None = None,
    ci: float | None = None,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile CI of `stat(indices)` under an i.i.d. bootstrap of `n` items."""
    cfg = eval_config()["bootstrap"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    reps = int(cfg["n_resamples"] if n_resamples is None else n_resamples)
    level = float(cfg["ci"] if ci is None else ci)
    vals = np.array([stat(rng.integers(0, n, size=n)) for _ in range(reps)])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return round(float(lo), 4), round(float(hi), 4)


def multilabel_report(
    gold: Sequence[set[str]], pred: Sequence[set[str]], labels: Sequence[str]
) -> dict[str, Any]:
    """Micro and macro F1 over `labels` plus per-label precision/recall/F1/support.

    Macro averages only labels with gold support > 0 (a label absent from gold has undefined
    recall); the labels it skips are listed."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    per: dict[str, dict[str, float]] = {}
    tp_all = fp_all = fn_all = 0
    for lab in labels:
        tp = sum(lab in g and lab in p for g, p in zip(gold, pred, strict=True))
        fp = sum(lab not in g and lab in p for g, p in zip(gold, pred, strict=True))
        fn = sum(lab in g and lab not in p for g, p in zip(gold, pred, strict=True))
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        prec, rec, f1 = _prf(tp, fp, fn)
        per[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
            "predicted": tp + fp,
        }
    supported = [lab for lab in labels if per[lab]["support"] > 0]
    return {
        "micro_f1": round(_prf(tp_all, fp_all, fn_all)[2], 4),
        "macro_f1": round(float(np.mean([per[x]["f1"] for x in supported])), 4)
        if supported
        else 0.0,
        "macro_over": supported,
        "exact_set_match": round(
            float(np.mean([g == p for g, p in zip(gold, pred, strict=True)])), 4
        )
        if gold
        else 0.0,
        "per_label": per,
    }
