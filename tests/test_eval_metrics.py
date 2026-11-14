"""Shared eval metrics (evals/metrics.py): hand-computed small cases."""

from __future__ import annotations

import numpy as np
import pytest

from evals.metrics import bootstrap_ci, classification_report, macro_f1, multilabel_report


def test_classification_report_by_hand() -> None:
    gold = ["pos", "pos", "neu", "neg"]
    pred = ["pos", "neu", "neu", "pos"]
    rep = classification_report(gold, pred, ["pos", "neu", "neg"])
    # pos: tp1 fp1 fn1 -> P .5 R .5 F .5; neu: tp1 fp1 fn0 -> P .5 R 1 F .6667; neg: 0
    assert rep["per_class"]["pos"]["recall"] == 0.5
    assert rep["per_class"]["neu"]["f1"] == pytest.approx(0.6667)
    assert rep["per_class"]["neg"]["f1"] == 0.0
    assert rep["macro_f1"] == pytest.approx((0.5 + 0.6667) / 3, abs=1e-4)
    assert rep["confusion_gold_rows_pred_cols"]["neg"]["pos"] == 1
    assert rep["accuracy"] == 0.5
    assert macro_f1(gold, pred, ["pos", "neu", "neg"]) == pytest.approx(rep["macro_f1"], abs=1e-4)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible() -> None:
    x = np.r_[np.ones(60), np.zeros(40)]
    lo, hi = bootstrap_ci(len(x), lambda i: float(x[i].mean()), n_resamples=500, seed=1)
    assert lo < 0.6 < hi
    assert (lo, hi) == bootstrap_ci(len(x), lambda i: float(x[i].mean()), n_resamples=500, seed=1)


def test_multilabel_micro_macro_by_hand() -> None:
    gold = [{"2.02"}, {"2.02", "5.02"}, {"none"}]
    pred = [{"2.02"}, {"2.02"}, {"8.01"}]
    rep = multilabel_report(gold, pred, ["2.02", "5.02", "8.01", "none"])
    # tp: 2.02 x2; fn: 5.02, none; fp: 8.01 -> micro P 2/3 R 2/4 F 0.5714
    assert rep["micro_f1"] == pytest.approx(0.5714, abs=1e-4)
    # macro over labels with gold support (2.02 F1 1, 5.02 0, none 0); 8.01 has no support
    assert rep["macro_over"] == ["2.02", "5.02", "none"]
    assert rep["macro_f1"] == pytest.approx(1 / 3, abs=1e-4)
    assert rep["exact_set_match"] == pytest.approx(1 / 3, abs=1e-4)
    assert rep["per_label"]["8.01"]["predicted"] == 1


def test_cluster_bootstrap_resamples_whole_groups() -> None:
    import numpy as np

    from evals.metrics import cluster_bootstrap_ci

    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    groups = [[0, 1], [2, 3], [4, 5]]  # group means 1, 0, 0.5
    lo, hi = cluster_bootstrap_ci(groups, lambda i: float(y[i].mean()))
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert cluster_bootstrap_ci([[0, 1]], lambda i: 0.0) is None  # one cluster: no interval
