"""Signal stack (D-40): growth_momentum, drift labels, impact model + calibration, event score,
stacked strategies' thresholds, and the selection rule. Offline, no database."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from evals.sim_fixtures import market
from src.impact.features import feature_names, growth_momentum
from src.impact.model import ImpactModel, ece, fit_calibrator, fit_models, reliability, scores
from src.marketview.clock import cutoff_for
from src.schemas import Fact
from src.sim.impact_train import split_validation
from src.sim.labels import drift_labels
from src.sim.select import Replay, choose
from src.strategies.base import strategy_params
from src.strategies.signal_stack import AggregateEqual, ImpactStrategy, event_score

AT = datetime(2026, 1, 30, 13, tzinfo=UTC)


def _fact(fp: str, end: date, value: float, name: str = "revenue") -> Fact:
    unit = "USD" if name == "revenue" else "USD/share"
    return Fact(available_at=AT, ticker="A", name=name, value=value, unit=unit, period_end=end,
                fiscal_period=fp, method="xbrl", source_ref="x")  # fmt: skip


def test_growth_momentum_is_yoy_on_the_same_fiscal_period() -> None:
    facts = [
        _fact("Q1-2025", date(2024, 12, 28), 100.0),
        _fact("FY-2025", date(2025, 9, 27), 999.0),  # different kind: ignored
        _fact("Q1-2026", date(2025, 12, 27), 115.0),
    ]
    assert growth_momentum(facts, "revenue", 20, 1.0) == pytest.approx(0.15)
    assert growth_momentum(facts[1:], "revenue", 20, 1.0) is None  # no year-ago match
    assert growth_momentum(facts, "eps_diluted", 20, 1.0) is None
    loss = [_fact("Q1-2025", date(2024, 12, 28), -2.0, "eps_diluted"),
            _fact("Q1-2026", date(2025, 12, 27), 1.0, "eps_diluted")]  # fmt: skip
    assert growth_momentum(loss, "eps_diluted", 20, 1.0) == 1.0  # +150 % clipped


def test_drift_labels_match_hand_computation_and_skip_the_last_day() -> None:
    days = [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]
    prices = market(
        {"A": [(100, 100), (102, 104), (104, 110)], "SPY": [(50, 50), (50, 51), (51, 51)]}, days
    )
    df = drift_labels(prices, ["A"], "SPY")
    # D = 06-02: A 102 -> 110 (+7.843 %), SPY 50 -> 51 (+2 %); D-1 and D+1 exist only for 06-02
    assert list(df["day"]) == [days[1]]
    assert df["drift_ar"].iloc[0] == pytest.approx(110 / 102 - 51 / 50)
    assert df["y"].iloc[0] == 1


def test_calibration_metrics_and_calibrators() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)  # perfectly calibrated by construction
    assert ece(p, y, 10) < 0.03
    assert ece(np.clip(p * 0.5 + 0.5, 0, 1), y, 10) > 0.15  # overconfident upward
    assert sum(r["n"] for r in reliability(p, y, 10)) == 4000
    squashed = 0.5 + (p - 0.5) * 0.2
    for name in ("platt", "isotonic"):
        cal = fit_calibrator(name, squashed, y)
        q = cal.transform(squashed)
        assert q.min() >= 0 and q.max() <= 1
        assert ece(q, y, 10) < ece(squashed, y, 10)
        assert np.all(np.diff(q[np.argsort(squashed)]) >= -1e-9)  # monotone


def test_models_fit_in_time_order_and_score() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 3))
    y = (x[:, 0] + 0.3 * rng.normal(size=600) > 0).astype(int)
    cfg = strategy_params("impact")
    models = fit_models(x[:400], y[:400], cfg)
    for m in models.values():
        s = scores(m.predict_proba(x[400:])[:, 1], y[400:], majority=1, bins=10)
        assert s["auc"] > 0.85 and 0 <= s["brier"] < s["brier_base_rate"]
        assert s["majority_accuracy"] == pytest.approx(y[400:].mean())


def test_split_validation_is_time_ordered() -> None:
    import pandas as pd

    days = [date(2026, 6, d) for d in range(1, 11)]
    val = pd.DataFrame({"day": days * 2, "y": 0})
    a, b = split_validation(val)
    assert a["day"].max() < b["day"].min() and len(a) + len(b) == len(val)


TABLE = {"2.02": {"mean_drift_ar": 0.004, "n": 50}, "1.01": {"mean_drift_ar": -0.02, "n": 40},
         "3.02": {"mean_drift_ar": 0.05, "n": 3}}  # fmt: skip


def test_event_score_uses_items_with_enough_train_events() -> None:
    f = dict.fromkeys(feature_names(), 0.0)
    assert event_score(f, TABLE) == 0.0
    f["ev_3_02"] = 1.0  # n = 3 < min_label_n: no weight
    assert event_score(f, TABLE) == 0.0
    f["ev_2_02"] = 1.0
    assert event_score(f, TABLE) == pytest.approx(0.004 / strategy_params("event")["score_scale"])
    f["ev_1_01"] = 1.0
    assert event_score(f, TABLE) < 0


class _Const:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile([1 - self.p, self.p], (len(x), 1))


class _Identity:
    def transform(self, p: np.ndarray) -> np.ndarray:
        return p


def _model(p: float) -> ImpactModel:
    return ImpactModel(feature_names(), "const", _Const(p), "identity", _Identity())


VIEW = SimpleNamespace(as_of=cutoff_for(date(2026, 6, 2)))


def test_impact_threshold_applies_to_calibrated_probability() -> None:
    f = dict.fromkeys(feature_names(), 0.0)
    s = ImpactStrategy(hold_days=1, entry_threshold=0.2, model=_model(0.55))
    p = s.probability(f)
    sig = s._signal("A", VIEW, 2 * p - 1, p)  # type: ignore[arg-type]
    assert sig.calibrated_p == pytest.approx(0.55) and sig.score == 0.0  # 0.1 <= 0.2: cash
    s2 = ImpactStrategy(hold_days=1, entry_threshold=0.05, model=_model(0.55))
    assert s2._signal("A", VIEW, 0.1, 0.55).score == pytest.approx(0.1)  # type: ignore[arg-type]
    assert s.min_hold_days == s.horizon_days == 1


def test_aggregate_is_the_equal_weight_mean() -> None:
    f = dict.fromkeys(feature_names(), 0.0)
    f.update(sent_mean=0.6, ev_2_02=1.0)
    agg = AggregateEqual(hold_days=3, entry_threshold=0.0, table=TABLE, model=_model(0.7))
    parts = agg.parts(f)
    want = np.mean([0.6, event_score(f, TABLE), 2 * 0.7 - 1])
    assert np.mean([v for v, _ in parts.values()]) == pytest.approx(want)


def test_replay_and_choose() -> None:
    raw = {("A", date(2026, 6, 2)): 0.15}
    r = Replay("x", raw, hold=3, thr=0.1)
    assert r.signal("A", VIEW).score == pytest.approx(0.15)  # type: ignore[arg-type]
    assert Replay("x", raw, 3, 0.2).signal("A", VIEW).score == 0.0  # type: ignore[arg-type]
    rows = [
        {"sharpe": 2.0, "round_trips": 3},
        {"sharpe": 1.0, "round_trips": 30},
        {"sharpe": 1.0, "round_trips": 12},
    ]
    best, ok = choose(rows, min_trades=10)
    assert ok and best["round_trips"] == 12  # eligible only; tie -> fewer trades
    best, ok = choose(rows[:1], min_trades=10)
    assert not ok and best["sharpe"] == 2.0


def test_impact_and_strategies_never_import_execution_or_labels() -> None:
    """R2: open(D) and drift_ar labels stay in src/sim; features and strategies never see them."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src"
    bad = re.compile(r"src\.sim\.(execution|labels|impact_train)|ExecutionPrices|drift_labels")
    offenders = [
        p.as_posix()
        for d in ("impact", "strategies", "nlp")
        for p in (root / d).rglob("*.py")
        if bad.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, offenders
