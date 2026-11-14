"""Arena service (src/sim/arena.py, D-42): trials log, ablation replay, record and report."""

from __future__ import annotations

import csv
import math
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.marketview.clock import cutoff_for
from src.schemas import DISCLAIMER, EvalResult
from src.sim import arena

D = date(2026, 6, 2)


def test_ablation_replay_averages_the_kept_components_and_applies_the_threshold() -> None:
    parts = {("A", D): {"sentiment": 0.6, "event": 0.0, "impact": -0.3}}
    view = SimpleNamespace(as_of=cutoff_for(D))
    keep = arena.AblationReplay(parts, ["sentiment", "event"], 3, 0.2, "without_impact")
    assert keep.signal("A", view).score == pytest.approx(0.3)
    assert keep.horizon_days == keep.min_hold_days == 3 and keep.name == "without_impact"
    low = arena.AblationReplay(parts, ["sentiment", "impact"], 1, 0.2, "without_event")
    assert low.signal("A", view).score == 0.0  # 0.15 <= threshold -> cash (R8)
    assert keep.signal("B", view).score == 0.0  # no recorded parts -> cash


def _board(name: str, sharpe: float) -> dict:
    return {"strategy": name, "sharpe": sharpe, "n_days": 64, "round_trips": 10}


def test_trial_rows_cover_every_family() -> None:
    selected = {"impact": {"period": "val_a", "start": "2026-06-01", "end": "2026-07-16",
                           "grid": [{"hold_days": 1, "entry_threshold": 0.0, "sharpe": 0.5,
                                     "round_trips": 12}]}}  # fmt: skip
    report = {"logreg": {"val_a": {"auc": 0.47}}, "hgb": {"val_a": {"auc": 0.5}},
              "calibration": {"platt": {}, "isotonic": {}}}  # fmt: skip
    dropped = {"variant": "without_event", "dropped": "event", "sharpe": 0.1, "round_trips": 3}
    ablation = [{"variant": "all components"}, dropped]
    rows = arena.trial_rows([_board("momentum", 2.0)], ablation, (D, D), selected, report)
    fam = sorted(r["family"] for r in rows)
    assert fam == ["ablation", "arena", "calibrator", "calibrator", "impact_model",
                   "impact_model", "selection_grid"]  # fmt: skip
    assert len({r["trial_id"] for r in rows}) == len(rows)
    assert set(rows[0]) == set(arena.TRIAL_FIELDS)


def test_trials_log_is_upserted_and_never_loses_a_trial(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    first = arena.trial_rows([_board("momentum", 2.0), _board("event", 1.0)], [{}], (D, D),
                             None, None)  # fmt: skip
    arena.upsert_trials(path, first, "2026-11-14T10:00:00+00:00")
    again = arena.trial_rows([_board("momentum", 2.5)], [{}], (D, D), None, None)
    out = arena.upsert_trials(path, again, "2026-11-15T10:00:00+00:00")
    by_id = {r["trial_id"]: r for r in out}
    assert len(out) == 2  # event is kept although this run did not produce it
    mom = by_id["arena:momentum:validation"]
    assert mom["first_logged_at"] == "2026-11-14T10:00:00+00:00"
    assert mom["last_logged_at"] == "2026-11-15T10:00:00+00:00"
    assert float(mom["sharpe_ann"]) == 2.5
    with path.open(encoding="utf-8", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_trial_sharpes_are_converted_to_daily_and_blanks_skipped() -> None:
    trials = [{"sharpe_ann": "2.0"}, {"sharpe_ann": ""}, {"sharpe_ann": "nan"}]
    assert arena.trial_sharpes_daily(trials, 252) == [pytest.approx(2 / math.sqrt(252))]


def _arena_dict() -> dict:
    row = {"strategy": "aggregate_equal", "sharpe": -1.0, "sharpe_ci95": (-2.0, 0.5),
           "sharpe_val_b": -2.0, "sharpe_val_b_ci95": None, "total_return": -0.01,
           "max_drawdown": 0.05, "round_trips": 5, "mean_exposure": 0.3, "costs_paid": 10.0,
           "choice_period": "val_a", "n_days": 64}  # fmt: skip
    ctl = {"n": 10, "mode": "names", "sharpe": -1.0, "percentile": 20.0,
           "random_sharpe_p05": -2.0, "random_sharpe_median": 0.0, "random_sharpe_p95": 2.0,
           "round_trips": 5, "random_round_trips_mean": 5.0, "mean_exposure": 0.3,
           "random_mean_exposure": 0.3}  # fmt: skip
    return {
        "split": "validation", "start": "2026-06-01", "end": "2026-08-31",
        "val_b_start": "2026-07-17", "n_days": 64, "leaderboard": [row],
        "correlation": {"aggregate_equal": {"aggregate_equal": 1.0}},
        "ablation": [{"variant": "all components", "dropped": None, "sharpe": -1.0,
                      "sharpe_ci95": None, "total_return": -0.01, "round_trips": 5,
                      "delta_sharpe_vs_all": 0.0}],
        "random_controls": {"aggregate_equal": {"names": ctl, "timing_and_names": ctl}},
        "deflated_sharpe": {"aggregate_equal": {"sharpe_daily": -0.06, "sr0_daily": 0.2,
                                                "skew": 0.0, "kurtosis": 3.0, "dsr": 0.001,
                                                "psr_vs_zero": 0.2}},
        "trials": {"n": 74, "n_with_sharpe": 62, "by_family": {"arena": 7}, "path": "x.csv"},
        "run_at": datetime(2026, 11, 14, tzinfo=UTC).isoformat(),
    }  # fmt: skip


def test_record_matches_the_eval_schema_and_the_report_carries_the_disclaimer() -> None:
    rec = arena.record(_arena_dict())
    EvalResult(**rec)  # AGENTS.md section 5
    assert rec["component"] == "arena" and rec["metric"] == "deflated_sharpe_aggregate_equal"
    assert rec["value"] == 0.001 and rec["target"] == 0.95 and "Below target" in rec["notes"]
    assert rec["details"]["disclaimer"] == DISCLAIMER
    doc = arena.render(rec)
    assert DISCLAIMER in doc
    for section in ("Leaderboard", "Correlation", "ablation", "Random controls", "Deflated"):
        assert section in doc
    assert not any(w in doc.lower() for w in (" caused", " drove", " explains"))  # R11


def test_cli_refuses_the_test_split(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        arena.main(["--split", "test"])
    assert "invalid choice" in capsys.readouterr().err


def test_make_arena_entry_point_is_the_sim_arena() -> None:
    from src.strategies import arena as entry

    assert entry.main is arena.main
