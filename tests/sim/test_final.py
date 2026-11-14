"""The single test-period run (src/sim/final.py; R3, L-01, D-43). Nothing here reads test data:
the arena is replaced by stubs, the lock and the artifacts live in tmp_path."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src import splits
from src.sim import arena, final
from src.sim import run as sim_run
from src.sim.backtest import resolve_split

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every prerequisite present; the lock in tmp_path."""
    models = tmp_path / "models"
    models.mkdir()
    for f in final.MODEL_FILES:
        (models / f).write_text("{}", encoding="utf-8")
    scores = tmp_path / "test_scores.parquet"
    scores.write_bytes(b"")
    scores.with_suffix(".manifest.json").write_text('{"complete": true}', encoding="utf-8")
    monkeypatch.setattr(final, "model_dir", lambda: models)
    monkeypatch.setattr(splits, "final_scores_path", lambda: scores)
    monkeypatch.setattr(splits, "FINAL_LOCK", tmp_path / "TEST_FINAL.lock")
    return tmp_path


def test_preflight_passes_when_everything_is_there(ready: Path) -> None:
    final.preflight()
    assert not splits.FINAL_LOCK.exists()  # a preflight never takes the lock


@pytest.mark.parametrize("broken", ["model", "parquet", "manifest"])
def test_preflight_refuses_without_touching_the_period(ready: Path, broken: str) -> None:
    if broken == "model":
        (ready / "models" / "impact.joblib").unlink()
    elif broken == "parquet":
        (ready / "test_scores.parquet").unlink()
    else:
        (ready / "test_scores.manifest.json").write_text('{"complete": false}', encoding="utf-8")
    with pytest.raises(final.FinalRunNotReadyError):
        final.run_final()
    assert not splits.FINAL_LOCK.exists()


def test_the_lock_exists_before_anything_is_read_and_a_crash_still_uses_the_period(
    ready: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    def crashing_arena(*args: object, **kw: object) -> dict:
        calls.append(splits.FINAL_LOCK.exists() and splits.in_final_run())
        raise RuntimeError("crash after reading")

    monkeypatch.setattr(arena, "run_arena", crashing_arena)
    with pytest.raises(RuntimeError, match="crash"):
        final.run_final()
    assert calls == [True]
    text = splits.FINAL_LOCK.read_text(encoding="utf-8")
    assert re.search(r"^started_at=.+^git_sha=.+^status=failed$", text, re.S | re.M)
    with pytest.raises(splits.TestPeriodLockedError):  # no retry: the period was used once
        final.run_final()
    assert calls == [True] and not splits.in_final_run()


def test_a_successful_run_writes_every_record_inside_the_period(
    ready: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []
    fake_run = type("R", (), {"result": type("B", (), {"sharpe": 1.0})()})()
    monkeypatch.setattr(arena, "run_arena", lambda *a, runs_out, **k: runs_out.update(
        {"buy_and_hold": fake_run, "momentum": fake_run}) or {})  # fmt: skip
    monkeypatch.setattr(arena, "record", lambda a: {"component": "arena_test"})
    monkeypatch.setattr(arena, "write", lambda rec: [ready / "arena_test.json"])
    monkeypatch.setattr(sim_run, "record", lambda res, target: {"target": target})
    monkeypatch.setattr(sim_run, "write", lambda rec: written.append(rec["target"]) or ready)
    monkeypatch.setattr("evals.run_all.refresh_doc", lambda: None)
    final.run_final()
    assert written == [0.0, 1.0]  # benchmark target 0, others target the benchmark Sharpe
    assert splits.FINAL_LOCK.read_text(encoding="utf-8").rstrip().endswith("status=ok")


def test_the_test_split_resolves_only_during_the_final_run(ready: Path) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        resolve_split(splits.FINAL_SPLIT)
    with splits.final_run():
        assert resolve_split(splits.FINAL_SPLIT) == splits.FINAL_SPLIT


@pytest.mark.parametrize("main", [arena.main, sim_run.main])
def test_both_final_flags_call_the_one_final_run(
    main: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[int] = []
    monkeypatch.setattr(final, "run_final", lambda: called.append(1))
    assert main(["--final"]) == 0  # type: ignore[operator]
    assert called == [1]


def test_make_test_final_runs_one_final_entry_point_and_does_not_write_the_lock() -> None:
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = mk.split("\ntest-final:", 1)[1]
    assert recipe.count("--final") == 1 and "src.strategies.arena --final" in recipe
    assert '> "$(FINAL_LOCK)"' not in recipe  # the lock is written by final_run(), first
    assert recipe.startswith(" need-train")  # the models must exist before the lock is taken
