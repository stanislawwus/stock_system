"""The test period is read exactly once, by `make test-final` (AGENTS.md R3, D-33).

`MarketView` calls `assert_readable(as_of)`. Any view at or after the first test cutoff raises
`TestPeriodLockedError` unless the process runs inside `final_run()`, which only the `--final`
entry points of `make test-final` open (both call `src.sim.final.run_final`), and which writes
`TEST_FINAL.lock` on entry and refuses once it exists (L-01, D-43).
"""

from __future__ import annotations

import contextvars
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import yaml

from src.marketview.clock import cutoff_for, to_utc

ROOT = Path(__file__).resolve().parents[1]

FINAL_LOCK = ROOT / "evals" / "results" / "TEST_FINAL.lock"
_final: contextvars.ContextVar[bool] = contextvars.ContextVar("final_run", default=False)


class TestPeriodLockedError(RuntimeError):
    """The test period was touched outside the single final run."""

    __test__ = False  # not a pytest class


@lru_cache(maxsize=1)
def first_test_cutoff() -> datetime:
    """09:00 ET of the first test day, in UTC: nothing at or after it is a development input."""
    return cutoff_for(
        yaml.safe_load((ROOT / "config" / "splits.yaml").read_text(encoding="utf-8"))["test"][
            "start"
        ]
    )


def assert_readable(as_of: datetime) -> None:
    if to_utc(as_of) >= first_test_cutoff() and not _final.get():
        raise TestPeriodLockedError(
            f"as_of {as_of.isoformat()} is in the test period; only `make test-final` may read "
            "it, once (AGENTS.md R3)"
        )


FINAL_SPLIT = "test"  # the split name only the final run may simulate


def in_final_run() -> bool:
    return _final.get()


NL = chr(10)  # lock file lines end in LF on every platform


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


@contextmanager
def final_run() -> Iterator[None]:
    """Open the test period, once. The lock is written **before** anything is read (L-01): it is
    created with exclusive create, so a crash, a retry or a second process can never read the
    period again. On exit the outcome (`status=ok|failed`) is appended; a failed final run still
    counts as the one run (R3)."""
    FINAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FINAL_LOCK.open("x", encoding="utf-8", newline=NL) as fh:
            fh.write(f"started_at={_now()}{NL}git_sha={_git_sha()}{NL}")
    except FileExistsError:
        raise TestPeriodLockedError(
            f"{FINAL_LOCK.name} exists: the test period was used once"
        ) from None
    token = _final.set(True)
    status = "failed"
    try:
        yield
        status = "ok"
    finally:
        _final.reset(token)
        with FINAL_LOCK.open("a", encoding="utf-8", newline=NL) as fh:
            fh.write(f"finished_at={_now()}{NL}status={status}{NL}")


def final_scores_path() -> Path:
    """Where `make features-test` writes the test-period sentiment parquet (D-38)."""
    nlp = yaml.safe_load((ROOT / "config" / "nlp.yaml").read_text(encoding="utf-8"))
    return ROOT / nlp["features_test"]["scores_path"]


def final_scores_for(as_of: datetime) -> Path | None:
    """The parquet a view at `as_of` must add to its `sentiment` rows, or None (D-38).

    None before the first test cutoff (dev views read DuckDB only). At or after it the caller has
    already passed `assert_readable`, i.e. runs inside `final_run()`; a missing file then fails
    loudly instead of leaving every test day without sentiment.
    """
    if to_utc(as_of) < first_test_cutoff():
        return None
    assert_readable(as_of)
    path = final_scores_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `make features-test` first (D-38)")
    return path
