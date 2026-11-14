"""The test period is read exactly once, by `make test-final` (AGENTS.md R3, D-33).

`MarketView` calls `assert_readable(as_of)`. Any view at or after the first test cutoff raises
`TestPeriodLockedError` unless the process runs inside `final_run()`, which only the `--final`
entry points of `make test-final` open, and which itself refuses once `TEST_FINAL.lock` exists.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
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


@contextmanager
def final_run() -> Iterator[None]:
    """Open the test period. Refuses when the lock file shows the final run already happened."""
    if FINAL_LOCK.exists():
        raise TestPeriodLockedError(f"{FINAL_LOCK.name} exists: the test period was used once")
    token = _final.set(True)
    try:
        yield
    finally:
        _final.reset(token)
