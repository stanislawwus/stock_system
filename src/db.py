"""Single DuckDB connection factory and the direct-access guard (AGENTS.md R7, decisions D-25).

- `connect()` is the only place that opens `data/stock.duckdb`. Readers get `read_only=True`;
  the one writer passes `writer=True` and takes `data/.duckdb.write.lock`.
- While a `guarded()` context is active (strategy evaluation, feature building, chat tools),
  `connect()` raises `DirectDataAccessError`: inside that context data is reachable only through
  `MarketView(as_of)`, which uses the private `_connect_for_marketview()` path.
- Table definitions: `src/marketview/schema.sql`.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

DEFAULT_DB_PATH = Path("data/stock.duckdb")
WRITE_LOCK_PATH = Path("data/.duckdb.write.lock")

_guard: contextvars.ContextVar[bool] = contextvars.ContextVar("marketview_guard", default=False)


class DirectDataAccessError(RuntimeError):
    """Raised when code bypasses MarketView inside a guarded context."""


class WriterBusyError(RuntimeError):
    """Raised when a second writer tries to open the database."""


@contextmanager
def guarded() -> Iterator[None]:
    """Forbid direct `connect()` calls for the duration of the block."""
    token = _guard.set(True)
    try:
        yield
    finally:
        _guard.reset(token)


def is_guarded() -> bool:
    return _guard.get()


def connect(db_path: Path | None = None, *, writer: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the database. Read-only unless `writer=True` (then the file lock is taken).

    Raises DirectDataAccessError inside `guarded()`, WriterBusyError if the lock is held.
    """
    raise NotImplementedError("design phase: see tests/test_marketview.py")


def _connect_for_marketview(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Read-only connection for MarketView only; bypasses the guard. Not for general use."""
    raise NotImplementedError("design phase")


def init_schema(db_path: Path | None = None) -> None:
    """Apply src/marketview/schema.sql (idempotent). Writer only."""
    raise NotImplementedError("design phase")
