"""Single DuckDB connection factory and the direct-access guard (AGENTS.md R7, decisions D-25).

- `connect()` is the only place that opens `data/stock.duckdb`. Readers get `read_only=True`;
  the one writer passes `writer=True` and takes `data/.duckdb.write.lock`.
- While a `guarded()` context is active (strategy evaluation, feature building, chat tools),
  `connect()` raises `DirectDataAccessError`: inside that context data is reachable only through
  `MarketView(as_of)`, which uses the private `_connect_for_marketview()` path.
- Table definitions: `src/marketview/schema.sql`.

The writer lock is an OS-level byte lock on an open file (msvcrt on Windows, fcntl elsewhere),
so a crashed writer never leaves a stale lock behind: the OS drops it with the process.
"""

from __future__ import annotations

import contextvars
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

import duckdb

DEFAULT_DB_PATH = Path("data/stock.duckdb")
WRITE_LOCK_PATH = Path("data/.duckdb.write.lock")
SCHEMA_PATH = Path(__file__).resolve().parent / "marketview" / "schema.sql"

_guard: contextvars.ContextVar[bool] = contextvars.ContextVar("marketview_guard", default=False)
_lock_handle: IO[bytes] | None = None


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


def db_path_default() -> Path:
    """`DUCKDB_PATH` from the environment, else `data/stock.duckdb`."""
    env = os.environ.get("DUCKDB_PATH")
    return Path(env) if env else DEFAULT_DB_PATH


def _lock_file(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def acquire_writer_lock(lock_path: Path | None = None) -> None:
    """Take the process-wide writer lock; WriterBusyError if another process holds it."""
    global _lock_handle
    if _lock_handle is not None:
        return
    path = lock_path or WRITE_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")  # noqa: SIM115 - held for the life of the writer
    try:
        _lock_file(handle)
    except OSError as exc:
        handle.close()
        raise WriterBusyError(
            f"another process holds {path}; DuckDB has a single writer (AGENTS.md R7)"
        ) from exc
    _lock_handle = handle


def release_writer_lock() -> None:
    global _lock_handle
    if _lock_handle is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt

            _lock_handle.seek(0)
            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        _lock_handle.close()
        _lock_handle = None


def connect(db_path: Path | None = None, *, writer: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the database. Read-only unless `writer=True` (then the file lock is taken).

    Raises DirectDataAccessError inside `guarded()`, WriterBusyError if the lock is held.
    """
    if is_guarded():
        raise DirectDataAccessError(
            "direct database access inside a guarded context; use MarketView(as_of) (AGENTS.md R2)"
        )
    path = Path(db_path) if db_path else db_path_default()
    if writer:
        acquire_writer_lock(path.parent / WRITE_LOCK_PATH.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(path))
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `make collect-dev` first")
    return duckdb.connect(str(path), read_only=True)


@contextmanager
def writer_session(db_path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """The single writer: lock + read-write connection + schema, released on exit."""
    con = connect(db_path, writer=True)
    try:
        con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        yield con
    finally:
        con.close()
        release_writer_lock()


def _connect_for_marketview(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Read-only connection for MarketView only; bypasses the guard. Not for general use."""
    path = Path(db_path) if db_path else db_path_default()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `make collect-dev` first")
    return duckdb.connect(str(path), read_only=True)


def init_schema(db_path: Path | None = None) -> None:
    """Apply src/marketview/schema.sql (idempotent). Writer only."""
    with writer_session(db_path):
        pass
