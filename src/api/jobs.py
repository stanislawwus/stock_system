"""Background job registry for long operations (collect, arena, backtest). Interface only.

Jobs run in a thread pool inside the API process; the collect job is the API's only writer and
takes the DuckDB write lock (R7). Results are kept in memory for the life of the process.
"""

from __future__ import annotations

from collections.abc import Callable

from src.schemas import JobStatus


class JobStore:
    def submit(self, name: str, fn: Callable[[], dict]) -> str:
        """Queue `fn`, return a job_id."""
        raise NotImplementedError("design phase")

    def status(self, job_id: str) -> JobStatus | None:
        raise NotImplementedError("design phase")


jobs = JobStore()
