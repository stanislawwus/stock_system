"""Background job registry for long operations (collect, arena). AGENTS.md R7, R13.

Jobs run in a small thread pool inside the API process (default one worker: the collect job is the
only writer the API can start, and it takes the DuckDB write lock). Results live in memory for the
life of the process; the oldest finished jobs are dropped past `jobs.max_kept`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from src.briefing import api_config
from src.schemas import JobStatus

log = logging.getLogger(__name__)


class JobStore:
    def __init__(self, max_workers: int | None = None, max_kept: int | None = None) -> None:
        cfg = api_config()["jobs"]
        self._pool = ThreadPoolExecutor(max_workers=max_workers or int(cfg["max_workers"]))
        self._max_kept = max_kept or int(cfg["max_kept"])
        self._jobs: OrderedDict[str, JobStatus] = OrderedDict()
        self._lock = threading.Lock()

    def _set(self, job_id: str, **changes: object) -> None:
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)

    def submit(self, name: str, fn: Callable[[], dict]) -> str:
        """Queue `fn`, return a job_id."""
        job_id = f"{name}-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._jobs[job_id] = JobStatus(job_id=job_id, status="queued")
            finished = [k for k, v in self._jobs.items() if v.status in ("done", "failed")]
            for k in finished[: max(len(self._jobs) - self._max_kept, 0)]:
                del self._jobs[k]

        def run() -> None:
            self._set(job_id, status="running")
            try:
                self._set(job_id, status="done", result=fn())
            except Exception as exc:  # a job never takes the API down
                log.exception("job %s failed", job_id)
                self._set(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")

        self._pool.submit(run)
        return job_id

    def status(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)


jobs = JobStore()
