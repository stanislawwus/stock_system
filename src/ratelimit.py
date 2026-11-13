"""Sliding-window rate limiter shared by src/llm.py and the collectors (AGENTS.md R5, R15)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Allow at most `max_calls` per `period` seconds; `wait()` blocks until a slot is free.

    `clock` and `sleep` are injectable so tests run without real waiting.
    """

    def __init__(
        self,
        max_calls: int,
        period: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls < 1 or period <= 0:
            raise ValueError("max_calls must be >= 1 and period > 0")
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Block until a call is allowed, record it, and return the seconds slept."""
        slept = 0.0
        with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return slept
                delay = self.period - (now - self._calls[0])
                self._sleep(delay)
                slept += delay
