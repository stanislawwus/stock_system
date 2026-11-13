"""The only LLM gateway (AGENTS.md R5, decisions D-06). Interface only; no implementation yet.

Guarantees the implementation must give:
- provider: OpenRouter; models restricted to the "Allowed" table of docs/inputs/free_models.md,
  ordered by config/llm.yaml; a model is skipped after 2 consecutive failures in a run
- disk cache keyed by sha256(model, prompt, params, schema name, items) under config `cache_dir`
- retries with exponential backoff (`max_retries`, `retry_backoff_seconds`), then next model
- JSON output validated against the pydantic `schema`; invalid output counts as a failure
- per-call JSONL record (`CallRecord`) appended to `<trace_dir>/<UTC date>.jsonl`
- persistent daily counter (`.llm_budget.json`, UTC day) with cap `daily_budget`; when reached
  raise BudgetExceeded BEFORE the HTTP call; cache hits do not consume budget
- every caller passes a deterministic `fallback`; `call_json` uses it on BudgetExceeded and on
  total failure, and reports `used_fallback=True`; it never returns partial results silently
- untrusted text is wrapped by `wrap_untrusted()`; model output is data, never executed
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

ALLOWED_MODELS_DOC = Path("docs/inputs/free_models.md")
CONFIG_PATH = Path("config/llm.yaml")
BUDGET_PATH = Path(".llm_budget.json")

T = TypeVar("T", bound=BaseModel)
Task = Literal["intake", "sentiment", "events", "facts", "chat", "relevance", "researcher"]


class LLMError(RuntimeError):
    """All models failed (network, rate limit, invalid JSON after retries)."""


class BudgetExceeded(LLMError):
    """The daily call cap is reached; the caller must use its deterministic fallback."""


@dataclass(frozen=True)
class LLMParams:
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0


class CallRecord(BaseModel):
    """One line of the JSONL trace."""

    ts: datetime
    task: Task
    model: str | None
    attempt: int
    n_items: int
    prompt_sha256: str
    cache_hit: bool
    status: Literal["ok", "invalid_json", "http_error", "budget_exceeded", "fallback"]
    latency_ms: int
    error: str | None = None


@dataclass
class LLMResult(Generic[T]):
    outputs: list[T]
    model: str | None  # None when the deterministic fallback produced the outputs
    used_fallback: bool
    cache_hits: int = 0
    records: list[CallRecord] = field(default_factory=list)


class DailyCounter:
    """Persistent per-UTC-day counter of HTTP calls with a hard cap."""

    def __init__(self, path: Path = BUDGET_PATH, cap: int | None = None) -> None:
        self.path = path
        self.cap = cap

    def remaining(self) -> int:
        raise NotImplementedError("design phase")

    def consume(self, n: int = 1) -> None:
        """Increment or raise BudgetExceeded without incrementing."""
        raise NotImplementedError("design phase")


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(
        model: str,
        prompt: str,
        params: LLMParams,
        schema: type[BaseModel],
        items: Sequence[dict],
    ) -> str:
        raise NotImplementedError("design phase")

    def get(self, key: str) -> list[dict] | None:
        raise NotImplementedError("design phase")

    def put(self, key: str, value: list[dict]) -> None:
        raise NotImplementedError("design phase")


def allowed_model_ids() -> set[str]:
    """Model ids from the 'Allowed' table of docs/inputs/free_models.md."""
    raise NotImplementedError("design phase")


def wrap_untrusted(text: str, max_chars: int = 6000) -> str:
    """Strip HTML, truncate, and wrap text in delimiters the system prompt declares as data (R6)."""
    raise NotImplementedError("design phase")


def call_json(
    task: Task,
    prompt: str,
    items: Sequence[dict],
    schema: type[T],
    *,
    fallback: Callable[[Sequence[dict]], list[T]],
    params: LLMParams | None = None,
    models: Sequence[str] | None = None,
) -> LLMResult[T]:
    """Run `prompt` over `items` in batches of `max_items_per_call`; return one `schema` per item.

    `fallback` is mandatory: a deterministic function producing the same schema without a model.
    """
    raise NotImplementedError("design phase")
