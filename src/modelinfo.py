"""Configured free models and today's budget use for GET /api/v1/models (read-only, R5).

Does not call a provider: it reads config/llm.yaml and the persistent daily counter.
"""

from __future__ import annotations

from src.llm import DailyCounter, load_config
from src.schemas import ModelInfo


def models_info(counter: DailyCounter | None = None) -> list[ModelInfo]:
    cfg = load_config()
    counter = counter or DailyCounter()
    used = counter.used()  # the budget is shared across models: every row shows the same count
    return [
        ModelInfo(
            id=m["id"],
            priority=i + 1,
            use_for=list(m["use_for"]),
            calls_today=used,
            daily_budget=counter.cap,
        )
        for i, m in enumerate(cfg["models"])
    ]
