"""Thin routers: validate -> call a service -> return a schema. No logic here (R13).

Until a service exists its endpoint answers 501 through `_common.not_implemented()`, so the
OpenAPI contract is complete from day one and contract tests pin it.
"""

from __future__ import annotations

from src.api.routers import analyze, brief, chat, evals, events, jobs, models, signals, watchlist

ROUTERS = [
    watchlist.router,
    brief.router,
    signals.router,
    events.router,
    evals.router,
    models.router,
    analyze.router,
    jobs.router,
    chat.router,
]
