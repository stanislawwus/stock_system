"""FastAPI application. Prefix /api/v1, Swagger at /docs, liveness at /health (AGENTS.md R13)."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routers import ROUTERS

API_PREFIX = "/api/v1"

app = FastAPI(
    title="Stock Awareness Assistant",
    version="0.1.0",
    description="News + SEC filings -> sentiment/events -> impact -> strategy arena -> simulator "
    "-> chat with citations. Every output is a simulation, not investment advice.",
)

for router in ROUTERS:
    app.include_router(router, prefix=API_PREFIX)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}
