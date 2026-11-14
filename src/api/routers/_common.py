"""Shared helpers for routers: the 501 stub and the service-error -> HTTP mapping."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException

from src.briefing import DataUnavailableError, TestPeriodLockedError, UnknownTickerError

T = TypeVar("T")

DB_ERRORS: dict[int | str, dict] = {
    403: {"description": "as_of is in the test period (AGENTS.md R3)"},
    404: {"description": "ticker not in the universe"},
    503: {"description": "database missing, or busy while make features/collect writes"},
}


def not_implemented(what: str) -> HTTPException:
    return HTTPException(status_code=501, detail=f"{what}: not implemented yet (design phase)")


def served(call: Callable[[], T]) -> T:
    """Run a read-service call; map its errors like GET /signals does (404, 403, 422, 503)."""
    try:
        return call()
    except UnknownTickerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TestPeriodLockedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
