from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query

from src.schemas import SignalsResponse
from src.strategies.service import TestPeriodLockedError, UnknownTickerError, signals_for

router = APIRouter(tags=["signals"])


@router.get(
    "/signals",
    response_model=SignalsResponse,
    summary="Every strategy's signal for a ticker as of the 09:00 ET cutoff of day D",
    responses={
        403: {"description": "as_of is in the test period (AGENTS.md R3)"},
        404: {"description": "ticker not in the universe"},
        503: {"description": "database missing: run make collect-dev and make features"},
    },
)
def get_signals(
    ticker: str = Query(min_length=1, max_length=10),
    as_of: date | datetime | None = Query(
        default=None,
        description="trading day D (cutoff 09:00 ET of D) or a tz-aware instant; "
        "default: the last validation day",
    ),
) -> SignalsResponse:
    try:
        return signals_for(ticker, as_of)
    except UnknownTickerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TestPeriodLockedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
