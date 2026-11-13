from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.routers._common import not_implemented
from src.schemas import EventsResponse

router = APIRouter(tags=["events"])


@router.get(
    "/events/{ticker}",
    response_model=EventsResponse,
    summary="8-K filings with event labels and extracted facts",
)
def get_events(ticker: str, lookback_days: int = Query(default=90, ge=1, le=400)) -> EventsResponse:
    raise not_implemented("events service")
