from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from src.api.routers._common import not_implemented
from src.schemas import SignalsResponse

router = APIRouter(tags=["signals"])


@router.get(
    "/signals",
    response_model=SignalsResponse,
    summary="All strategy signals and the aggregated signal for a ticker as of the cutoff of day D",
)
def get_signals(
    ticker: str = Query(min_length=1),
    day: date | None = Query(default=None, description="trading day D; default: today"),
) -> SignalsResponse:
    raise not_implemented("signals service")
