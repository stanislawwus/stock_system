from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from src.api.routers._common import DB_ERRORS, served
from src.briefing import events_for, news_for, prices_for
from src.schemas import EventsResponse, NewsResponse, PricesResponse

router = APIRouter(tags=["events"])


AS_OF = Query(default=None, description="trading day D (cutoff 09:00 ET); default: last val day")


@router.get(
    "/events/{ticker}",
    response_model=EventsResponse,
    summary="SEC filings with event labels and extracted facts",
    responses=DB_ERRORS,
)
def get_events(
    ticker: str,
    lookback_days: int = Query(default=90, ge=1, le=400),
    as_of: date | None = AS_OF,
) -> EventsResponse:
    return served(lambda: events_for(ticker, lookback_days, as_of))


@router.get(
    "/news/{ticker}",
    response_model=NewsResponse,
    summary="Newest news with sentiment, citable by id, url and available_at",
    responses=DB_ERRORS,
)
def get_news(
    ticker: str,
    lookback_days: int = Query(default=7, ge=1, le=90),
    as_of: date | None = AS_OF,
) -> NewsResponse:
    return served(lambda: news_for(ticker, lookback_days, as_of))


@router.get(
    "/prices/{ticker}",
    response_model=PricesResponse,
    summary="Daily bars that had closed by the cutoff of day D",
    responses=DB_ERRORS,
)
def get_prices(
    ticker: str,
    lookback_days: int = Query(default=120, ge=1, le=800),
    as_of: date | None = AS_OF,
) -> PricesResponse:
    return served(lambda: prices_for(ticker, lookback_days, as_of))
