from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from src.api.routers._common import DB_ERRORS, served
from src.briefing import brief_for
from src.schemas import Brief

router = APIRouter(tags=["brief"])


@router.get(
    "/brief",
    response_model=Brief,
    summary="Morning brief for the watchlist as of day D",
    responses={k: v for k, v in DB_ERRORS.items() if k != 404},
)
def get_brief(
    day: date | None = Query(
        default=None, description="trading day D (cutoff 09:00 ET); default: last validation day"
    ),
) -> Brief:
    return served(lambda: brief_for(day))
