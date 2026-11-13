from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from src.api.routers._common import not_implemented
from src.schemas import Brief

router = APIRouter(tags=["brief"])


@router.get("/brief", response_model=Brief, summary="Morning brief for the watchlist as of day D")
def get_brief(
    day: date | None = Query(default=None, description="trading day D; default: today"),
) -> Brief:
    raise not_implemented("brief service")
