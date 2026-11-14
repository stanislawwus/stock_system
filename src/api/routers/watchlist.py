from __future__ import annotations

from fastapi import APIRouter

from src.briefing import watchlist
from src.schemas import Watchlist

router = APIRouter(tags=["watchlist"])


@router.get("/watchlist", response_model=Watchlist, summary="Configured watchlist and benchmark")
def get_watchlist() -> Watchlist:
    return watchlist()
