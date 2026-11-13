from __future__ import annotations

from fastapi import APIRouter

from src.api.routers._common import not_implemented
from src.schemas import Watchlist

router = APIRouter(tags=["watchlist"])


@router.get("/watchlist", response_model=Watchlist, summary="Configured watchlist and benchmark")
def get_watchlist() -> Watchlist:
    raise not_implemented("watchlist service")
