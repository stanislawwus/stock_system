from __future__ import annotations

from fastapi import APIRouter

from src.api.routers._common import not_implemented
from src.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["nlp"])


@router.post(
    "/analyze", response_model=AnalyzeResponse, summary="Arbitrary text -> sentiment, events, facts"
)
def post_analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    raise not_implemented("analyze service")
