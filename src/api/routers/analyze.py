from __future__ import annotations

from fastapi import APIRouter

from src.nlp.service import analyze
from src.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["nlp"])


@router.post(
    "/analyze", response_model=AnalyzeResponse, summary="Arbitrary text -> sentiment (events later)"
)
def post_analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    return analyze(body)
