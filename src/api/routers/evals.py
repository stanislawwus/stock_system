from __future__ import annotations

from fastapi import APIRouter, Query

from src.evalstore import load_results
from src.schemas import EvalResult

router = APIRouter(tags=["evals"])


@router.get("/evals", response_model=list[EvalResult], summary="Latest eval result per component")
def get_evals(
    component: str | None = Query(default=None, description="e.g. backtest_finbert_v0_validation"),
) -> list[EvalResult]:
    return load_results(component)
