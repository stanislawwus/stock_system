from __future__ import annotations

from fastapi import APIRouter

from src.api.routers._common import not_implemented
from src.schemas import EvalResult

router = APIRouter(tags=["evals"])


@router.get("/evals", response_model=list[EvalResult], summary="Latest eval result per component")
def get_evals() -> list[EvalResult]:
    raise not_implemented("evals reader")
