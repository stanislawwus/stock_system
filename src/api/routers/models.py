from __future__ import annotations

from fastapi import APIRouter

from src.api.routers._common import not_implemented
from src.schemas import ModelInfo

router = APIRouter(tags=["models"])


@router.get(
    "/models", response_model=list[ModelInfo], summary="Configured free models and budget use"
)
def get_models() -> list[ModelInfo]:
    raise not_implemented("llm config reader")
