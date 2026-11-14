from __future__ import annotations

from fastapi import APIRouter

from src.modelinfo import models_info
from src.schemas import ModelInfo

router = APIRouter(tags=["models"])


@router.get(
    "/models", response_model=list[ModelInfo], summary="Configured free models and budget use"
)
def get_models() -> list[ModelInfo]:
    return models_info()
