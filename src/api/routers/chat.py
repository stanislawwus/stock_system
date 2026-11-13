from __future__ import annotations

from fastapi import APIRouter

from src.api.routers._common import not_implemented
from src.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Question -> answer with citations")
def post_chat(body: ChatRequest) -> ChatResponse:
    raise not_implemented("chat agent")
