from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.chat.agent import DataUnavailableError, answer
from src.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Question -> answer from read-only tools with numbered citations (no LLM in the path)",
    responses={503: {"description": "database missing, or busy while a writer holds it"}},
)
def post_chat(body: ChatRequest) -> ChatResponse:
    """Refusals (unknown ticker, guaranteed profit, test period) are 200 with `refused=true`."""
    try:
        return answer(body.question).response
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
