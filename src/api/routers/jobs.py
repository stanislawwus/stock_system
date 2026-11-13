from __future__ import annotations

from fastapi import APIRouter, status

from src.api.routers._common import not_implemented
from src.schemas import ArenaRequest, CollectRequest, JobAccepted, JobStatus

router = APIRouter(tags=["jobs"])


@router.post(
    "/collect",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start data collection for the given periods (background job)",
)
def post_collect(body: CollectRequest) -> JobAccepted:
    raise not_implemented("collect job")


@router.post(
    "/arena",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run every strategy on a split and refresh the leaderboard (background job)",
)
def post_arena(body: ArenaRequest) -> JobAccepted:
    raise not_implemented("arena job")


@router.get("/jobs/{job_id}", response_model=JobStatus, summary="Status and result of a job")
def get_job(job_id: str) -> JobStatus:
    raise not_implemented("job store")
