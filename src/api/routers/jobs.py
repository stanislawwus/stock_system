from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.jobs import jobs
from src.collectors.job import CollectForbidden, check_collect_allowed, collect_job
from src.evalstore import stored_arena
from src.schemas import ArenaRequest, CollectRequest, JobAccepted, JobStatus

router = APIRouter(tags=["jobs"])


@router.post(
    "/collect",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start data collection for the given periods (background job, opt-in)",
    responses={403: {"description": "API collect disabled, or the test period was requested"}},
)
def post_collect(body: CollectRequest) -> JobAccepted:
    try:
        check_collect_allowed(body.periods)
    except CollectForbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return JobAccepted(job_id=jobs.submit("collect", lambda: collect_job(body.periods)))


@router.post(
    "/arena",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Leaderboard of the strategy arena for a split (background job, stored result)",
)
def post_arena(body: ArenaRequest) -> JobAccepted:
    """Serves the leaderboard `make arena` stored; it does not re-run the strategies (D-46)."""
    return JobAccepted(job_id=jobs.submit("arena", lambda: stored_arena(body.split)))


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Status and result of a job",
    responses={404: {"description": "unknown job id"}},
)
def get_job(job_id: str) -> JobStatus:
    found = jobs.status(job_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return found
