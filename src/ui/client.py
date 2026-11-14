"""httpx client for the API: the only way the Streamlit UI reads anything (AGENTS.md R13).

Failures arrive as two exceptions so a page can show an honest state instead of a traceback:
`ApiUnreachable` (no server / timeout) and `ApiError` (the API answered with a non-2xx status;
501 and 503 mean "degraded": the feature or its data is not available right now).
"""

from __future__ import annotations

import os
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from src.schemas import (
    AnalyzeResponse,
    Brief,
    ChatResponse,
    EvalResult,
    EventsResponse,
    JobStatus,
    ModelInfo,
    NewsResponse,
    PricesResponse,
    SignalsResponse,
    Watchlist,
)

M = TypeVar("M", bound=BaseModel)
DEGRADED_STATUSES = frozenset({501, 503})
JOB_POLL_SECONDS = 0.5
SLOW_TIMEOUT = 180.0  # POST /analyze: the first call loads the FinBERT weights


class ApiUnreachable(RuntimeError):
    """The API did not answer (connection refused, timeout)."""


class ApiTimeout(ApiUnreachable):
    """The API accepted the connection but did not answer in time (e.g. a model cold start)."""


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail

    @property
    def degraded(self) -> bool:
        return self.status in DEGRADED_STATUSES


def _detail(r: httpx.Response) -> str:
    try:
        d = r.json().get("detail", r.text)
    except (ValueError, AttributeError):
        return r.text[:300]
    return d if isinstance(d, str) else str(d)[:300]


class ApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("API_URL", "http://127.0.0.1:8000/api/v1")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def _send(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.TimeoutException as exc:
            raise ApiTimeout(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ApiUnreachable(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise ApiError(r.status_code, _detail(r))
        return r.json()

    def _get(self, path: str, model: type[M], **params: Any) -> M:
        clean = {k: v for k, v in params.items() if v is not None}
        return model.model_validate(self._send("GET", path, params=clean))

    def health(self) -> bool:
        try:
            return self._http.get("/health", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def watchlist(self) -> Watchlist:
        return self._get("/watchlist", Watchlist)

    def brief(self, day: str | None = None) -> Brief:
        return self._get("/brief", Brief, day=day)

    def signals(self, ticker: str, as_of: str | None = None) -> SignalsResponse:
        """`as_of`: trading day D (YYYY-MM-DD) or a tz-aware ISO instant; None = API default."""
        return self._get("/signals", SignalsResponse, ticker=ticker, as_of=as_of)

    def events(self, ticker: str, as_of: str | None = None) -> EventsResponse:
        return self._get(f"/events/{ticker}", EventsResponse, as_of=as_of)

    def news(self, ticker: str, as_of: str | None = None) -> NewsResponse:
        return self._get(f"/news/{ticker}", NewsResponse, as_of=as_of)

    def prices(self, ticker: str, as_of: str | None = None) -> PricesResponse:
        return self._get(f"/prices/{ticker}", PricesResponse, as_of=as_of)

    def evals(self, component: str | None = None) -> list[EvalResult]:
        raw = self._send("GET", "/evals", params={"component": component} if component else None)
        return [EvalResult.model_validate(x) for x in raw]

    def models(self) -> list[ModelInfo]:
        return [ModelInfo.model_validate(x) for x in self._send("GET", "/models")]

    def analyze(self, text: str, ticker: str | None = None) -> AnalyzeResponse:
        body = {"text": text, "ticker": ticker}
        raw = self._send("POST", "/analyze", json=body, timeout=SLOW_TIMEOUT)
        return AnalyzeResponse.model_validate(raw)

    def chat(self, question: str) -> ChatResponse:
        return ChatResponse.model_validate(self._send("POST", "/chat", json={"question": question}))

    def job(self, job_id: str) -> JobStatus:
        return JobStatus.model_validate(self._send("GET", f"/jobs/{job_id}"))

    def run_arena(self, split: str = "validation", timeout: float = 30.0) -> JobStatus:
        """POST /arena, then poll the job until it is done or failed (or `timeout` seconds pass)."""
        job_id = self._send("POST", "/arena", json={"split": split})["job_id"]
        deadline = time.monotonic() + timeout
        while True:
            st = self.job(job_id)
            if st.status in ("done", "failed") or time.monotonic() > deadline:
                return st
            time.sleep(JOB_POLL_SECONDS)
