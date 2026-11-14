"""Streamlit client v0: the httpx client against the real app (in-process), view helpers, and a
headless render of the page. Offline: no server, no database (the API calls are faked)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest

from src.api.main import app
from src.schemas import EvalResult, Signal, SignalsResponse
from src.ui.client import ApiClient
from src.ui.views import backtest_summary, equity_frame, signals_table

AS_OF = datetime(2026, 7, 15, 13, tzinfo=UTC)


def resp() -> SignalsResponse:
    return SignalsResponse(
        ticker="AAPL",
        as_of=AS_OF,
        signals=[
            Signal(ticker="AAPL", as_of=AS_OF, strategy="finbert_v0", score=0.3, reasons=["n=2"]),
            Signal(ticker="AAPL", as_of=AS_OF, strategy="momentum", score=-0.1),
        ],
        aggregated=None,
    )


def record() -> EvalResult:
    return EvalResult(
        component="backtest_finbert_v0_validation",
        metric="sharpe",
        value=0.5,
        target=1.3,
        why_this_metric="w",
        gold_set="g",
        n=2,
        split="validation",
        model="rules",
        git_sha="x",
        run_at=AS_OF,
        details={
            "equity_curve": [["2026-06-01", 100000.0], ["2026-06-02", 100500.0]],
            "total_return": 0.005,
            "max_drawdown": 0.0,
            "n_trades": 2,
        },
    )


def test_signals_table_marks_cash_for_non_positive_scores() -> None:
    df = signals_table(resp())
    assert df.to_dict("records") == [
        {"strategy": "finbert_v0", "score": 0.3, "position": "long", "reasons": "n=2"},
        {"strategy": "momentum", "score": -0.1, "position": "cash", "reasons": ""},
    ]


def test_equity_frame_and_summary() -> None:
    df = equity_frame(record())
    assert list(df["equity"]) == [100000.0, 100500.0]
    assert df.index[0].date() == date(2026, 6, 1)
    assert backtest_summary(record())["trades"] == 2


def test_client_talks_to_the_api_over_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client against the real FastAPI app through an in-process ASGI-to-WSGI transport."""
    from fastapi.testclient import TestClient

    tc = TestClient(app)

    def handler(request: httpx.Request) -> httpx.Response:
        r = tc.request(request.method, str(request.url), content=request.content)
        return httpx.Response(r.status_code, content=r.content, headers=r.headers)

    c = ApiClient("http://testserver/api/v1", transport=httpx.MockTransport(handler))
    assert c.health()
    recs = c.evals("simulator")
    assert [r.component for r in recs] == ["simulator"]


def test_client_maps_failures_to_typed_errors() -> None:
    from src.ui.client import ApiError, ApiTimeout, ApiUnreachable

    def boom(kind: type[Exception]):
        def handler(request: httpx.Request) -> httpx.Response:
            raise kind("x", request=request)

        return handler

    def client(handler) -> ApiClient:
        return ApiClient("http://t/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(ApiTimeout):
        client(boom(httpx.ReadTimeout)).watchlist()
    with pytest.raises(ApiUnreachable):
        client(boom(httpx.ConnectError)).watchlist()
    c = client(lambda r: httpx.Response(501, json={"detail": "chat: not implemented"}))
    with pytest.raises(ApiError) as exc:
        c.chat("q")
    assert exc.value.status == 501 and exc.value.degraded and "not implemented" in exc.value.detail
    assert not ApiError(404, "x").degraded
