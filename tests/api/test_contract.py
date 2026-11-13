"""API contract (AGENTS.md R13): the OpenAPI stub pins paths, methods, status codes and schemas."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import API_PREFIX, app
from src.schemas import DISCLAIMER

client = TestClient(app)
design_phase = pytest.mark.xfail(strict=True, reason="design phase: service not implemented")

CONTRACT = {
    ("get", "/watchlist"): 200,
    ("get", "/brief"): 200,
    ("get", "/signals"): 200,
    ("get", "/events/{ticker}"): 200,
    ("get", "/evals"): 200,
    ("get", "/models"): 200,
    ("post", "/analyze"): 200,
    ("post", "/collect"): 202,
    ("post", "/arena"): 202,
    ("post", "/chat"): 200,
    ("get", "/jobs/{job_id}"): 200,
}


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_openapi_declares_every_endpoint_with_its_success_code() -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for (method, path), code in CONTRACT.items():
        full = API_PREFIX + path
        assert full in paths, f"missing {full}"
        assert method in paths[full], f"missing {method.upper()} {full}"
        assert str(code) in paths[full][method]["responses"], f"{method} {full} must declare {code}"


def test_signal_bearing_schemas_carry_the_disclaimer() -> None:
    comps = client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("SignalsResponse", "Brief", "AnalyzeResponse", "ChatResponse"):
        prop = comps[name]["properties"]["disclaimer"]
        assert prop["default"] == DISCLAIMER, name


def test_validation_errors_are_422_before_any_service_runs() -> None:
    assert client.post(f"{API_PREFIX}/analyze", json={"text": ""}).status_code == 422
    assert client.post(f"{API_PREFIX}/collect", json={"periods": ["future"]}).status_code == 422
    assert client.get(f"{API_PREFIX}/signals").status_code == 422  # ticker is required


def test_unimplemented_endpoints_answer_501_not_500() -> None:
    assert client.get("/api/v1/watchlist").status_code == 501


@design_phase
def test_collect_returns_a_job_id() -> None:
    r = client.post(f"{API_PREFIX}/collect", json={"periods": ["train"]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    status = client.get(f"{API_PREFIX}/jobs/{job_id}").json()["status"]
    assert status in {"queued", "running", "done"}


@design_phase
def test_watchlist_matches_config() -> None:
    body = client.get(f"{API_PREFIX}/watchlist").json()
    assert body["benchmark"] == "SPY"
    assert {i["ticker"] for i in body["items"]} >= {"AAPL", "JPM", "XOM"}
