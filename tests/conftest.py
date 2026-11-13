"""Test-wide isolation: no network for the LLM gateway, no real budget, cache or traces (R12)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from src import llm


@pytest.fixture(autouse=True)
def _offline_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "llm_cache"))
    monkeypatch.setenv("LLM_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("LLM_DAILY_BUDGET", raising=False)
    monkeypatch.setattr(llm, "BUDGET_PATH", tmp_path / "llm_budget.json")
    llm.reset_run_state()
    llm.set_mock_responder(None)
    yield
    llm.set_mock_responder(None)
    llm.reset_run_state()
