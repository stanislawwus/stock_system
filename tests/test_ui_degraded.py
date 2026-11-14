"""The UI says "degraded" when the daily LLM budget is used up (D-48); offline, faked client."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.schemas import ModelInfo
from src.ui import views
from src.ui.client import ApiClient, ApiUnreachable

APP = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"


def model(used: int, cap: int) -> ModelInfo:
    return ModelInfo(id="m:free", priority=1, use_for=["chat"], calls_today=used, daily_budget=cap)


def test_budget_rule() -> None:
    assert views.llm_budget_exhausted([model(200, 200)])
    assert views.llm_budget_exhausted([model(0, 0)])  # LLM_DAILY_BUDGET=0
    assert not views.llm_budget_exhausted([model(3, 200)])
    assert not views.llm_budget_exhausted([])


def _app(monkeypatch: pytest.MonkeyPatch, models: object) -> AppTest:
    monkeypatch.setattr(ApiClient, "health", lambda self: True)
    if isinstance(models, Exception):

        def boom(self: ApiClient) -> None:
            raise models  # type: ignore[misc]

        monkeypatch.setattr(ApiClient, "models", boom)
    else:
        monkeypatch.setattr(ApiClient, "models", lambda self: models)
    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_banner_shown_when_budget_is_used_up(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(monkeypatch, [model(0, 0)])
    assert any("Degraded mode" in w.value for w in at.warning)


def test_no_banner_with_budget_left_or_unreachable_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for models in ([model(1, 200)], ApiUnreachable("down")):
        at = _app(monkeypatch, models)
        assert not any("Degraded mode" in w.value for w in at.warning)
