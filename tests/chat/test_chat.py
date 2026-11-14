"""Chat agent (D-46): tools, fixed plan, template composer, refusals, injection, cached rephrase,
code checks and the /chat contract. Offline: fixture DuckDB from evals/chat_fixtures.py."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evals import chat_fixtures as fx
from src import llm
from src.api.main import API_PREFIX, app
from src.chat import agent, checks, tools
from src.schemas import DISCLAIMER, ChatResponse, Citation

Q_WHY = "Why did AAPL move on 2026-07-15?"


@pytest.fixture()
def chat_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = fx.build(tmp_path / "chat.duckdb")
    monkeypatch.setenv("DUCKDB_PATH", str(path))
    return path


@pytest.fixture()
def poisoned_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = fx.build(tmp_path / "poisoned.duckdb", poisoned=True)
    monkeypatch.setenv("DUCKDB_PATH", str(path))
    ranking = tools.config()["ranking"]
    old = ranking["top_news"]
    ranking["top_news"] = 10  # show every poisoned headline
    yield path
    ranking["top_news"] = old


@pytest.fixture()
def no_provider(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail the test if anything reaches the provider transport (the chat path must not)."""
    calls: list[str] = []

    def boom(model: str, *a: object, **k: object) -> str:
        calls.append(model)
        raise AssertionError("the chat path must not call a model")

    monkeypatch.setattr(llm, "_transport", boom)
    return calls


# --------------------------------------------------------------------------- tools


def test_price_move_and_abnormal_return(chat_db: Path) -> None:
    r = agent.answer(Q_WHY)
    m = r.tool_outputs["get_prices"]
    assert m["ret"] == pytest.approx(0.05) and m["bench_ret"] == pytest.approx(-0.01)
    assert m["abnormal_ret"] == pytest.approx(0.06)
    assert m["prev_day"] == "2026-07-14"


def test_news_window_relevance_and_ranking(chat_db: Path) -> None:
    r = agent.answer(Q_WHY)
    ids = [n["id"] for n in r.tool_outputs["get_news"]]
    assert ids == ["n_beat", "n_probe"]  # n_late after close, n_old before, n_wrap irrelevant
    scores = [n["score"] for n in r.tool_outputs["get_news"]]
    assert scores == sorted(scores, reverse=True)
    assert [f["accession_no"] for f in r.tool_outputs["get_events"]] == [fx.ACCESSION]


def test_backtest_summary_reads_the_committed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = {
        "target": 1.2,
        "details": {
            "start": "2026-06-01",
            "end": "2026-08-31",
            "total_return": 0.05,
            "sharpe": 1.5,
            "max_drawdown": 0.02,
            "n_trades": 7,
        },
    }
    (tmp_path / "backtest_momentum_validation.json").write_text(json.dumps(rec), "utf-8")
    monkeypatch.setattr(tools, "RESULTS_DIR", tmp_path)
    s = tools.run_backtest_summary("momentum")
    assert s is not None and s.sharpe == 1.5 and s.benchmark_sharpe == 1.2
    assert tools.run_backtest_summary("momentum", "test") is None  # never the test period (R3)
    r = agent.answer("How did the momentum strategy perform in the backtest?")
    assert r.response.tools_used == ["run_backtest_summary"]
    assert "Sharpe 1.50 vs 1.20" in r.response.answer
    assert checks.numbers_ok(r.response.answer, r.tool_outputs)


# --------------------------------------------------------------------------- composer


def test_why_move_answer_is_built_from_tool_outputs_only(chat_db: Path) -> None:
    r = agent.answer(Q_WHY)
    resp = r.response
    assert resp.intent == "why_move" and resp.ticker == "AAPL" and resp.method == "rules"
    assert resp.tools_used == agent.PLANS["why_move"]
    a = resp.answer
    assert "closed at 105.00 on 2026-07-15, up 5.00% from 100.00" in a
    assert "abnormal return of AAPL vs SPY was +6.00 percentage points" in a
    assert "The whole market fell that day (SPY down 1.00%)" in a
    assert "coincided in time" in a and a.endswith(DISCLAIMER)
    assert checks.numbers_ok(a, r.tool_outputs, Q_WHY)
    assert checks.citation_markers_ok(a, resp.citations)
    assert checks.wording_ok(a)
    assert [(c.n, c.kind, c.source_ref) for c in resp.citations] == [
        (1, "news", "n_beat"),
        (2, "news", "n_probe"),
        (3, "filing", fx.ACCESSION),
    ]
    assert all(c.ticker == "AAPL" for c in resp.citations)
    assert f"{llm.DATA_OPEN} Apple beats estimates on iPhone demand {llm.DATA_CLOSE}" in a


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Why did Apple jump on July 15?", "why_move"),
        ("What was the close of AAPL on 2026-07-15?", "price"),
        ("Show me news for AAPL on 2026-07-15", "news"),
        ("What is the signal for AAPL on 2026-07-15?", "signals"),
        ("How did the momentum strategy perform in the backtest?", "backtest"),
    ],
)
def test_intent_and_fixed_plan(question: str, intent: str) -> None:
    p = agent.parse(question)
    assert p.intent == intent
    if intent not in ("backtest",):
        assert p.ticker == "AAPL"


def test_news_intent_cites_only_the_ticker_s_items_of_that_day(chat_db: Path) -> None:
    r = agent.answer("Show me news for AAPL on 2026-07-15")
    assert r.response.tools_used == ["get_news", "get_events"]
    refs = {c.source_ref for c in r.response.citations}
    assert refs == {"n_beat", "n_probe", "n_late", fx.ACCESSION}  # n_late: same ET day
    assert checks.numbers_ok(r.response.answer, r.tool_outputs, "2026-07-15")


def test_weekend_day_uses_the_last_session(chat_db: Path) -> None:
    r = agent.answer("Why did AAPL move on 2026-07-18?")  # Saturday
    assert "2026-07-18 was not a trading session; using 2026-07-17" in r.response.answer
    assert checks.numbers_ok(r.response.answer, r.tool_outputs, "2026-07-18")


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("Why did XYZQ move on 2026-07-15?", "unknown_ticker"),
        ("Why did $ZZZZ drop?", "unknown_ticker"),
        ("Which stock is guaranteed to make me a profit?", "guaranteed_profit"),
        ("Give me a risk-free profit trade on AAPL", "guaranteed_profit"),
        ("Why did the market move?", "no_ticker"),
        ("Why did AAPL move on 2026-10-01?", "test_period"),
    ],
)
def test_refusals(chat_db: Path, no_provider: list[str], question: str, reason: str) -> None:
    r = agent.answer(question)
    assert r.response.refused and r.response.refusal_reason == reason
    assert r.response.intent == "refuse" and r.response.citations == []
    assert r.response.tools_used == [] and r.response.answer.endswith(DISCLAIMER)


# --------------------------------------------------------------------------- injection (R6)


def test_poisoned_headlines_stay_quoted_data(poisoned_db: Path, no_provider: list[str]) -> None:
    r = agent.answer(Q_WHY)
    a = r.response.answer
    body = checks.outside_quotes(a)
    for marker in fx.POISON_MARKERS:
        assert marker not in body, marker
    assert a.count(llm.DATA_OPEN) == a.count(llm.DATA_CLOSE) == 7  # 2 clean + 5 poisoned
    assert "<script>" not in a
    # same verdict and numbers as the clean market
    assert r.tool_outputs["get_prices"]["abnormal_ret"] == pytest.approx(0.06)
    assert checks.numbers_ok(a, r.tool_outputs, Q_WHY) and checks.wording_ok(a)
    assert all(c.ticker == "AAPL" for c in r.response.citations)
    assert "m_msft" not in {c.source_ref for c in r.response.citations}
    assert no_provider == []


# --------------------------------------------------------------------------- planted errors


def test_checks_catch_a_wrong_number_and_a_foreign_citation(chat_db: Path) -> None:
    r = agent.answer(Q_WHY)
    a = r.response.answer
    wrong = a.replace("105.00", "106.37", 1)
    assert not checks.numbers_ok(wrong, r.tool_outputs, Q_WHY)
    assert checks.unsupported_numbers(wrong, r.tool_outputs, Q_WHY) == ["106.37"]
    extra = Citation(
        n=9,
        source_ref="m_msft",
        kind="news",
        ticker="MSFT",
        available_at=fx._t(fx.DAY, 13),
        quote="x",
    )
    assert not checks.citation_markers_ok(a, [*r.response.citations, extra])
    assert not checks.wording_ok(a.replace("coincided in time with", "caused"))
    assert not checks.wording_ok(a.replace(DISCLAIMER, ""))


# --------------------------------------------------------------------------- cached rephrase


def _seed_cache(template: str, rephrased: str) -> None:
    cfg = llm.load_config()
    params = llm.LLMParams(max_tokens=int(cfg["max_tokens"]))
    model = llm._select_models("chat", None, cfg)[0]
    items = [{"answer": template}]
    key = llm.DiskCache.key(model, agent.REPHRASE_PROMPT, params, agent.Rephrased, items)
    llm.DiskCache(llm._cache_dir(cfg)).put(key, [{"answer": rephrased}])


def test_rephrase_reads_the_cache_only_and_is_guarded(
    chat_db: Path, monkeypatch: pytest.MonkeyPatch, no_provider: list[str]
) -> None:
    template = agent.answer(Q_WHY).response.answer.removesuffix("\n" + DISCLAIMER)
    monkeypatch.setenv("CHAT_REPHRASE", "1")
    # no cache entry: template, and no provider call
    assert agent.answer(Q_WHY).response.method == "rules"
    # a faithful rephrase is used
    good = "In short: " + template
    _seed_cache(template, good)
    r = agent.answer(Q_WHY)
    assert r.response.method == "llm" and r.response.answer.startswith("In short:")
    # a rephrase that changes a number is rejected -> template
    _seed_cache(template, template.replace("105.00", "107.00"))
    assert agent.answer(Q_WHY).response.method == "rules"
    _seed_cache(template, template.replace("coincided in time with", "was caused by"))
    assert agent.answer(Q_WHY).response.method == "rules"
    assert no_provider == []


# --------------------------------------------------------------------------- API contract


def test_post_chat_contract(chat_db: Path) -> None:
    client = TestClient(app)
    r = client.post(f"{API_PREFIX}/chat", json={"question": Q_WHY})
    assert r.status_code == 200
    body = ChatResponse.model_validate(r.json())
    assert body.disclaimer == DISCLAIMER and len(body.citations) == 3
    r = client.post(f"{API_PREFIX}/chat", json={"question": "guaranteed profit please"})
    assert r.status_code == 200 and r.json()["refused"] is True
    assert client.post(f"{API_PREFIX}/chat", json={"question": ""}).status_code == 422


def test_post_chat_without_database_is_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "missing.duckdb"))
    r = TestClient(app).post(f"{API_PREFIX}/chat", json={"question": Q_WHY})
    assert r.status_code == 503
