"""Streamlit pages rendered headless with a faked API client (offline, no server, no database).

Every page is checked for: the happy path, the empty state, the error state (unreachable API),
the degraded state (API answers 501/503), the disclaimer, and (arena, brief, ticker, evals)
the in-sample marking.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from src import evalstore
from src.schemas import (
    DISCLAIMER,
    AnalyzeResponse,
    Brief,
    BriefItem,
    ChatResponse,
    Citation,
    Event,
    EventsResponse,
    Filing,
    JobStatus,
    ModelInfo,
    NewsEntry,
    NewsItem,
    NewsResponse,
    PriceBar,
    PricesResponse,
    Sentiment,
    Signal,
    SignalsResponse,
    Watchlist,
    WatchlistItem,
)
from src.ui import views
from src.ui.client import ApiClient, ApiError, ApiTimeout, ApiUnreachable

AS_OF = datetime(2026, 8, 31, 13, tzinfo=UTC)
STORED = evalstore.load_results()  # the repo's own stored records: realistic arena / backtests


def sig(strategy: str = "aggregate_equal", score: float = 0.2) -> Signal:
    return Signal(ticker="AAPL", as_of=AS_OF, strategy=strategy, score=score, reasons=["r"])


def brief() -> Brief:
    ev = Event(
        source_ref="a", ticker="AAPL", labels=["2.02"], confidence=0.9, method="rules",
        available_at=AS_OF,
    )  # fmt: skip
    return Brief(
        as_of=AS_OF,
        items=[
            BriefItem(ticker="AAPL", last_close=300.0, change_pct=1.5, sentiment_index=0.2,
                      events=[ev], signal=sig()),
            BriefItem(ticker="XOM", last_close=None, change_pct=None, sentiment_index=None),
        ],
    )  # fmt: skip


def run(fn, monkeypatch: pytest.MonkeyPatch, **api: Any) -> AppTest:
    """Run `fn` (a script that renders one page) with ApiClient methods replaced by `api`."""
    for name, impl in api.items():
        if isinstance(impl, Exception):
            monkeypatch.setattr(ApiClient, name, lambda self, *a, _e=impl, **k: _raise(_e))
        else:
            monkeypatch.setattr(ApiClient, name, lambda self, *a, _v=impl, **k: _v)
    at = AppTest.from_function(fn, default_timeout=30).run()
    assert not at.exception, at.exception
    return at


def _raise(e: Exception) -> None:
    raise e


def texts(at: AppTest) -> str:
    parts = [c.value for c in at.caption] + [m.value for m in at.markdown]
    parts += [x.value for grp in (at.info, at.warning, at.error, at.success) for x in grp]
    return "\n".join(parts)


def has_disclaimer(at: AppTest) -> bool:
    return DISCLAIMER in texts(at)


def page_brief() -> None:
    from src.ui.client import ApiClient
    from src.ui.screens import brief

    brief.render(ApiClient())


def page_ticker() -> None:
    from src.ui.client import ApiClient
    from src.ui.screens import ticker

    ticker.render(ApiClient())


def page_ask() -> None:
    from src.ui.client import ApiClient
    from src.ui.screens import ask

    ask.render(ApiClient())


def page_arena() -> None:
    from src.ui.client import ApiClient
    from src.ui.screens import arena

    arena.render(ApiClient())


def page_analyzer() -> None:
    from src.ui.client import ApiClient
    from src.ui.screens import analyzer

    analyzer.render(ApiClient())


def page_evals() -> None:
    from src.ui.client import ApiClient
    from src.ui.screens import evals

    evals.render(ApiClient())


DEGRADED = ApiError(501, "chat agent: not implemented yet (design phase)")
DOWN = ApiUnreachable("ConnectError")
PAGES = [page_brief, page_ticker, page_ask, page_arena, page_analyzer, page_evals]


# --------------------------------------------------------------------------- brief


def test_brief_happy_path_blank_cells_and_in_sample_note(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run(page_brief, monkeypatch, brief=brief(), evals=STORED)
    df = at.dataframe[0].value
    assert df["ticker"].tolist() == ["AAPL", "XOM"]
    assert df["last close"].isna().tolist() == [False, True]  # no data: blank, not 0
    assert has_disclaimer(at)
    # the default day is the last validation day (val_b): aggregate_equal is not in-sample there
    assert any("Out of sample" in c.value for c in at.caption)


def test_brief_marks_an_in_sample_day(monkeypatch: pytest.MonkeyPatch) -> None:
    def script() -> None:
        import streamlit as st

        st.session_state["x"] = 1
        from src.ui.client import ApiClient
        from src.ui.screens import brief

        brief.render(ApiClient())

    at = run(script, monkeypatch, brief=brief(), evals=STORED)
    at.text_input[0].set_value("2026-06-10").run()  # val_a: the calibrator was chosen here
    assert any("IN-SAMPLE" in w.value for w in at.warning)


def test_brief_empty_error_and_degraded_states(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = Brief(as_of=AS_OF, items=[])
    at = run(page_brief, monkeypatch, brief=empty, evals=STORED)
    assert any("watchlist is empty" in i.value for i in at.info) and has_disclaimer(at)
    at = run(page_brief, monkeypatch, brief=DOWN)
    assert any("not reachable" in e.value for e in at.error)
    at = run(page_brief, monkeypatch, brief=ApiError(503, "database busy"))
    assert any("Degraded" in w.value for w in at.warning) and has_disclaimer(at)
    at = run(page_brief, monkeypatch, brief=ApiError(403, "test period"))
    assert any("403" in e.value for e in at.error)


# --------------------------------------------------------------------------- ticker


def _bars() -> list[PriceBar]:
    return [
        PriceBar(ticker="AAPL", bar_date=date(2026, 8, d), open=1, high=2, low=1, close=1.5 + d,
                 volume=1, source="test", available_at=AS_OF)
        for d in (27, 28)
    ]  # fmt: skip


def ticker_api() -> dict[str, Any]:
    n = NewsItem(id="abc1234567890", ticker="AAPL", published_at=AS_OF, headline="Apple [beats]",
                 url="http://x/y", source="test", available_at=AS_OF)  # fmt: skip
    s = Sentiment(source_ref="abc1234567890", ticker="AAPL", label=1, score=0.7, method="finbert",
                  available_at=AS_OF)  # fmt: skip
    f = Filing(accession_no="0001-1", ticker="AAPL", cik="1", form="8-K", items=["2.02"],
               accepted_at=AS_OF, filed_date=date(2026, 8, 30), url="http://sec/1",
               available_at=AS_OF)  # fmt: skip
    return {
        "watchlist": Watchlist(items=[WatchlistItem(ticker="AAPL")], benchmark="SPY"),
        "prices": PricesResponse(ticker="AAPL", as_of=AS_OF, bars=_bars()),
        "signals": SignalsResponse(ticker="AAPL", as_of=AS_OF, signals=[sig("impact", 0.3)],
                                   aggregated=None),
        "events": EventsResponse(ticker="AAPL", filings=[f], events=[], facts=[]),
        "news": NewsResponse(ticker="AAPL", as_of=AS_OF, items=[NewsEntry(news=n, sentiment=s)]),
        "evals": STORED,
    }  # fmt: skip


def test_ticker_page_cites_news_and_lists_filings(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run(page_ticker, monkeypatch, **ticker_api())
    md = "\n".join(m.value for m in at.markdown)
    assert "[Apple (beats)](http://x/y)" in md and "id `abc1234567`" in md and "UTC" in md
    assert at.dataframe[0].value["strategy"].tolist() == ["impact"]  # signals table
    assert has_disclaimer(at)


def test_ticker_page_flags_an_in_sample_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    val_a = datetime(2026, 6, 10, 13, tzinfo=UTC)  # 'impact' was calibrated on val_a
    s = Signal(ticker="AAPL", as_of=val_a, strategy="impact", score=0.3)
    api = ticker_api()
    api["signals"] = SignalsResponse(ticker="AAPL", as_of=val_a, signals=[s], aggregated=None)
    at = run(page_ticker, monkeypatch, **api)
    assert any("IN-SAMPLE" in w.value for w in at.warning)


def test_ticker_page_empty_and_degraded_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    api = ticker_api()
    api["prices"] = PricesResponse(ticker="AAPL", as_of=AS_OF, bars=[])
    api["news"] = NewsResponse(ticker="AAPL", as_of=AS_OF, items=[])
    api["events"] = EventsResponse(ticker="AAPL", filings=[], events=[], facts=[])
    api["signals"] = ApiError(503, "database busy")
    at = run(page_ticker, monkeypatch, **api)
    infos = " ".join(i.value for i in at.info)
    assert "No price bars" in infos and "No news" in infos and "No filings" in infos
    assert any("Degraded" in w.value for w in at.warning) and has_disclaimer(at)


def test_ticker_page_without_a_watchlist_falls_back_to_a_text_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ticker_api()
    api["watchlist"] = DOWN
    at = run(page_ticker, monkeypatch, **api)
    assert any("not reachable" in e.value for e in at.error)
    assert at.text_input[0].value == "AAPL"


# --------------------------------------------------------------------------- ask


def chat_ok() -> ChatResponse:
    c = Citation(source_ref="n1", kind="news", available_at=AS_OF, quote="Apple beat")
    return ChatResponse(
        answer="Apple reported on the same day.", citations=[c], method="rules", intent="why_move"
    )


def test_ask_shows_a_degraded_state_while_chat_is_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = run(page_ask, monkeypatch, chat=DEGRADED)
    assert any("No questions yet" in i.value for i in at.info)  # empty state
    at.text_area[0].set_value("What happened to AAPL?").run()
    at.button[0].click().run()
    assert any("Degraded" in w.value for w in at.warning)
    assert any("not implemented" in w.value for w in at.warning) and has_disclaimer(at)


def test_ask_renders_citations_and_warns_about_uncited_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = run(page_ask, monkeypatch, chat=chat_ok())
    at.text_area[0].set_value("What happened?").run()
    at.button[0].click().run()
    md = "\n".join(m.value for m in at.markdown)
    assert "`news` `n1`" in md and "Apple beat" in md and DISCLAIMER in texts(at)
    bare = chat_ok().model_copy(update={"citations": []})
    at = run(page_ask, monkeypatch, chat=bare)
    at.text_area[0].set_value("What happened?").run()
    at.button[0].click().run()
    assert any("no citations" in w.value for w in at.warning)


def test_ask_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run(page_ask, monkeypatch, chat=DOWN)
    at.text_area[0].set_value("q").run()
    at.button[0].click().run()
    assert any("not reachable" in e.value for e in at.error)


# --------------------------------------------------------------------------- arena


def test_arena_table_has_cis_flags_curves_and_a_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run(page_arena, monkeypatch, evals=STORED)
    board = at.dataframe[0].value
    assert "Sharpe 95% CI" in board.columns and "val_b 95% CI" in board.columns
    flagged = board.loc[board["in-sample on validation"] == "IN-SAMPLE", "strategy"].tolist()
    assert "aggregate_equal" in flagged and "buy_and_hold" not in flagged
    assert any("IN-SAMPLE rows" in w.value for w in at.warning)
    assert len(at.select_slider) == 1  # the replay control
    assert has_disclaimer(at)


def test_arena_refresh_button_reads_the_job_result(monkeypatch: pytest.MonkeyPatch) -> None:
    done = JobStatus(job_id="j", status="done", result={
        "source": "evals/results/arena.json", "recomputed": False, "leaderboard": [],
    })  # fmt: skip
    at = run(page_arena, monkeypatch, evals=STORED, run_arena=done)
    at.button[0].click().run()
    assert any("recomputed: False" in s.value for s in at.success)
    failed = JobStatus(job_id="j", status="failed", error="LookupError: no stored arena")
    at = run(page_arena, monkeypatch, evals=STORED, run_arena=failed)
    at.button[0].click().run()
    assert any("Arena job failed" in e.value for e in at.error)


def test_arena_empty_error_and_degraded_states(monkeypatch: pytest.MonkeyPatch) -> None:
    no_arena = [r for r in STORED if r.component != "arena"]
    at = run(page_arena, monkeypatch, evals=no_arena)
    assert any("make arena" in i.value for i in at.info) and has_disclaimer(at)
    at = run(page_arena, monkeypatch, evals=DOWN)
    assert any("not reachable" in e.value for e in at.error)
    at = run(page_arena, monkeypatch, evals=ApiError(503, "x"))
    assert any("Degraded" in w.value for w in at.warning)


# --------------------------------------------------------------------------- analyzer


def analyze_ok() -> AnalyzeResponse:
    s = Sentiment(source_ref="adhoc", ticker="BA", label=-1, score=-0.8, method="lexicon",
                  available_at=AS_OF)  # fmt: skip
    return AnalyzeResponse(sentiment=s, facts=[], method="lexicon")


def test_analyzer_states(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run(page_analyzer, monkeypatch, analyze=analyze_ok())
    assert any("Enter some text" in i.value for i in at.info) and has_disclaimer(at)
    assert at.button[0].disabled  # nothing to analyze yet
    at.text_area[0].set_value("Boeing shares plunge").run()
    at.button[0].click().run()
    assert [m.value for m in at.metric] == ["negative", "-0.800", "lexicon"]
    assert any("event classifier" in i.value for i in at.info)
    at = run(page_analyzer, monkeypatch, analyze=ApiError(422, "bad"))
    at.text_area[0].set_value("x").run()
    at.button[0].click().run()
    assert any("422" in e.value for e in at.error)


# --------------------------------------------------------------------------- evals


def test_evals_table_has_target_value_ci_and_in_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    models = [ModelInfo(id="m:free", priority=1, use_for=["chat"], calls_today=2, daily_budget=9)]
    at = run(page_evals, monkeypatch, evals=STORED, models=models)
    table = at.dataframe[0].value
    assert {"value", "target", "95% CI", "in-sample", "meets target"} <= set(table.columns)
    assert (table["in-sample"] == "IN-SAMPLE").any() and (table["95% CI"] != "n/a").any()
    assert not any(c.startswith("backtest_") for c in table["component"])  # hidden by default
    assert at.dataframe[-1].value["id"].tolist() == ["m:free"]
    assert has_disclaimer(at)


def test_evals_empty_error_and_degraded_models(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run(page_evals, monkeypatch, evals=[])
    assert any("make eval" in i.value for i in at.info) and has_disclaimer(at)
    at = run(page_evals, monkeypatch, evals=DOWN)
    assert any("not reachable" in e.value for e in at.error)
    at = run(page_evals, monkeypatch, evals=STORED, models=ApiError(503, "x"))
    assert any("Degraded" in w.value for w in at.warning)
    at = run(page_evals, monkeypatch, evals=STORED, models=[])
    assert any("No models" in i.value for i in at.info)


# --------------------------------------------------------------------------- app and helpers


@pytest.mark.parametrize("page", PAGES, ids=lambda f: f.__name__)
def test_every_page_survives_an_unreachable_api(page, monkeypatch: pytest.MonkeyPatch) -> None:
    down = {n: DOWN for n in ("brief", "watchlist", "prices", "signals", "events", "news",
                              "evals", "models", "chat", "analyze", "run_arena")}  # fmt: skip
    at = run(page, monkeypatch, **down)
    assert not at.exception
    if page not in (page_analyzer, page_ask):  # these only call the API on a click
        assert any("not reachable" in e.value for e in at.error)
    assert has_disclaimer(at)


def test_app_entry_shows_the_disclaimer_and_stops_when_the_api_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ApiClient, "health", lambda self: False)
    page = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception
    assert any(DISCLAIMER in c.value for c in at.caption)
    assert any("API not reachable" in e.value for e in at.error)


def test_app_entry_renders_the_default_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ApiClient, "health", lambda self: True)
    monkeypatch.setattr(ApiClient, "brief", lambda self, d=None: brief())
    monkeypatch.setattr(ApiClient, "evals", lambda self, c=None: STORED)
    page = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception
    assert at.dataframe[0].value["ticker"].tolist() == ["AAPL", "XOM"]


def test_sample_status_reads_only_stored_records() -> None:
    recs = views.by_component(STORED)
    assert views.sample_status(date(2026, 6, 10), "aggregate_equal", recs)[0] is True
    assert views.sample_status(date(2026, 8, 31), "aggregate_equal", recs)[0] is False
    assert views.sample_status(date(2026, 8, 31), "buy_and_hold", recs)[0] is False
    assert views.sample_status(date(2026, 3, 10), "impact", recs)[0] is True  # train-fitted
    assert views.sample_status(date(2026, 8, 31), "nope", recs)[0] is None
    assert views.sample_status(date(2026, 8, 31), "x", {})[0] is None


def test_equity_curves_rebase_and_replay() -> None:
    recs = [r for r in STORED if r.component.endswith("_validation") and "backtest" in r.component]
    curves = views.equity_curves(recs, ["buy_and_hold", "momentum"])
    assert set(curves.columns) == {"buy_and_hold", "momentum"}
    assert (curves.iloc[0] == 1.0).all()
    t = views.replay_table(curves, curves.index[5])
    assert len(t) == 2 and (t["max drawdown so far"] <= 0).all()


def test_a_timeout_is_not_reported_as_an_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run(page_analyzer, monkeypatch, analyze=ApiTimeout("ReadTimeout"))
    at.text_area[0].set_value("x").run()
    at.button[0].click().run()
    assert any("did not answer in time" in e.value for e in at.error)
    assert not any("not reachable" in e.value for e in at.error)


def test_ask_shows_a_refusal_as_a_refusal_not_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = ChatResponse(answer="I cannot give advice.", citations=[], method="rules",
                     intent="refuse", refused=True, refusal_reason="investment advice")  # fmt: skip
    at = run(page_ask, monkeypatch, chat=r)
    at.text_area[0].set_value("Should I buy NVDA?").run()
    at.button[0].click().run()
    assert any("investment advice" in i.value for i in at.info)
    assert not any("no citations" in w.value for w in at.warning)
