"""Contract tests for the endpoints the Streamlit client uses: /watchlist, /brief, /events/{t},
/news/{t}, /prices/{t}, /evals, /models, /collect, /arena, /jobs/{id}.

Offline: a small DuckDB fixture through DUCKDB_PATH, stored eval files read from a tmp dir.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import briefing, db, evalstore
from src.api.main import API_PREFIX, app
from src.marketview.clock import bar_close_for, cutoff_for
from src.schemas import (
    DISCLAIMER,
    Brief,
    EvalResult,
    EventsResponse,
    JobStatus,
    ModelInfo,
    NewsResponse,
    PricesResponse,
    Watchlist,
)

client = TestClient(app)
DAY = date(2026, 7, 15)  # validation, a Wednesday
URL = API_PREFIX


@pytest.fixture()
def api_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "ui.duckdb"
    days = [d.date() for d in pd.bdate_range("2026-03-02", "2026-07-31")]
    bars = pd.DataFrame(
        [
            (t, d, 100 + i * k, 101 + i * k, 99 + i * k, 100 + i * k, 1, "test", bar_close_for(d))
            for t, k in (("AAPL", 0.2), ("JPM", 0.1), ("SPY", 0.1))
            for i, d in enumerate(days)
        ],
        columns=["t", "d", "o", "h", "l", "c", "v", "s", "a"],
    )  # fmt: skip
    c = cutoff_for(DAY)
    with db.writer_session(path) as con:
        con.register("bars", bars)
        con.execute("INSERT INTO price_bars SELECT * FROM bars")
        for i, h, at, score in [
            ("n1", "Apple beats estimates", c - timedelta(hours=4), 0.8),
            ("n3", "Apple surges after the open", c + timedelta(hours=1), 0.9),  # after cutoff
        ]:
            con.execute(
                "INSERT INTO news VALUES (?, 'AAPL', ?, ?, '', 'http://x/' || ?, 'test', ?)",
                [i, at, h, i, at],
            )
            con.execute(
                "INSERT INTO sentiment VALUES (?, 'AAPL', 1, ?, 'finbert', ?)", [i, score, at]
            )
        at = c - timedelta(days=2)
        con.execute(
            "INSERT INTO filings VALUES ('0001-26-1', 'AAPL', '1', '8-K', ['2.02'], ?, ?, 'u', "
            "NULL, ?)",
            [at, at.date(), at],
        )
        con.execute(  # JPM: an 8-K that was never classified, and one older than the 7-day window
            "INSERT INTO filings VALUES ('0002-26-1', 'JPM', '2', '8-K', ['2.02'], ?, ?, 'u', "
            "NULL, ?)",
            [at, at.date(), at],
        )
        old = c - timedelta(days=9)
        con.execute(
            "INSERT INTO filings VALUES ('0002-26-0', 'JPM', '2', '8-K', ['7.01'], ?, ?, 'u', "
            "NULL, ?)",
            [old, old.date(), old],
        )
        con.execute(
            "INSERT INTO events VALUES ('0001-26-1', 'AAPL', ['2.02'], 0.9, 'rules', ?)", [at]
        )
        con.execute(  # an event on a news item: not a filing event, must not be listed
            "INSERT INTO events VALUES ('n1', 'AAPL', ['8.01'], 0.5, 'rules', ?)",
            [c - timedelta(hours=4)],
        )
        con.execute(
            "INSERT INTO facts VALUES ('AAPL', 'revenue', 9e10, 'USD', ?, 'Q2', 'xbrl', "
            "'0001-26-1', NULL, ?)",
            [date(2026, 6, 27), at],
        )
    monkeypatch.setenv("DUCKDB_PATH", str(path))
    cfg = {
        **briefing.api_config(),
        "brief": {**briefing.api_config()["brief"], "strategy": "finbert_v0"},
    }
    monkeypatch.setattr(briefing, "api_config", lambda: cfg)
    return path


def test_watchlist_matches_config() -> None:
    r = client.get(f"{URL}/watchlist")
    assert r.status_code == 200
    body = Watchlist.model_validate(r.json())
    assert body.benchmark == "SPY"
    assert {i.ticker for i in body.items} >= {"AAPL", "JPM", "XOM"}
    assert {i.kind for i in body.items} <= {"stock", "crypto"}


def test_brief_for_a_day(api_db: Path) -> None:
    r = client.get(f"{URL}/brief", params={"day": DAY.isoformat()})
    assert r.status_code == 200
    body = Brief.model_validate(r.json())
    assert body.disclaimer == DISCLAIMER and body.as_of == cutoff_for(DAY)
    by = {i.ticker: i for i in body.items}
    assert set(by) >= {"AAPL", "JPM"}
    aapl = by["AAPL"]
    assert aapl.last_close is not None and aapl.change_pct is not None
    assert aapl.sentiment_index == pytest.approx(0.8)  # n3 is after the cutoff
    assert aapl.signal is not None and aapl.signal.strategy == "finbert_v0"
    assert [e.source_ref for e in aapl.events] == ["0001-26-1", "n1"]
    assert aapl.filings_7d == 1 and aapl.news_7d == 1  # n3 is after the cutoff
    # regression: an 8-K with no row in the events table must still be counted (all-zero column)
    assert by["JPM"].events == [] and by["JPM"].filings_7d == 1 and by["JPM"].news_7d == 0
    assert (
        by["XOM"].last_close is None and by["XOM"].sentiment_index is None
    )  # no data: empty, not 500


def test_brief_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get(f"{URL}/brief", params={"day": "2026-10-01"}).status_code == 403  # R3
    assert client.get(f"{URL}/brief", params={"day": "not-a-date"}).status_code == 422
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "missing.duckdb"))
    assert client.get(f"{URL}/brief").status_code == 503


def test_events_lists_filing_events_and_facts(api_db: Path) -> None:
    r = client.get(f"{URL}/events/aapl", params={"as_of": DAY.isoformat()})
    assert r.status_code == 200
    body = EventsResponse.model_validate(r.json())
    assert [f.accession_no for f in body.filings] == ["0001-26-1"]
    assert [e.labels for e in body.events] == [["2.02"]]  # the news-item event is not a filing's
    assert [f.name for f in body.facts] == ["revenue"]


def test_events_error_paths(api_db: Path) -> None:
    assert client.get(f"{URL}/events/ZZZZ").status_code == 404
    assert client.get(f"{URL}/events/AAPL", params={"lookback_days": 0}).status_code == 422
    assert client.get(f"{URL}/events/AAPL", params={"as_of": "2026-10-01"}).status_code == 403


def test_news_is_cutoff_bound_and_cites_its_source(api_db: Path) -> None:
    r = client.get(f"{URL}/news/AAPL", params={"as_of": DAY.isoformat()})
    assert r.status_code == 200
    body = NewsResponse.model_validate(r.json())
    assert body.disclaimer == DISCLAIMER
    (entry,) = body.items
    assert entry.news.id == "n1" and entry.news.url == "http://x/n1"
    assert entry.news.available_at <= body.as_of
    assert entry.sentiment is not None and entry.sentiment.score == pytest.approx(0.8)
    assert client.get(f"{URL}/news/ZZZZ").status_code == 404


def test_prices_stop_at_the_cutoff(api_db: Path) -> None:
    r = client.get(f"{URL}/prices/AAPL", params={"as_of": DAY.isoformat(), "lookback_days": 5})
    assert r.status_code == 200
    body = PricesResponse.model_validate(r.json())
    assert len(body.bars) == 5
    assert body.bars[-1].bar_date == DAY - timedelta(days=1)  # the open/close of D are not served
    assert all(b.available_at <= body.as_of for b in body.bars)
    assert client.get(f"{URL}/prices/AAPL", params={"as_of": "2026-10-01"}).status_code == 403


def test_evals_carry_ci_and_the_in_sample_flag() -> None:
    r = client.get(f"{URL}/evals")
    assert r.status_code == 200
    recs = {x["component"]: EvalResult.model_validate(x) for x in r.json()}
    assert {"simulator", "arena", "impact", "backtest_impact_validation"} <= set(recs)
    assert recs["impact"].ci95 is not None and recs["impact"].ci95[0] < recs["impact"].ci95[1]
    bt = recs["backtest_impact_validation"]
    assert bt.in_sample and bt.in_sample_note and bt.ci95 is not None
    assert not recs["backtest_buy_and_hold_validation"].in_sample
    assert recs["arena"].in_sample  # headline aggregate_equal was chosen on val_a
    assert recs["simulator"].ci95 is None  # exact-match fixture: no CI is stored, none invented


def test_eval_enrichment_reads_only_what_the_record_states(tmp_path: Path) -> None:
    base = json.loads((evalstore.RESULTS_DIR / "simulator.json").read_text(encoding="utf-8"))
    (tmp_path / "x.json").write_text(
        json.dumps({**base, "component": "x", "details": {"in_sample": "fit on val_a"}}),
        encoding="utf-8",
    )
    (rec,) = evalstore.load_results(results_dir=tmp_path)
    assert rec.in_sample and rec.in_sample_note == "fit on val_a" and rec.ci95 is None


def test_models_lists_config_and_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import llm, modelinfo

    counter = llm.DailyCounter(tmp_path / "b.json", cap=7)
    counter.consume(3)
    monkeypatch.setattr(modelinfo, "DailyCounter", lambda: counter)
    r = client.get(f"{URL}/models")
    assert r.status_code == 200
    models = [ModelInfo.model_validate(x) for x in r.json()]
    assert [m.priority for m in models] == list(range(1, len(models) + 1))
    assert all(m.calls_today == 3 and m.daily_budget == 7 for m in models)
    assert all(m.id.endswith(":free") for m in models)


def _wait(job_id: str) -> JobStatus:
    for _ in range(100):
        st = JobStatus.model_validate(client.get(f"{URL}/jobs/{job_id}").json())
        if st.status in ("done", "failed"):
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_arena_job_serves_the_stored_leaderboard() -> None:
    r = client.post(f"{URL}/arena", json={"split": "validation"})
    assert r.status_code == 202
    st = _wait(r.json()["job_id"])
    assert st.status == "done" and st.result is not None
    assert st.result["recomputed"] is False
    assert {row["strategy"] for row in st.result["leaderboard"]} >= {"buy_and_hold", "momentum"}


def test_arena_job_without_a_stored_result_fails_cleanly() -> None:
    r = client.post(f"{URL}/arena", json={"split": "train"})  # make arena writes validation only
    st = _wait(r.json()["job_id"])
    assert st.status == "failed" and "make arena" in (st.error or "")


def test_arena_rejects_the_test_split() -> None:
    assert client.post(f"{URL}/arena", json={"split": "test"}).status_code == 422


def test_collect_is_off_by_default_and_never_serves_the_test_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_ENABLE_COLLECT", raising=False)
    assert client.post(f"{URL}/collect", json={"periods": ["train"]}).status_code == 403
    monkeypatch.setenv("API_ENABLE_COLLECT", "1")
    assert client.post(f"{URL}/collect", json={"periods": ["test"]}).status_code == 403


def test_collect_returns_a_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.collectors import job

    monkeypatch.setenv("API_ENABLE_COLLECT", "1")
    seen: list[tuple] = []

    def fake_run(periods, sources, *a, **k):
        seen.append((periods, sources))
        return {"quality": {"failures": []}, "tickers": 1}

    monkeypatch.setattr(job, "run", fake_run)  # no network, no database write
    r = client.post(f"{URL}/collect", json={"periods": ["train"]})
    assert r.status_code == 202
    st = _wait(r.json()["job_id"])
    assert st.status == "done" and seen == [(["train"], ["prices", "news", "filings", "xbrl"])]


def test_unknown_job_is_404() -> None:
    assert client.get(f"{URL}/jobs/nope").status_code == 404
