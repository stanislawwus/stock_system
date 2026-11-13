"""Finnhub news and daily prices: parsing, window splitting, dedup, available_at (D-24)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src import db
from src.collectors import news, prices
from src.collectors.common import Period
from src.marketview.clock import bar_close_for

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "finnhub" / "company_news.json"
VALIDATION = Period("validation", date(2026, 6, 1), date(2026, 8, 31))


def _items() -> list[dict]:
    return json.loads(FIX.read_text(encoding="utf-8"))


def test_news_rows_use_provider_datetime_and_drop_bad_items() -> None:
    rows = news.parse_items(_items(), "AAPL", date(2026, 7, 1), date(2026, 7, 31))
    assert [r["headline"] for r in rows] == [
        "Apple beats on revenue",
        "Apple supplier news",
        "Apple beats on revenue",
    ]
    r = rows[0]
    assert r["available_at"] == r["published_at"] == datetime(2026, 7, 31, 12, 13, 20, tzinfo=UTC)
    assert r["id"] == news.news_id(r["url"], r["headline"]) and len(r["id"]) == 40


def test_windows_cover_the_range_without_overlap() -> None:
    w = news.windows(date(2026, 1, 1), date(2026, 3, 15), 31)
    assert w[0] == (date(2026, 1, 1), date(2026, 1, 31))
    assert w[-1][1] == date(2026, 3, 15)
    for (_, b), (c, _) in zip(w, w[1:], strict=False):
        assert c == b + timedelta(days=1)


def test_truncated_windows_are_split_until_single_days(monkeypatch: pytest.MonkeyPatch) -> None:
    fh = news.FinnhubNews(client=None)
    fh.cfg = {**fh.cfg, "truncation_threshold": 3}
    calls: list[tuple[date, date]] = []

    def fake_get(period: Period, ticker: str, a: date, b: date) -> list[dict]:
        calls.append((a, b))
        if a <= date(2026, 7, 2) <= b:  # one very busy day: every window holding it is full
            return [{"n": i} for i in range(5)]
        return [{"n": 0}]

    monkeypatch.setattr(fh, "_get", fake_get)
    out = fh.fetch(VALIDATION, "AAPL", date(2026, 7, 1), date(2026, 7, 4))
    assert (date(2026, 7, 1), date(2026, 7, 4)) == calls[0]
    assert fh.truncated == ["AAPL:2026-07-02"]
    # 07-01..04 full -> 07-01..02 full -> 07-01 (1) + 07-02 (5, still full: flagged); 07-03..04 (1)
    assert len(out) == 1 + 5 + 1


def test_non_list_payload_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Resp:
        def json(self) -> dict:
            return {"error": "You don't have access to this resource."}

    class Client:
        calls = 0

        def get(self, url: str, params: dict) -> Resp:
            return Resp()

        def close(self) -> None: ...

    monkeypatch.setenv("FINNHUB_API_KEY", "x")
    monkeypatch.setattr(news, "raw_dir", lambda: tmp_path)
    fh = news.FinnhubNews(client=Client())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unexpected payload"):
        fh.fetch(VALIDATION, "AAPL", date(2026, 7, 1), date(2026, 7, 1))


def test_news_insert_is_idempotent_and_keeps_one_row_per_ticker(tmp_path: Path) -> None:
    rows = news.parse_items(_items(), "AAPL", date(2026, 7, 1), date(2026, 7, 31))
    msft = news.parse_items(_items()[:1], "MSFT", date(2026, 7, 1), date(2026, 7, 31))
    with db.writer_session(tmp_path / "n.duckdb") as con:
        assert news.insert_news(con, rows) == 2
        assert news.insert_news(con, rows) == 0
        assert news.insert_news(con, msft) == 1  # same article, other ticker


def test_free_window_start_is_one_year_back() -> None:
    assert news.free_window_start(date(2026, 11, 13)) == date(2025, 11, 13)


# --------------------------------------------------------------------------- prices


def _yf_frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-05-29", "2026-06-01", "2026-06-02", "2026-06-03"]).rename("Date")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, None],
            "High": [1.5, 2.5, 3.5, 4.5],
            "Low": [0.5, 1.5, 2.5, 3.5],
            "Close": [1.2, 2.2, 3.2, 4.2],
            "Adj Close": [1.1, 2.1, 3.1, 4.1],
            "Volume": [10, 20, None, 40],
        },
        index=idx,
    ).reset_index()


def test_bars_keep_the_period_and_use_bar_close_as_available_at() -> None:
    df = prices.bar_rows("AAPL", "yfinance", _yf_frame(), VALIDATION)
    assert list(df["bar_date"]) == [date(2026, 6, 1), date(2026, 6, 2)]  # May 29 out, NaN open out
    assert df["close"].tolist() == [2.2, 3.2]  # Close, not Adj Close (D-29)
    assert df["volume"].tolist() == [20.0, 0.0]
    assert df["available_at"].iloc[0].to_pydatetime() == bar_close_for(date(2026, 6, 1))


def test_stooq_style_csv_is_accepted_and_malformed_rejected() -> None:
    stooq = pd.DataFrame(
        {
            "Date": ["2026-06-01"],
            "Open": [1],
            "High": [2],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [100],
        }  # fmt: skip
    )
    assert len(prices.normalise(stooq, VALIDATION)) == 1
    with pytest.raises(ValueError):
        prices.normalise(stooq.drop(columns=["Close"]), VALIDATION)


def test_bars_upsert_replaces_same_day(tmp_path: Path) -> None:
    df = prices.bar_rows("AAPL", "yfinance", _yf_frame(), VALIDATION)
    with db.writer_session(tmp_path / "p.duckdb") as con:
        prices.insert_bars(con, df)
        prices.insert_bars(con, df.assign(close=9.9))
        assert con.execute("SELECT count(*), max(close) FROM price_bars").fetchall() == [(2, 9.9)]
