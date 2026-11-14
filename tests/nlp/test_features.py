"""`make features`: sentiment rows inherit the news available_at, the table is the cache (D-16)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src import db
from src.marketview import MarketView
from src.nlp import features
from src.nlp.sentiment import LexiconBackend

LEX = LexiconBackend(["beats"], ["plunge"])
T = datetime(2026, 7, 14, 15, 0, tzinfo=UTC)  # validation


class Counting(LexiconBackend):
    def __init__(self) -> None:
        super().__init__(["beats"], ["plunge"])
        self.calls = 0

    def predict(self, texts):
        self.calls += len(texts)
        return super().predict(texts)


@pytest.fixture()
def news_db(tmp_path: Path) -> Path:
    path = tmp_path / "f.duckdb"
    rows = [
        ("a", "AAPL", T, "Apple beats estimates"),
        ("a", "MSFT", T, "Apple beats estimates"),  # one article, two tickers
        ("b", "AAPL", T + timedelta(hours=1), "Apple shares plunge"),
        ("c", "XOM", T, "Exxon beats"),  # not a requested ticker
        ("old", "AAPL", datetime(2024, 3, 1, tzinfo=UTC), "Apple beats"),  # warmup, out of window
    ]
    with db.writer_session(path) as con:
        for i, t, at, h in rows:
            con.execute(
                "INSERT INTO news VALUES (?, ?, ?, ?, '', 'u', 'test', ?)", [i, t, at, h, at]
            )
    return path


def test_rows_inherit_available_at_and_are_scored_once_per_article(news_db: Path) -> None:
    b = Counting()
    out = features.build(["validation"], ["AAPL", "MSFT"], b, news_db)
    assert out["new_rows"] == 3 and b.calls == 2  # article "a" scored once for two tickers
    view = MarketView(T + timedelta(hours=2), news_db)
    got = {(r.source_ref, r.ticker): (r.label, r.available_at) for r in view.sentiment("AAPL")}
    assert got == {("a", "AAPL"): (1, T), ("b", "AAPL"): (-1, T + timedelta(hours=1))}
    assert [r.source_ref for r in view.sentiment("MSFT", method="lexicon")] == ["a"]
    assert view.sentiment("MSFT", method="finbert") == []


def test_a_second_run_scores_nothing(news_db: Path) -> None:
    features.build(["validation"], ["AAPL", "MSFT"], LEX, news_db)
    b = Counting()
    assert features.build(["validation"], ["AAPL", "MSFT"], b, news_db)["new_rows"] == 0
    assert b.calls == 0


def test_sentiment_is_invisible_before_its_news(news_db: Path) -> None:
    features.build(["validation"], ["AAPL"], LEX, news_db)
    assert MarketView(T - timedelta(seconds=1), news_db).sentiment("AAPL") == []


@pytest.mark.parametrize("split", ["warmup", "nonsense"])
def test_only_train_and_validation_are_accepted(split: str) -> None:
    with pytest.raises(ValueError):
        features.window([split])
