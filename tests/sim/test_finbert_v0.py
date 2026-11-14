"""finbert_v0: mean sentiment of relevant, deduplicated news in (previous cutoff, as_of] (D-36)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src import db
from src.marketview import MarketView
from src.marketview.clock import cutoff_for
from src.strategies.finbert_v0 import FinbertV0, previous_cutoff

WED = date(2026, 7, 15)
MON = date(2026, 7, 13)
H = timedelta(hours=1)
STRAT = FinbertV0(method="finbert", min_articles=1, horizon_days=1)


def add(path: Path, rows: list[tuple[str, str, datetime, str, float, str]]) -> None:
    """(id, ticker, available_at, headline, score, method) -> news + sentiment rows."""
    with db.writer_session(path) as con:
        for i, t, at, h, score, method in rows:
            con.execute(
                "INSERT OR IGNORE INTO news VALUES (?, ?, ?, ?, '', 'u', 'test', ?)",
                [i, t, at, h, at],
            )
            label = (score > 0) - (score < 0)
            con.execute(
                "INSERT INTO sentiment VALUES (?, ?, ?, ?, ?, ?)", [i, t, label, score, method, at]
            )


def sig(path: Path, day: date):
    return STRAT.signal("AAPL", MarketView(cutoff_for(day), path))


@pytest.fixture()
def path(tmp_path: Path) -> Path:
    p = tmp_path / "s.duckdb"
    c, fb = cutoff_for(WED), "finbert"
    add(
        p,
        [
            # yesterday's story, before the window (previous cutoff = c - 24 h)
            ("y", "AAPL", c - H * 25, "Apple beats estimates on iPhone", 0.9, fb),
            ("a", "AAPL", c - H * 5, "Apple opens new campus in Texas", 0.4, fb),
            ("b", "AAPL", c - H * 3, "Apple faces EU antitrust probe", -0.6, fb),
            # copy of yesterday's story 23 h later: dropped by dedup although inside the window
            ("y2", "AAPL", c - H * 2, "UPDATE 1-Apple beats estimates on iPhone", 0.9, fb),
            # tagged AAPL but about someone else: not relevant
            ("x", "AAPL", c - H, "Microsoft shares surge", 0.95, fb),
            # same article scored by another backend: ignored by the method filter
            ("a", "AAPL", c - H * 5, "Apple opens new campus in Texas", -1.0, "lexicon"),
        ],
    )
    return p


def test_previous_cutoff_skips_the_weekend() -> None:
    assert previous_cutoff(cutoff_for(WED)) == cutoff_for(date(2026, 7, 14))
    assert previous_cutoff(cutoff_for(MON)) == cutoff_for(date(2026, 7, 10))


def test_mean_of_relevant_unique_news_in_the_window(path: Path) -> None:
    s = sig(path, WED)
    assert s.score == pytest.approx((0.4 - 0.6) / 2)
    assert s.reasons[0] == "news_n=2"
    assert set(s.reasons[2:]) == {"a", "b"}
    assert s.as_of == cutoff_for(WED) and s.strategy == "finbert_v0"


def test_no_news_means_cash(path: Path) -> None:
    s = STRAT.signal("MSFT", MarketView(cutoff_for(WED), path))
    assert s.score == 0.0 and "too_few" in s.reasons


def test_shift_news_after_the_cutoff_does_not_change_the_signal(path: Path) -> None:
    before = sig(path, WED)
    c = cutoff_for(WED)
    add(
        path,
        [
            ("late", "AAPL", c + timedelta(seconds=1), "Apple plunges on recall", -0.99, "finbert"),
            ("late2", "AAPL", c + timedelta(days=1), "Apple cuts guidance", -0.99, "finbert"),
        ],
    )
    assert sig(path, WED) == before
    assert sig(path, date(2026, 7, 16)).score < 0  # visible one day later, as it must be


def test_an_item_exactly_at_the_cutoff_is_visible(path: Path) -> None:
    c = cutoff_for(WED)
    add(path, [("at", "AAPL", c, "Apple raises dividend by 4 percent", 1.0, "finbert")])
    assert sig(path, WED).reasons[0] == "news_n=3"


def test_lexicon_method_reads_only_lexicon_rows(path: Path) -> None:
    lex = FinbertV0(method="lexicon", min_articles=1, horizon_days=1)
    s = lex.signal("AAPL", MarketView(cutoff_for(WED), path))
    assert s.score == -1.0 and s.reasons[0] == "news_n=1"


def test_naive_window_start_is_utc_aware() -> None:
    assert previous_cutoff(datetime(2026, 7, 15, 13, tzinfo=UTC)).tzinfo is not None
