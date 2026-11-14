"""`make features-test` (D-38): test-period inference to parquet, resumable, news text only."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src import db, splits
from src.marketview import MarketView, cutoff_for
from src.nlp import features_test as ft
from src.nlp.sentiment import LexiconBackend

ROOT = Path(__file__).resolve().parents[2]
FIRST_CUTOFF = cutoff_for(date(2026, 9, 1))
NEWS = [
    ("pre", "AAPL", datetime(2026, 8, 10, 15, tzinfo=UTC), "Apple beats"),  # before the pad
    ("dev", "AAPL", datetime(2026, 8, 28, 15, tzinfo=UTC), "Apple beats"),  # already in DuckDB
    ("pad", "AAPL", datetime(2026, 8, 31, 20, tzinfo=UTC), "Apple shares plunge"),
    ("t1", "AAPL", datetime(2026, 9, 2, 12, tzinfo=UTC), "Apple beats estimates"),
    ("t1", "MSFT", datetime(2026, 9, 2, 12, tzinfo=UTC), "Apple beats estimates"),
    ("t2", "AAPL", datetime(2026, 10, 1, 12, tzinfo=UTC), "Apple shares plunge"),
    ("t3", "AAPL", datetime(2026, 10, 2, 12, tzinfo=UTC), "Apple beats"),
    ("x", "XOM", datetime(2026, 9, 2, 12, tzinfo=UTC), "Exxon beats"),  # not requested
    ("late", "AAPL", datetime(2026, 11, 10, 12, tzinfo=UTC), "Apple beats"),  # after test end
]
EXPECTED = {("pad", "AAPL"), ("t1", "AAPL"), ("t1", "MSFT"), ("t2", "AAPL"), ("t3", "AAPL")}
TICKERS = ["AAPL", "MSFT"]


class Counting(LexiconBackend):
    def __init__(self, fail_after: int | None = None) -> None:
        super().__init__(["beats"], ["plunge"])
        self.calls = 0
        self.fail_after = fail_after

    def predict(self, texts):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise KeyboardInterrupt("simulated interruption")
        self.calls += len(texts)
        return super().predict(texts)


@pytest.fixture()
def news_db(tmp_path: Path) -> Path:
    """Only `news` and `sentiment` exist: touching prices, labels or anything else would fail."""
    path = tmp_path / "ft.duckdb"
    with db.writer_session(path) as con:
        for i, t, at, h in NEWS:
            con.execute(
                "INSERT INTO news VALUES (?, ?, ?, ?, '', 'u', 'test', ?)", [i, t, at, h, at]
            )
        con.execute(
            "INSERT INTO sentiment VALUES ('dev', 'AAPL', 1, 0.5, 'lexicon', ?)", [NEWS[1][2]]
        )
        tables = [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()]
        for name in tables:
            if name not in ("news", "sentiment"):
                con.execute(f"DROP TABLE {name}")
    return path


@pytest.fixture()
def out(tmp_path: Path) -> dict[str, Path]:
    return {"scores_path": tmp_path / "scores.parquet", "parts_dir": tmp_path / "parts"}


def _pairs(path: Path) -> set[tuple[str, str]]:
    df = pd.read_parquet(path)
    return set(zip(df["source_ref"], df["ticker"], strict=True))


def test_window_covers_the_test_period_plus_the_lookback_pad() -> None:
    lo, hi = ft.window()
    assert lo == FIRST_CUTOFF - timedelta(days=7)
    assert hi == cutoff_for(date(2026, 11, 6))


def test_scores_go_to_parquet_and_never_to_duckdb(
    news_db: Path, out: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_writer(*a: object, **k: object) -> None:
        raise AssertionError("features-test must not take the DuckDB write lock (R7)")

    monkeypatch.setattr(db, "acquire_writer_lock", no_writer)
    b = Counting()
    summary = ft.build(TICKERS, b, news_db, **out)
    assert _pairs(out["scores_path"]) == EXPECTED
    assert b.calls == 4  # t1 is scored once for two tickers
    df = pd.read_parquet(out["scores_path"])
    assert list(df.columns) == ["source_ref", "ticker", "label", "score", "method", "available_at"]
    assert str(df["available_at"].dt.tz) == "UTC"
    t2 = df[df["source_ref"] == "t2"].iloc[0]
    assert t2["available_at"] == NEWS[5][2]  # inherits the news item's available_at (D-16)
    con = db.connect(news_db)
    try:
        assert con.execute("SELECT count(*) FROM sentiment").fetchone()[0] == 1
    finally:
        con.close()
    assert set(summary) == {"method", "new_rows", "rows_total", "seconds", "path"}  # no scores
    manifest = json.loads(out["scores_path"].with_suffix(".manifest.json").read_text("utf-8"))
    assert manifest["complete"] and manifest["rows"] == len(EXPECTED)
    assert not {"score", "label", "mean"} & set(manifest)


def test_an_interrupted_run_resumes_from_its_checkpoints(
    news_db: Path, out: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(KeyboardInterrupt):
        ft.build(TICKERS, Counting(fail_after=2), news_db, chunk=1, **out)
    assert len(list(out["parts_dir"].glob("part-*.parquet"))) == 2
    assert not out["scores_path"].exists()
    resumed = Counting()
    ft.build(TICKERS, resumed, news_db, chunk=1, **out)
    assert resumed.calls == 2  # only the two articles left
    fresh = {"scores_path": tmp_path / "f.parquet", "parts_dir": tmp_path / "f"}
    ft.build(TICKERS, Counting(), news_db, **fresh)
    key = ["source_ref", "ticker"]
    a = pd.read_parquet(out["scores_path"]).sort_values(key, ignore_index=True)
    b = pd.read_parquet(fresh["scores_path"]).sort_values(key, ignore_index=True)
    pd.testing.assert_frame_equal(a, b)
    again = Counting()
    ft.build(TICKERS, again, news_db, **out)
    assert again.calls == 0


def test_a_resume_with_another_model_is_refused(news_db: Path, out: dict[str, Path]) -> None:
    out["parts_dir"].mkdir()
    meta = {"method": "finbert", "model": "ProsusAI/finbert", "revision": "other"}
    (out["parts_dir"] / ft.META).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="delete it"):
        ft.build(TICKERS, Counting(), news_db, **out)


def test_the_module_reads_no_prices_labels_or_strategies() -> None:
    src = (ROOT / "src" / "nlp" / "features_test.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]  # skip the docstring, which names what it does not do
    imports = re.findall(r"^(?:from|import) (\S+)", code, flags=re.M)
    banned = ("src.sim", "src.strategies", "src.impact", "src.marketview", "src.api", "src.llm")
    assert not [m for m in imports if m.startswith(banned) and m != "src.marketview.clock"]
    for word in ("price_bars", "impact_labels", "execution", "open_price", "MarketView", "FROM "):
        assert word not in code, word  # the only SQL is features.pending (news + sentiment)


# --------------------------------------------------------------------------- final-run read


@pytest.fixture()
def final(
    news_db: Path, out: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(splits, "FINAL_LOCK", tmp_path / "TEST_FINAL.lock")
    monkeypatch.setattr(splits, "final_scores_path", lambda: out["scores_path"])
    ft.build(TICKERS, Counting(), news_db, **out)
    return news_db


def test_the_final_run_reads_test_scores_from_the_parquet(final: Path) -> None:
    as_of = NEWS[6][2] - timedelta(seconds=1)  # one second before t3
    with splits.final_run():
        rows = MarketView(as_of, final).sentiment("AAPL", lookback_days=60, method="lexicon")
    assert [r.source_ref for r in rows] == ["dev", "pad", "t1", "t2"]  # t3 is after as_of
    assert [r.score for r in rows][0] == 0.5  # the DuckDB row, not a rescored copy


def test_a_pair_in_both_sources_is_read_once(final: Path, out: dict[str, Path]) -> None:
    df = pd.read_parquet(out["scores_path"])
    dup = df.iloc[[0]].assign(source_ref="dev", available_at=NEWS[1][2], score=-1.0)
    pd.concat([df, dup]).to_parquet(out["scores_path"], index=False)
    with splits.final_run():
        rows = MarketView(cutoff_for(date(2026, 9, 3)), final).sentiment("AAPL", 30)
    assert [(r.source_ref, r.score) for r in rows if r.source_ref == "dev"] == [("dev", 0.5)]


def test_dev_views_never_read_the_parquet(final: Path) -> None:
    rows = MarketView(FIRST_CUTOFF - timedelta(seconds=1), final).sentiment("AAPL", 30)
    assert [r.source_ref for r in rows] == ["dev"]  # "pad" is only in the parquet


def test_the_final_run_fails_loudly_without_the_parquet(
    news_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(splits, "FINAL_LOCK", tmp_path / "TEST_FINAL.lock")
    monkeypatch.setattr(splits, "final_scores_path", lambda: tmp_path / "missing.parquet")
    with splits.final_run(), pytest.raises(FileNotFoundError, match="make features-test"):
        MarketView(cutoff_for(date(2026, 9, 3)), news_db).sentiment("AAPL")
