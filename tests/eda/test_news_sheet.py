"""The labelling sheet is stratified, deterministic, test-free and has BLANK label columns."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from evals import make_news_sheet as sheet_mod
from src import db
from src.eda.data import open_dev
from src.nlp import clean

TICKERS = ["AAA", "BBB", "CCC"]
REAL = clean.config()  # the unpatched config


def _cfg() -> dict:
    cfg = copy.deepcopy(REAL)
    cfg["label_sheet"]["quotas"] = {
        "2.02": 2,
        "5.02": 2,
        "1.01": 2,
        "7.01_8.01": 2,
        "other": 1,
        "none": 6,
    }
    return cfg


@pytest.fixture()
def con(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    monkeypatch.setattr(clean, "config", _cfg)
    path = tmp_path / "s.duckdb"
    base = datetime(2025, 12, 2, 15, 0, tzinfo=UTC)
    with db.writer_session(path) as c:
        n = 0
        for day in range(240):
            for t in TICKERS:
                ts = base + timedelta(days=day, minutes=7 * n % 300)
                text = f"{t} reports distinct story number {n} about segment {n * 7}"
                c.execute(
                    "INSERT INTO news VALUES (?, ?, ?, ?, '', ?, 'test', ?)",
                    [f"id{n}", t, ts, text, f"http://x/{n}", ts],
                )
                n += 1
        # unusable rows: must never be drawn
        ts = base + timedelta(days=5)
        c.execute("INSERT INTO news VALUES ('bad1','AAA',?,'Form 8.3','','u1','test',?)", [ts, ts])
        late = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
        c.execute(
            "INSERT INTO news VALUES ('late', 'AAA', ?, 'AAA test period headline here', "
            "'', 'u2', 'test', ?)",
            [late, late],
        )
        filings = [
            ("AAA", ["2.02"]),
            ("BBB", ["5.02", "9.01"]),
            ("CCC", ["1.01"]),
            ("AAA", ["8.01"]),
            ("BBB", ["3.02"]),
        ]
        for i, (t, items) in enumerate(filings):
            acc = base + timedelta(days=30 * (i + 1))
            c.execute(
                "INSERT INTO filings VALUES (?, ?, '1', '8-K', ?, ?, ?, 'u', NULL, ?)",
                [f"acc{i}", t, items, acc, acc.date(), acc],
            )
    return open_dev(path)


def test_sheet_is_stratified_blank_and_free_of_unusable_rows(con) -> None:  # noqa: ANN001
    sheet, strata = sheet_mod.build(con)
    assert len(sheet) == 15 and sheet["id"].is_unique
    assert (sheet["sentiment"] == "").all() and (sheet["events"] == "").all()
    assert not {"bad1", "late"} & set(sheet["id"])
    assert "stratum" not in sheet.columns  # the 8-K item stays out of the annotator's view
    counts = strata["stratum"].value_counts().to_dict()
    assert counts["2.02"] == 2 and counts["5.02"] == 2 and counts["none"] >= 6
    assert sheet["ticker"].nunique() == 3 and set(sheet["year"]) == {2025, 2026}


def test_sheet_is_deterministic(con) -> None:  # noqa: ANN001
    a, _ = sheet_mod.build(con)
    b, _ = sheet_mod.build(con)
    pd.testing.assert_frame_equal(a, b)


def test_a_thin_stratum_shortfall_moves_to_none(con, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    def cfg() -> dict:
        c = _cfg()
        c["label_sheet"]["quotas"]["1.01"] = 500  # far more than exist
        return c

    monkeypatch.setattr(clean, "config", cfg)
    _, strata = sheet_mod.build(con)
    assert (strata["stratum"] == "1.01").sum() < 500
    assert len(strata) >= 15 and (strata["stratum"] == "none").sum() > 6


def test_files_have_the_provenance_header_and_blank_columns(con, tmp_path: Path) -> None:  # noqa: ANN001
    sheet, strata = sheet_mod.build(con)
    sheet_mod.write(sheet, strata, tmp_path)
    text = (tmp_path / "news_100.draft.csv").read_text(encoding="utf-8")
    assert text.startswith("# gold_set: news_100") and "llm_suggestions: no" in text
    back = pd.read_csv(tmp_path / "news_100.draft.csv", comment="#", keep_default_na=False)
    assert list(back.columns) == sheet_mod.COLUMNS and (back["sentiment"] == "").all()
    assert (tmp_path / "news_100.strata.csv").exists()


def test_sheet_refuses_to_overwrite_a_frozen_gold_set(tmp_path) -> None:
    (tmp_path / "news_100.csv").write_text("# gold_set: news_100", encoding="utf-8")
    with pytest.raises(sheet_mod.FrozenGoldSetError):
        sheet_mod.write(pd.DataFrame(), pd.DataFrame(), tmp_path)
    assert not (tmp_path / "news_100.strata.csv").exists()
