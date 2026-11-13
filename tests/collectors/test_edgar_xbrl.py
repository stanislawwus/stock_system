"""EDGAR filings and XBRL facts: parsing, period filtering, available_at (D-24, D-29, D-30)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.collectors import edgar, xbrl
from src.collectors.common import Company, Period

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "edgar"
AAPL = Company("AAPL", "technology", "0000320193")
VALIDATION = Period("validation", date(2026, 6, 1), date(2026, 8, 31))
TEST = Period("test", date(2026, 9, 1), date(2026, 11, 6))


def _block() -> dict:
    return json.loads((FIX / "submissions_block.json").read_text(encoding="utf-8"))


def test_acceptance_is_parsed_as_utc_and_naive_values_are_rejected() -> None:
    ts = edgar.parse_acceptance("2026-07-30T20:30:28.000Z")
    assert ts == datetime(2026, 7, 30, 20, 30, 28, tzinfo=UTC)
    with pytest.raises(ValueError):
        edgar.parse_acceptance("2026-07-30T20:30:28")


def test_filter_keeps_wanted_forms_inside_the_period_only() -> None:
    kept = edgar.filter_columns(_block(), ["8-K", "10-Q", "10-K", "4"], VALIDATION)
    # S-8 dropped (form), April 8-K dropped (before), Sept 1 8-K dropped (test period).
    # 2026-08-31T21:30Z is 17:30 ET on Aug 31 -> still validation.
    assert kept["accessionNumber"] == [
        "0000320193-26-000024",
        "0000320193-26-000020",
        "0000320193-26-000018",
    ]


def test_filing_rows_carry_items_url_and_available_at() -> None:
    kept = edgar.filter_columns(_block(), ["8-K", "10-Q"], VALIDATION)
    rows = {r["accession_no"]: r for r in edgar.filing_rows(kept, AAPL, AAPL.cik)}
    r = rows["0000320193-26-000018"]
    assert r["items"] == ["2.02", "9.01"]
    assert r["available_at"] == r["accepted_at"] == datetime(2026, 7, 30, 20, 30, 28, tzinfo=UTC)
    assert r["filed_date"] == date(2026, 7, 30)
    assert r["url"].endswith("/320193/000032019326000018/0000320193-26-000018-index.htm")
    assert rows["0000320193-26-000020"]["items"] == []


def test_index_page_yields_the_press_release_exhibit() -> None:
    docs = edgar.parse_index_documents((FIX / "index_8k.htm").read_text(encoding="utf-8"))
    assert [d["type"] for d in docs] == ["8-K", "EX-99.1", "GRAPHIC"]
    assert docs[0]["href"].startswith("/Archives/")  # iXBRL viewer prefix removed
    chosen = edgar.choose_document(docs, "8-K", ["EX-99.1", "EX-99"])
    assert chosen is not None and chosen["href"].endswith("a8-kex991q3202606272026.htm")
    primary = edgar.choose_document(docs[:1], "8-K", ["EX-99.1"])
    assert primary is not None and primary["type"] == "8-K"


def test_company_with_former_cik_lists_both() -> None:
    xom = Company("XOM", "energy", "0002115436", ("0000034088",))
    assert xom.ciks == ("0002115436", "0000034088")


# --------------------------------------------------------------------------- XBRL

ACCEPT = {"0000320193-26-000020": datetime(2026, 7, 31, 10, 1, 2, tzinfo=UTC)}


def _facts() -> dict:
    return json.loads((FIX / "companyfacts.json").read_text(encoding="utf-8"))


def test_facts_are_stamped_with_the_filing_acceptance_not_the_filed_date() -> None:
    reduced = xbrl.split_by_period(_facts(), [VALIDATION, TEST], ACCEPT)
    rows = xbrl.fact_rows(AAPL, reduced["validation"], ACCEPT)
    by_name = {r["name"]: r for r in rows}
    rev = by_name["revenue"]
    # `Revenues` is listed first in config/collectors.yaml, so it wins over the contract-revenue
    # concept; only the 3-month current quarter is kept (no YTD, no prior-year comparative)
    assert rev["value"] == 95_000_000_000 and rev["period_end"] == date(2026, 6, 27)
    assert rev["available_at"] == ACCEPT["0000320193-26-000020"]
    assert rev["fiscal_period"] == "Q3-2026" and rev["method"] == "xbrl"
    assert by_name["eps_diluted"]["unit"] == "USD/share"
    assert "net_income" not in by_name  # instant fact without a start: not a duration
    assert sum(r["name"] == "revenue" for r in rows) == 1, "first concept wins, no duplicates"


def test_unknown_accession_falls_back_to_filed_date_2200_new_york() -> None:
    reduced = xbrl.split_by_period(
        _facts(), [Period("train", date(2026, 3, 1), date(2026, 5, 29))], {}
    )
    rows = xbrl.fact_rows(AAPL, reduced["train"], {})
    assert len(rows) == 1
    # filed 2026-05-01 -> 22:00 EDT = 02:00 UTC next day: never earlier than real acceptance
    assert rows[0]["available_at"] == datetime(2026, 5, 2, 2, 0, tzinfo=UTC)


def test_test_period_facts_never_reach_the_dev_cache() -> None:
    reduced = xbrl.split_by_period(_facts(), [VALIDATION], ACCEPT)
    vals = [
        e["val"] for units in reduced["validation"].values() for es in units.values() for e in es
    ]
    assert 416_000_000_000 not in vals  # 10-K filed 2026-10-31 belongs to the test period
    assert all(
        e["form"] in ("10-Q", "10-K")
        for units in reduced["validation"].values()
        for es in units.values()
        for e in es
    )
