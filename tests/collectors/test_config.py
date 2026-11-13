"""Universe, watchlist and env example stay consistent (D-29, R9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.collectors import common

ROOT = Path(__file__).resolve().parents[2]


def test_universe_is_60_to_80_unique_companies_with_ciks() -> None:
    companies, benchmark = common.load_universe()
    tickers = [c.ticker for c in companies]
    ciks = [k for c in companies for k in c.ciks]
    assert 60 <= len(companies) <= 80
    assert len(set(tickers)) == len(tickers) and len(set(ciks)) == len(ciks)
    assert all(len(k) == 10 and k.isdigit() for k in ciks)
    assert benchmark == "SPY" and benchmark not in tickers


def test_watchlist_is_ten_names_inside_the_universe() -> None:
    companies, _ = common.load_universe()
    wl = common.watchlist_tickers()
    assert len(wl) == 10 and set(wl) <= {c.ticker for c in companies}


def test_periods_come_from_the_frozen_splits() -> None:
    ps = common.load_periods(["validation", "warmup", "train"])
    assert [p.name for p in ps] == ["warmup", "train", "validation"]
    with pytest.raises(ValueError):
        common.load_periods(["holdout"])


def test_env_example_lists_every_key_with_an_empty_value() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    pairs = dict(ln.split("=", 1) for ln in lines if "=" in ln and not ln.startswith("#"))
    for key in ("OPENROUTER_API_KEY", "FINNHUB_API_KEY", "SEC_USER_AGENT"):
        assert pairs.get(key) == "", f"{key} must be listed with an empty value"


def test_missing_sec_user_agent_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.collectors import edgar

    monkeypatch.setattr(common, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        edgar.sec_client()
    monkeypatch.setenv("SEC_USER_AGENT", "no-contact-given")
    with pytest.raises(RuntimeError, match="e-mail"):
        edgar.sec_client()
