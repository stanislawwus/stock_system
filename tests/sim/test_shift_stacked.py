"""Shift test for the news and signal-stack strategies (time-leak-check step 7; L-03).

A fixture DB with bars, news + FinBERT sentiment, 8-K filings with items and XBRL revenue facts
for AAPL and MSFT. Signals for days D1..Dn are computed; then rows that become public only after
the pivot day's 09:00 ET cutoff are added (corrupted bars from the pivot day on, extreme news
and sentiment, 8-Ks, a revenue fact) and the signals are recomputed. Up to and including the
pivot day nothing may change; later days must see the new rows (otherwise the fixture is inert).
The impact model is a fixed logistic function of *all* features, so every feature can move it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import db
from src.impact.features import feature_names
from src.impact.model import ImpactModel
from src.marketview.clock import MARKET_TZ, bar_close_for, cutoff_for
from src.sim.engine import collect_signals, default_view_factory
from src.strategies.finbert_v0 import FinbertV0
from src.strategies.hedge import AggregateHedge
from src.strategies.signal_stack import AggregateEqual, EventStrategy, ImpactStrategy
from tests.sim.conftest import business_days, wave

TICKERS = ["AAPL", "MSFT"]
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft"}
TABLE = {"2.02": {"mean_drift_ar": 0.004, "n": 50}, "1.01": {"mean_drift_ar": -0.02, "n": 40}}
DAYS = [d for d in business_days() if date(2026, 6, 1) <= d <= date(2026, 7, 10)]
PIVOT = 15


class _Logistic:
    """p = sigmoid(x . w) with fixed weights: every feature moves the probability."""

    def __init__(self, n: int) -> None:
        self.w = np.random.default_rng(7).normal(0, 0.5, n)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = 1 / (1 + np.exp(-np.nan_to_num(x) @ self.w))
        return np.column_stack([1 - p, p])


class _Identity:
    def transform(self, p: np.ndarray) -> np.ndarray:
        return p


MODEL = ImpactModel(feature_names(), "logistic", _Logistic(len(feature_names())), "identity",
                    _Identity())  # fmt: skip


def _at(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=MARKET_TZ)


def _insert(path: Path, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    with db.writer_session(path) as con:
        con.register("df", df)
        cols = ", ".join(df.columns)
        con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM df")


def _bars(days: list[date], scale: float = 1.0) -> list[dict]:
    all_days = business_days()
    rows = []
    for k, t in enumerate([*TICKERS, "SPY"]):
        key = "SPY" if t == "SPY" else ("A", "B")[k]
        for d in days:
            c = wave(all_days.index(d), key) * scale
            rows.append({"ticker": t, "bar_date": d, "open": c * 0.998, "high": c, "low": c * 0.997,
                         "close": c, "volume": 1e6, "source": "test",
                         "available_at": bar_close_for(d)})  # fmt: skip
    return rows


VOCAB = (
    "shares",
    "rally",
    "probe",
    "recall",
    "outlook",
    "chip",
    "cloud",
    "deal",
    "margin",
    "guidance",
    "lawsuit",
    "dividend",
    "buyback",
    "supplier",
    "tariff",
    "upgrade",
    "downgrade",
    "launch",
    "delay",
    "record",
    "slump",
    "merger",
    "patent",
    "strike",
    "forecast",
    "beats",
    "misses",
    "cuts",
    "raises",
    "expands",
    "exits",
    "wins",
    "loses",
    "plans",
    "warns",
)


def _words(seed: str, n: int = 8) -> str:
    rng = np.random.default_rng(int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16))
    return " ".join(rng.choice(VOCAB, size=n, replace=False))


def _news(days: list[date], hours: tuple[int, ...], tone: float | None = None) -> tuple:
    news, sent = [], []
    for i, d in enumerate(days):
        for t in TICKERS:
            for h in hours:
                head = f"{NAMES[t]} {_words(f'{d}{h}{tone}{t}')}"  # distinct: dedup keeps it
                nid = hashlib.sha1(head.encode()).hexdigest()
                at = _at(d, h, 30)
                news.append({"id": nid, "ticker": t, "published_at": at, "headline": head,
                             "summary": "", "url": f"u/{nid}", "source": "test",
                             "available_at": at})  # fmt: skip
                s = tone if tone is not None else math.sin(i + h + len(t))
                sent.append({"source_ref": nid, "ticker": t, "label": int(np.sign(s)),
                             "score": float(s), "method": "finbert",
                             "available_at": at})  # fmt: skip
    return news, sent


def _filings(days: list[date], tag: str) -> list[dict]:
    out = []
    for i, d in enumerate(days):
        t = TICKERS[i % 2]
        acc = f"{tag}-{d}-{t}"
        at = _at(d, 7 if tag == "dev" else 10)
        out.append({"accession_no": acc, "ticker": t, "cik": "1", "form": "8-K",
                    "items": ["2.02"] if i % 3 else ["1.01"], "accepted_at": at, "filed_date": d,
                    "url": acc, "text_path": None, "available_at": at})  # fmt: skip
    return out


def _fact(t: str, end: date, fp: str, value: float, at: datetime) -> dict:
    return {"ticker": t, "name": "revenue", "value": value, "unit": "USD", "period_end": end,
            "fiscal_period": fp, "method": "xbrl", "source_ref": f"{t}-{fp}", "grounding": None,
            "available_at": at}  # fmt: skip


@pytest.fixture()
def world(tmp_path: Path) -> Path:
    path = tmp_path / "world.duckdb"
    _insert(path, "price_bars", _bars(business_days()))
    news, sent = _news([d for d in DAYS], (7, 12))  # 07:30 before, 12:30 after the cutoff
    _insert(path, "news", news)
    _insert(path, "sentiment", sent)
    _insert(path, "filings", _filings(DAYS[::2], "dev"))
    facts = [_fact(t, date(2025, 3, 29), "Q2-2025", 100.0, _at(date(2025, 5, 1), 16))
             for t in TICKERS]  # fmt: skip
    facts += [_fact(t, date(2026, 3, 28), "Q2-2026", 110.0, _at(date(2026, 5, 1), 16))
              for t in TICKERS]  # fmt: skip
    _insert(path, "facts", facts)
    return path


def _future(path: Path) -> None:
    """Rows that become public only after the pivot day's cutoff."""
    later = DAYS[PIVOT:]
    _insert(path, "price_bars", _bars(later, scale=3.0))  # bars of the pivot day close at 16:00
    news, sent = _news(later, (10,), tone=-1.0)  # 10:30 ET: after the 09:00 cutoff
    _insert(path, "news", news)
    _insert(path, "sentiment", sent)
    _insert(path, "filings", _filings(later, "late"))
    _insert(path, "facts", [_fact(t, date(2026, 6, 27), "Q3-2026", 999.0,
                                  _at(DAYS[PIVOT], 9, 30)) for t in TICKERS])  # fmt: skip
    _insert(path, "facts", [_fact(t, date(2025, 6, 28), "Q3-2025", 100.0,
                                  _at(DAYS[PIVOT], 9, 30)) for t in TICKERS])  # fmt: skip


STRATEGIES: dict[str, Callable[[], object]] = {
    "finbert_v0": lambda: FinbertV0("finbert", 1, 1),
    "event": lambda: EventStrategy(3, -2.0, table=TABLE),
    "impact": lambda: ImpactStrategy(1, -2.0, model=MODEL),
    "aggregate_equal": lambda: AggregateEqual(1, -2.0, table=TABLE, model=MODEL),
    "aggregate_hedge": lambda: AggregateHedge(1, -2.0, 3.0, None, 20, table=TABLE, model=MODEL),
}


def _run(make: Callable[[], object], path: Path) -> list[list[float]]:
    strat, vf = make(), default_view_factory(path)  # a fresh object: Hedge walks forward
    out = []
    for d in DAYS:
        sig = collect_signals(strat, TICKERS, d, vf)  # type: ignore[arg-type]
        out.append([round(sig[t].score, 12) for t in TICKERS])
    return out


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_signals_up_to_the_pivot_ignore_everything_published_after_its_cutoff(
    name: str, world: Path
) -> None:
    before = _run(STRATEGIES[name], world)
    assert len({tuple(r) for r in before}) > 1, "fixture inert: the strategy never changes"
    _future(world)
    after = _run(STRATEGIES[name], world)
    assert after[: PIVOT + 1] == before[: PIVOT + 1]  # D = pivot: newest input is pre-cutoff
    assert after[PIVOT + 1 :] != before[PIVOT + 1 :]  # and the new rows do arrive later


def test_the_future_rows_are_really_after_the_pivot_cutoff() -> None:
    assert _at(DAYS[PIVOT], 9, 30) > cutoff_for(DAYS[PIVOT])
    assert _at(DAYS[PIVOT], 10, 30) > cutoff_for(DAYS[PIVOT])
    assert bar_close_for(DAYS[PIVOT]) > cutoff_for(DAYS[PIVOT])
    assert _at(DAYS[0], 7, 30) < cutoff_for(DAYS[0]) < _at(DAYS[0], 12, 30)
    assert timedelta(0) < _at(DAYS[PIVOT], 9, 30) - cutoff_for(DAYS[PIVOT])


@pytest.mark.parametrize("name", ["finbert_v0", "event", "aggregate_equal"])
def test_the_shift_test_catches_a_leak(name: str, world: Path) -> None:
    """Mutation check: a view whose cutoff is 3 h late (12:00 ET) sees the 10:30 rows of the
    pivot day, so the same comparison must fail."""
    from src.marketview import MarketView

    def run() -> list[list[float]]:
        # straight to strategy.signal: collect_signals itself would reject the wrong as_of
        strat, out = STRATEGIES[name](), []
        for d in DAYS:
            with MarketView(cutoff_for(d) + timedelta(hours=3), world) as view:
                out.append([round(strat.signal(t, view).score, 12) for t in TICKERS])  # type: ignore[attr-defined]
        return out

    before = run()
    _future(world)
    assert run()[: PIVOT + 1] != before[: PIVOT + 1]
