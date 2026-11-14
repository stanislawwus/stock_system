"""Read-only tools of the chat agent (AGENTS.md R2, R7, R13; D-46).

Every tool is a plain service function. History is read only through a `MarketView(as_of)` that
the caller opened inside `src.db.guarded()`; nothing here writes, and nothing reads the open of a
later day. `get_signals` reuses the /signals service; `run_backtest_summary` reads the committed
backtest record of a strategy (it does not start a simulation: that is a long job, R13).
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from src.collectors.common import ROOT
from src.marketview import MarketView
from src.nlp.relevance import mentions
from src.schemas import SignalsResponse
from src.strategies.service import signals_for

CONFIG_PATH = ROOT / "config" / "chat.yaml"
RESULTS_DIR = ROOT / "evals" / "results"

__all__ = [
    "BacktestSummary",
    "PriceMove",
    "RankedFiling",
    "RankedNews",
    "config",
    "get_events",
    "get_news",
    "get_prices",
    "get_signals",
    "run_backtest_summary",
]


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PriceMove(_Out):
    """close(D-1) -> close(D) of `ticker` and of the benchmark on the same two sessions."""

    ticker: str
    day: date
    prev_day: date
    close: float
    prev_close: float
    ret: float
    benchmark: str
    bench_close: float
    bench_prev_close: float
    bench_ret: float
    abnormal_ret: float  # ret - bench_ret (market-adjusted, same convention as impact_labels)
    window_start: datetime  # close(D-1): news/filings after it and up to close(D) are candidates
    window_end: datetime


class RankedNews(_Out):
    id: str
    ticker: str
    headline: str
    source: str
    available_at: datetime
    relevance: float
    sentiment: float | None
    recency: float
    score: float


class RankedFiling(_Out):
    accession_no: str
    ticker: str
    form: str
    items: list[str]
    available_at: datetime
    url: str


class BacktestSummary(_Out):
    strategy: str
    split: str
    start: str
    end: str
    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    benchmark_sharpe: float  # the record's target: buy-and-hold on the same names and days


def _bars_until(view: MarketView, ticker: str, day: date) -> list[Any]:
    lookback = int(config()["window"]["max_lookback_calendar_days"]) + 2
    return [b for b in view.prices(ticker, lookback_days=lookback) if b.bar_date <= day]


def get_prices(view: MarketView, ticker: str, day: date) -> PriceMove | None:
    """The last session on or before `day` visible at `view.as_of`, against the session before it.

    None when either the ticker or the benchmark lacks two bars (no data, holiday gap > lookback).
    """
    bench = str(config()["benchmark"])
    bars = _bars_until(view, ticker, day)
    if len(bars) < 2:
        return None
    prev, cur = bars[-2], bars[-1]
    bbars = {b.bar_date: b for b in _bars_until(view, bench, cur.bar_date)}
    if cur.bar_date not in bbars or prev.bar_date not in bbars:
        return None
    bcur, bprev = bbars[cur.bar_date], bbars[prev.bar_date]
    ret = cur.close / prev.close - 1
    bret = bcur.close / bprev.close - 1
    extra = timedelta(hours=float(config()["window"]["extra_hours_before"]))
    return PriceMove(
        ticker=ticker,
        day=cur.bar_date,
        prev_day=prev.bar_date,
        close=cur.close,
        prev_close=prev.close,
        ret=ret,
        benchmark=bench,
        bench_close=bcur.close,
        bench_prev_close=bprev.close,
        bench_ret=bret,
        abnormal_ret=ret - bret,
        window_start=prev.available_at - extra,
        window_end=cur.available_at,
    )


def _lookback_days(view: MarketView, start: datetime) -> int:
    return max(1, math.ceil((view.as_of - start).total_seconds() / 86400) + 1)


def get_news(
    view: MarketView, ticker: str, start: datetime, end: datetime, top: int | None = None
) -> list[RankedNews]:
    """News of `ticker` with start < available_at <= end, ranked by relevance, |sentiment| and
    recency (weights in config/chat.yaml). Relevance: 1.0 when the headline names the company,
    0.5 when only the summary does; rows below `min_relevance` and repeated headlines are dropped.
    """
    cfg = config()["ranking"]
    top = int(cfg["top_news"]) if top is None else top
    days = _lookback_days(view, start)
    span = max((end - start).total_seconds(), 1.0)
    sent = {
        s.source_ref: s.score for s in view.sentiment(ticker, lookback_days=days, method="finbert")
    }
    seen: set[str] = set()
    out: list[RankedNews] = []
    for n in view.news(ticker, lookback_days=days):
        if not (start < n.available_at <= end):
            continue
        key = " ".join(n.headline.lower().split())
        if key in seen:
            continue
        seen.add(key)
        if mentions(ticker, n.headline):
            rel = 1.0
        elif mentions(ticker, n.summary):
            rel = 0.5
        else:
            rel = 0.0
        if rel < float(cfg["min_relevance"]):
            continue
        s = sent.get(n.id)
        rec = 1.0 - (end - n.available_at).total_seconds() / span
        score = (
            float(cfg["w_relevance"]) * rel
            + float(cfg["w_sentiment"]) * abs(s or 0.0)
            + float(cfg["w_recency"]) * rec
        )
        out.append(
            RankedNews(
                id=n.id,
                ticker=n.ticker,
                headline=n.headline,
                source=n.source,
                available_at=n.available_at,
                relevance=rel,
                sentiment=s,
                recency=round(rec, 4),
                score=round(score, 4),
            )
        )
    out.sort(key=lambda r: (-r.score, r.id))
    return out[:top]


def get_events(
    view: MarketView, ticker: str, start: datetime, end: datetime, top: int | None = None
) -> list[RankedFiling]:
    """8-K filings of `ticker` accepted in (start, end], most recent first."""
    top = int(config()["ranking"]["top_filings"]) if top is None else top
    rows = view.filings(ticker, lookback_days=_lookback_days(view, start), forms=("8-K",))
    out = [
        RankedFiling(
            accession_no=f.accession_no,
            ticker=f.ticker,
            form=f.form,
            items=list(f.items),
            available_at=f.available_at,
            url=f.url,
        )
        for f in rows
        if start < f.available_at <= end
    ]
    out.sort(key=lambda r: (r.available_at, r.accession_no), reverse=True)
    return out[:top]


def get_signals(ticker: str, day: date) -> SignalsResponse:
    """Every strategy's signal for `ticker` at the 09:00 ET cutoff of `day` (/signals service)."""
    return signals_for(ticker, day)


def run_backtest_summary(strategy: str, split: str = "validation") -> BacktestSummary | None:
    """Summary of the committed backtest record `evals/results/backtest_<strategy>_<split>.json`.

    Only `train` and `validation` exist before the final run (R3). None when there is no record.
    """
    path = RESULTS_DIR / f"backtest_{strategy}_{split}.json"
    if split not in ("train", "validation") or not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    d = rec["details"]
    return BacktestSummary(
        strategy=strategy,
        split=split,
        start=str(d["start"]),
        end=str(d["end"]),
        total_return=float(d["total_return"]),
        sharpe=float(d["sharpe"]),
        max_drawdown=float(d["max_drawdown"]),
        n_trades=int(d["n_trades"]),
        benchmark_sharpe=float(rec["target"]),
    )
