"""Pydantic contracts shared by collectors, NLP, strategies, simulator, API and LLM output.

Every stored record extends `Record` and therefore carries `available_at` (tz-aware UTC): the moment
the information became publicly observable (AGENTS.md R2, decisions D-24). Keep this file minimal.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

DISCLAIMER = "This is a simulation, not investment advice."

Source = Literal["yfinance", "stooq", "binance", "finnhub", "edgar", "demo", "test"]
Method = Literal["llm", "rules", "lexicon", "finbert", "index", "xbrl", "regex", "model"]

# 8-K item numbers used as the event taxonomy (D-10). `none` = no event in the text.
EventLabel = Literal[
    "1.01",
    "2.01",
    "2.02",
    "2.05",
    "3.02",
    "5.02",
    "5.07",
    "7.01",
    "8.01",
    "9.01",
    "other_item",
    "none",
]


class Record(BaseModel):
    """Base for anything persisted or served through MarketView."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available_at: AwareDatetime = Field(description="UTC moment the record became public")


# --------------------------------------------------------------------------- raw inputs


class PriceBar(Record):
    """Daily OHLCV bar. available_at = bar close time (16:00 ET stocks, candle close crypto)."""

    ticker: str
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: Source


class NewsItem(Record):
    """One news article. available_at = publication datetime from the provider."""

    id: str = Field(description="stable id, sha1 of url + headline")
    ticker: str
    published_at: AwareDatetime
    headline: str
    summary: str = ""
    url: str
    source: Source


class Filing(Record):
    """SEC filing header. available_at = EDGAR acceptanceDateTime."""

    accession_no: str
    ticker: str
    cik: str
    form: Literal["8-K", "10-Q", "10-K", "4"]
    items: list[str] = Field(default_factory=list, description="8-K item numbers from the index")
    accepted_at: AwareDatetime
    filed_date: date
    url: str
    text_path: str | None = Field(default=None, description="raw cache path of the main document")


class Span(BaseModel):
    """Grounding of an extracted value in its source text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_ref: str = Field(description="news id or accession number")
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end < self.start:
            raise ValueError("span end before start")
        return self


class Fact(Record):
    """A financial number. XBRL facts: available_at = filed date of the 10-Q/10-K (gold).
    Extracted facts: available_at = that of the source text; `grounding` is required."""

    ticker: str
    name: Literal["revenue", "eps_diluted", "net_income"]
    value: float
    unit: Literal["USD", "USD/share"]
    period_end: date
    fiscal_period: str = Field(description="e.g. Q3-2026")
    method: Method
    source_ref: str
    grounding: Span | None = None

    @model_validator(mode="after")
    def _extracted_facts_are_grounded(self) -> Fact:
        if self.method in ("llm", "regex") and self.grounding is None:
            raise ValueError("extracted facts must carry a grounding span")
        return self


# --------------------------------------------------------------------------- NLP outputs


class Sentiment(Record):
    """available_at = that of the source text (D-16)."""

    source_ref: str
    ticker: str
    label: Literal[-1, 0, 1]
    score: float = Field(ge=-1, le=1)
    method: Method


class Event(Record):
    """Multi-label event classification of a news item or filing. available_at = source's."""

    source_ref: str
    ticker: str
    labels: list[EventLabel] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    method: Method

    @model_validator(mode="after")
    def _none_is_exclusive(self) -> Event:
        if "none" in self.labels and len(self.labels) > 1:
            raise ValueError("'none' cannot be combined with other labels")
        return self


# --------------------------------------------------------------------------- decisions


class Signal(Record):
    """Strategy output for one ticker at one cutoff. Long-only: score <= 0 means cash (R8).
    available_at == as_of."""

    ticker: str
    as_of: AwareDatetime
    strategy: str
    score: float = Field(ge=-1, le=1)
    calibrated_p: float | None = Field(default=None, ge=0, le=1, description="P(drift_ar > 0)")
    reasons: list[str] = Field(default_factory=list, description="source_refs or indicator names")

    @model_validator(mode="before")
    @classmethod
    def _default_available_at(cls, data: dict) -> dict:
        if isinstance(data, dict) and "available_at" not in data and "as_of" in data:
            data = {**data, "available_at": data["as_of"]}
        return data

    @model_validator(mode="after")
    def _available_at_is_as_of(self) -> Signal:
        if self.available_at != self.as_of:
            raise ValueError("Signal.available_at must equal as_of")
        return self


class Trade(Record):
    """A simulator fill: at the open of D (entries, score <= 0 exits) or at the close of D+h
    (time exits, end of split). available_at = fill time."""

    ticker: str
    day: date
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0)
    price: float = Field(gt=0)
    commission: float = Field(ge=0)
    strategy: str
    reason: str = ""


class BacktestResult(Record):
    """Simulator summary for one strategy on one split. available_at = end of the period."""

    strategy: str
    split: Literal["train", "validation", "test"]
    start: date
    end: date
    initial_capital: float
    final_equity: float
    total_return: float
    sharpe: float | None
    max_drawdown: float
    n_trades: int
    hit_rate: float | None
    equity_curve: list[tuple[date, float]] = Field(default_factory=list)
    annualised_return: float | None = None
    mean_daily_return: float | None = None
    sharpe_ci95: tuple[float, float] | None = Field(default=None, description="block bootstrap")
    mean_daily_return_ci95: tuple[float, float] | None = None
    turnover: float | None = Field(default=None, description="one-way, (buys+sells)/2/mean equity")
    n_days: int = 0
    costs_paid: float = 0.0
    tickers: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- API only


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    result: dict | None = None
    error: str | None = None


class WatchlistItem(BaseModel):
    ticker: str
    sector: str | None = None
    kind: Literal["stock", "crypto", "benchmark"] = "stock"


class Watchlist(BaseModel):
    items: list[WatchlistItem]
    benchmark: str


class BriefItem(BaseModel):
    ticker: str
    last_close: float | None
    change_pct: float | None
    sentiment_index: float | None
    events: list[Event] = Field(
        default_factory=list,
        description="classified events (events table, written by `make features`) in the window",
    )
    filings_7d: int = Field(
        default=0, description="8-K filings accepted in the window, counted from the filings table"
    )
    news_7d: int = Field(default=0, description="news items published in the window")
    signal: Signal | None = None


class Brief(BaseModel):
    as_of: AwareDatetime
    items: list[BriefItem]
    disclaimer: str = DISCLAIMER


class SignalsResponse(BaseModel):
    ticker: str
    as_of: AwareDatetime
    signals: list[Signal]
    aggregated: Signal | None
    risk_verdict: Literal["allow", "veto", "n/a"] = "n/a"
    disclaimer: str = DISCLAIMER


class EventsResponse(BaseModel):
    ticker: str
    filings: list[Filing]
    events: list[Event]
    facts: list[Fact]


class NewsEntry(BaseModel):
    """A news item with its sentiment row; cite it by `news.id`, `news.url`, `news.available_at`."""

    news: NewsItem
    sentiment: Sentiment | None = None


class NewsResponse(BaseModel):
    ticker: str
    as_of: AwareDatetime
    items: list[NewsEntry]
    disclaimer: str = DISCLAIMER


class PricesResponse(BaseModel):
    ticker: str
    as_of: AwareDatetime
    bars: list[PriceBar]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    ticker: str | None = None


class AnalyzeResponse(BaseModel):
    sentiment: Sentiment
    events: Event | None = Field(default=None, description="None until the event classifier exists")
    facts: list[Fact]
    method: Method
    disclaimer: str = DISCLAIMER


class CollectRequest(BaseModel):
    periods: list[Literal["warmup", "train", "validation", "test"]]


class ArenaRequest(BaseModel):
    split: Literal["train", "validation"]


class Citation(BaseModel):
    n: int = Field(default=0, ge=0, description="the [n] marker in the answer text (chat)")
    source_ref: str = Field(description="news id, 8-K accession number, or fact/price ref")
    kind: Literal["news", "filing", "fact", "price"]
    ticker: str | None = None
    available_at: AwareDatetime
    quote: str


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=2000, description="e.g. 'Why did AAPL move on 2026-07-15?'"
    )


ChatIntent = Literal["why_move", "price", "news", "signals", "backtest", "refuse"]


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    method: Method = Field(description="rules = template composer; llm = cached rephrase (D-46)")
    intent: ChatIntent
    ticker: str | None = None
    as_of: AwareDatetime | None = None
    tools_used: list[str] = Field(default_factory=list, description="tool plan, in call order")
    refused: bool = False
    refusal_reason: str | None = None
    disclaimer: str = DISCLAIMER


class EvalResult(BaseModel):
    """Mirror of evals/results/<component>.json (AGENTS.md section 5)."""

    component: str
    metric: str
    value: float
    target: float
    why_this_metric: str
    gold_set: str
    n: int
    split: str
    model: str
    git_sha: str
    run_at: datetime
    notes: str = ""
    details: dict | None = Field(default=None, description="component-specific extras")
    ci95: tuple[float, float] | None = Field(
        default=None, description="95% bootstrap CI of `value`, read from `details` when stored"
    )
    in_sample: bool = Field(
        default=False, description="the value was fitted or chosen on data it is reported on"
    )
    in_sample_note: str | None = None


class ModelInfo(BaseModel):
    id: str
    priority: int
    use_for: list[str]
    calls_today: int
    daily_budget: int
