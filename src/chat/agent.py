"""Chat agent behind POST /api/v1/chat (AGENTS.md R2, R5, R6, R11, R13; D-46).

A fixed plan, no model in the loop:

1. parse the question (rules): intent, ticker (universe only), day; refuse unknown tickers and
   requests for guaranteed profit;
2. run the intent's tool plan (`PLANS`) on a `MarketView(as_of)` inside `src.db.guarded()`;
   for "why did X move on D?" as_of = 16:00 ET of D (the bar close: the move is known then);
3. compose the answer with a deterministic template built ONLY from tool outputs, with numbered
   citations [n] (news id or 8-K accession number); untrusted headlines are shown inside the
   gateway's delimiters; wording is "coincided in time", never causal; the disclaimer is appended;
4. optional rephrase (flag `rephrase.enabled` or env CHAT_REPHRASE=1): a cached LLM response is
   replayed via `src.llm.cached_json` (no provider call, ever, on this path) and accepted only if
   `checks.rephrase_ok` finds the same numbers, citations and quoted data; otherwise the template.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import duckdb
from pydantic import BaseModel, Field

from src import llm
from src.chat import tools
from src.chat.checks import rephrase_ok
from src.collectors.common import load_periods
from src.db import guarded
from src.marketview import MarketView
from src.marketview.clock import MARKET_TZ, bar_close_for
from src.nlp.intake import extract_tickers
from src.schemas import DISCLAIMER, ChatIntent, ChatResponse, Citation
from src.splits import TestPeriodLockedError
from src.strategies.registry import BUILDERS
from src.strategies.service import DataUnavailableError, known_tickers
from src.textutil import strip_html

__all__ = ["PLANS", "ChatResult", "DataUnavailableError", "Parsed", "answer", "parse"]

PLANS: dict[str, list[str]] = {
    "why_move": ["get_prices", "get_news", "get_events"],
    "price": ["get_prices"],
    "news": ["get_news", "get_events"],
    "signals": ["get_signals"],
    "backtest": ["run_backtest_summary"],
    "refuse": [],
}
COINCIDE = (
    "These items coincided in time with the move; co-occurrence in time is not evidence of a "
    "causal link."
)
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TEXT_DATE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Parsed:
    intent: ChatIntent
    ticker: str | None = None
    day: date | None = None
    strategy: str | None = None
    split: str = "validation"
    refusal_reason: str | None = None


@dataclass
class ChatResult:
    """The API response plus everything the code checks need (tests, eval)."""

    response: ChatResponse
    parsed: Parsed
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    window: tuple[datetime, datetime] | None = None


class Rephrased(BaseModel):
    answer: str = Field(description="the same answer, rephrased; numbers and [n] markers kept")


REPHRASE_PROMPT = (
    "Rephrase the answer below for a retail investor in plain English. Keep every number, every "
    "citation marker like [1], every date and every quoted block exactly as written. Do not add "
    "facts. Never say a news item caused, drove or explains a move; say it coincided in time."
)


# --------------------------------------------------------------------------- parsing


def _cfg() -> dict[str, Any]:
    return tools.config()


def default_day() -> date:
    (val,) = load_periods(["validation"])
    return val.end


def _parse_day(q: str) -> date | None:
    m = _ISO.search(q)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    m = _TEXT_DATE.search(q)
    if m:
        year = int(m[3]) if m[3] else default_day().year
        try:
            return date(year, MONTHS[m[1][:3].lower()], int(m[2]))
        except ValueError:
            return None
    return None


def _unknown_symbols(q: str, known: set[str]) -> list[str]:
    stop = {str(s) for s in _cfg()["not_tickers"]}
    dollar = [s.upper() for s in re.findall(r"\$([A-Za-z]{1,6})\b", q)]
    caps = re.findall(r"(?<![\w$])([A-Z]{2,6})(?![\w])", _ISO.sub(" ", q))
    cands = dollar + [c for c in caps if c not in stop]
    return [c for c in dict.fromkeys(cands) if c not in known]


def _strategy(q: str) -> str | None:
    low = q.lower().replace("-", "_").replace(" ", "_")
    for name in sorted(BUILDERS, key=len, reverse=True):
        if name in low:
            return name
    return None


def parse(question: str) -> Parsed:
    """Rules only: intent, ticker, day; refusals first (R11, docs/01 scope)."""
    q = question.strip()
    low = q.lower()
    for pat in _cfg()["refuse_patterns"]["guaranteed_profit"]:
        if re.search(pat, low):
            return Parsed("refuse", refusal_reason="guaranteed_profit")
    known = known_tickers()
    strategy = _strategy(q)
    split = "train" if re.search(r"\btrain(ing)?\b", low) else "validation"
    if strategy and re.search(r"backtest|sharpe|perform|return|drawdown|strategy", low):
        return Parsed("backtest", strategy=strategy, split=split)
    found = [t for t in extract_tickers(q) if t in known]
    if not found:
        found = [t for t in re.findall(r"(?<![\w$])\$?([A-Z]{1,5})\b", q) if t in known]
    unknown = _unknown_symbols(q, known)
    if not found:
        reason = "unknown_ticker" if unknown else "no_ticker"
        return Parsed("refuse", refusal_reason=reason)
    ticker = found[0]
    day = _parse_day(q) or default_day()
    if re.search(r"\bsignal|suggest|recommend|should i (buy|sell|hold)", low):
        return Parsed("signals", ticker=ticker, day=day)
    if re.search(r"\bwhy\b|\bmove|jump|drop|fell|fall|rose|rise|surge|plunge|slump|rall", low):
        return Parsed("why_move", ticker=ticker, day=day)
    if re.search(r"\bnews\b|headline|filing|8-k", low):
        return Parsed("news", ticker=ticker, day=day)
    if re.search(r"\bclose|closing|price\b", low):
        return Parsed("price", ticker=ticker, day=day)
    return Parsed("why_move", ticker=ticker, day=day)


# --------------------------------------------------------------------------- composing


def _pct(x: float) -> str:
    return f"{abs(x) * 100:.2f}%"


def _signed_pp(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}{abs(x) * 100:.2f}"


def _ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M")


def quote_untrusted(text: str) -> str:
    """Headline as data: HTML stripped, delimiters removed from inside, truncated, wrapped (R6)."""
    clean = strip_html(text).replace(llm.DATA_OPEN, " ").replace(llm.DATA_CLOSE, " ")
    clean = " ".join(clean.split())
    cap = int(_cfg()["headline_max_chars"])
    if len(clean) > cap:
        clean = clean[:cap] + " [truncated]"
    return f"{llm.DATA_OPEN} {clean} {llm.DATA_CLOSE}"


def _items_block(
    news: list[tools.RankedNews], filings: list[tools.RankedFiling]
) -> tuple[list[str], list[Citation]]:
    lines: list[str] = []
    cites: list[Citation] = []
    for n in news:
        k = len(cites) + 1
        sent = "n/a" if n.sentiment is None else f"{n.sentiment:+.2f}"
        lines.append(
            f"[{k}] news, {n.source}, {_ts(n.available_at)} UTC, FinBERT sentiment {sent}: "
            f"{quote_untrusted(n.headline)}"
        )
        cites.append(
            Citation(
                n=k,
                source_ref=n.id,
                kind="news",
                ticker=n.ticker,
                available_at=n.available_at,
                quote=quote_untrusted(n.headline),
            )
        )
    for f in filings:
        k = len(cites) + 1
        items = ", ".join(f.items) if f.items else "none listed"
        lines.append(
            f"[{k}] {f.form} {f.accession_no}, accepted {_ts(f.available_at)} UTC, items {items}"
        )
        cites.append(
            Citation(
                n=k,
                source_ref=f.accession_no,
                kind="filing",
                ticker=f.ticker,
                available_at=f.available_at,
                quote=f"{f.form} items {items}",
            )
        )
    return lines, cites


def compose_why_move(
    m: tools.PriceMove, news: list[tools.RankedNews], filings: list[tools.RankedFiling], asked: date
) -> tuple[str, list[Citation]]:
    cfg = _cfg()
    t, b = m.ticker, m.benchmark
    out: list[str] = []
    if m.day != asked:
        out.append(f"{asked.isoformat()} was not a trading session; using {m.day.isoformat()}.")
    direction = "up" if m.ret >= 0 else "down"
    out.append(
        f"{t} closed at {m.close:.2f} on {m.day.isoformat()}, {direction} {_pct(m.ret)} from "
        f"{m.prev_close:.2f} on {m.prev_day.isoformat()}."
    )
    out.append(
        f"{b} moved {_signed_pp(m.bench_ret)}% over the same sessions ({m.bench_prev_close:.2f} "
        f"to {m.bench_close:.2f}), so the abnormal return of {t} vs {b} was "
        f"{_signed_pp(m.abnormal_ret)} percentage points."
    )
    thr = float(cfg["market_move_threshold"])
    if m.bench_ret <= -thr:
        out.append(f"The whole market fell that day ({b} down {_pct(m.bench_ret)}).")
    elif m.bench_ret >= thr:
        out.append(f"The whole market rose that day ({b} up {_pct(m.bench_ret)}).")
    if abs(m.abnormal_ret) < float(cfg["small_abnormal_threshold"]):
        out.append(f"Relative to {b}, the move of {t} was small: mostly in line with the market.")
    lines, cites = _items_block(news, filings)
    if lines:
        out.append(
            f"News and filings published between the {m.prev_day.isoformat()} close and the "
            f"{m.day.isoformat()} close that coincided in time with the move "
            "(ranked by relevance, sentiment and recency; headlines are untrusted quoted data):"
        )
        out.extend(lines)
        out.append(COINCIDE)
    else:
        out.append(
            f"No relevant news or 8-K filings for {t} were found between the "
            f"{m.prev_day.isoformat()} close and the {m.day.isoformat()} close."
        )
    return "\n".join(out), cites


def compose_news(
    ticker: str, day: date, news: list[tools.RankedNews], filings: list[tools.RankedFiling]
) -> tuple[str, list[Citation]]:
    lines, cites = _items_block(news, filings)
    if not lines:
        return f"No relevant news or 8-K filings for {ticker} on {day.isoformat()} (ET).", []
    head = (
        f"Top news and 8-K filings for {ticker} on {day.isoformat()} (ET), ranked by relevance, "
        "sentiment and recency (headlines are untrusted quoted data):"
    )
    return "\n".join([head, *lines]), cites


def compose_price(m: tools.PriceMove, asked: date) -> str:
    pre = ""
    if m.day != asked:
        pre = f"{asked.isoformat()} was not a trading session; using {m.day.isoformat()}. "
    direction = "up" if m.ret >= 0 else "down"
    return (
        f"{pre}{m.ticker} closed at {m.close:.2f} on {m.day.isoformat()}, {direction} "
        f"{_pct(m.ret)} from {m.prev_close:.2f} on {m.prev_day.isoformat()}."
    )


def compose_signals(ticker: str, resp: Any) -> str:
    lines = [
        f"Strategy suggestions for {ticker} as of {_ts(resp.as_of)} UTC (09:00 ET cutoff); "
        "a score <= 0 means cash (long-only):"
    ]
    for s in sorted(resp.signals, key=lambda s: s.strategy):
        p = "" if s.calibrated_p is None else f", calibrated p {s.calibrated_p:.2f}"
        lines.append(f"- {s.strategy}: score {s.score:+.2f}{p}")
    lines.append("These are suggestions with a confidence, not predictions.")
    return "\n".join(lines)


def compose_backtest(s: tools.BacktestSummary) -> str:
    return (
        f"Simulated backtest of {s.strategy} on the {s.split} split ({s.start} to {s.end}): total "
        f"return {_signed_pp(s.total_return)}%, Sharpe {s.sharpe:.2f} vs {s.benchmark_sharpe:.2f} "
        f"for buy-and-hold on the same names and days, max drawdown {_pct(s.max_drawdown)}, "
        f"{s.n_trades} trades. Past simulated results say nothing certain about the future."
    )


REFUSALS = {
    "guaranteed_profit": "I cannot promise or suggest guaranteed profits: no strategy here is "
    "risk-free, and every output is a simulation with uncertain results.",
    "unknown_ticker": "I can only answer about tickers in the configured universe "
    "(config/universe.yaml); the symbol in the question is not one of them.",
    "no_ticker": "Please name a ticker or company from the configured universe "
    "(config/universe.yaml), e.g. 'Why did AAPL move on 2026-07-15?'.",
    "test_period": "That date is in the locked test period; it is read exactly once by the final "
    "evaluation run (AGENTS.md R3). Ask about a date up to the end of the validation period.",
    "unknown_strategy": "There is no committed backtest record for that strategy and split.",
}


def _finish(text: str) -> str:
    return f"{text}\n{DISCLAIMER}"


def _refuse(parsed: Parsed, reason: str) -> ChatResult:
    p = Parsed("refuse", ticker=parsed.ticker, day=parsed.day, refusal_reason=reason)
    resp = ChatResponse(
        answer=_finish(REFUSALS[reason]),
        citations=[],
        method="rules",
        intent="refuse",
        ticker=parsed.ticker,
        refused=True,
        refusal_reason=reason,
    )
    return ChatResult(resp, p)


def _rephrase_enabled() -> bool:
    env = os.environ.get("CHAT_REPHRASE")
    if env is not None:
        return env.strip() in {"1", "true", "yes"}
    return bool(_cfg()["rephrase"]["enabled"])


def _maybe_rephrase(text: str) -> tuple[str, str]:
    """(answer, method). Cache replay only; any mismatch keeps the template (fallback)."""
    if not _rephrase_enabled():
        return text, "rules"
    got = llm.cached_json("chat", REPHRASE_PROMPT, [{"answer": text}], Rephrased)
    if got and rephrase_ok(got[0].answer, text):
        return got[0].answer, "llm"
    return text, "rules"


def _day_window(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(0, 0), tzinfo=MARKET_TZ).astimezone(UTC)
    return start - timedelta(microseconds=1), start + timedelta(days=1) - timedelta(microseconds=1)


def answer(question: str, db_path: Any = None) -> ChatResult:
    """Answer `question` from tool outputs only. Raises DataUnavailableError (no DB / writer)."""
    parsed = parse(question)
    if parsed.intent == "refuse":
        return _refuse(parsed, parsed.refusal_reason or "no_ticker")
    outputs: dict[str, Any] = {}
    window: tuple[datetime, datetime] | None = None
    cites: list[Citation] = []
    as_of: datetime | None = None
    try:
        if parsed.intent == "backtest":
            assert parsed.strategy is not None
            s = tools.run_backtest_summary(parsed.strategy, parsed.split)
            if s is None:
                return _refuse(parsed, "unknown_strategy")
            outputs["run_backtest_summary"] = s.model_dump(mode="json")
            text = compose_backtest(s)
        elif parsed.intent == "signals":
            assert parsed.ticker and parsed.day
            resp = tools.get_signals(parsed.ticker, parsed.day)
            outputs["get_signals"] = resp.model_dump(mode="json")
            as_of = resp.as_of
            text = compose_signals(parsed.ticker, resp)
        else:
            assert parsed.ticker and parsed.day
            t, d = parsed.ticker, parsed.day
            as_of = bar_close_for(d) if parsed.intent != "news" else _day_window(d)[1]
            with MarketView(as_of, db_path) as view, guarded():
                if parsed.intent == "news":
                    window = _day_window(d)
                    news = tools.get_news(view, t, *window)
                    filings = tools.get_events(view, t, *window)
                    outputs["get_news"] = [n.model_dump(mode="json") for n in news]
                    outputs["get_events"] = [f.model_dump(mode="json") for f in filings]
                    text, cites = compose_news(t, d, news, filings)
                else:
                    move = tools.get_prices(view, t, d)
                    outputs["get_prices"] = None if move is None else move.model_dump(mode="json")
                    if move is None:
                        text = f"No price data for {t} around {d.isoformat()} in the database."
                    elif parsed.intent == "price":
                        text = compose_price(move, d)
                    else:
                        window = (move.window_start, move.window_end)
                        news = tools.get_news(view, t, *window)
                        filings = tools.get_events(view, t, *window)
                        outputs["get_news"] = [n.model_dump(mode="json") for n in news]
                        outputs["get_events"] = [f.model_dump(mode="json") for f in filings]
                        text, cites = compose_why_move(move, news, filings, d)
    except TestPeriodLockedError:
        return _refuse(parsed, "test_period")
    except FileNotFoundError as exc:
        raise DataUnavailableError(str(exc)) from exc
    except duckdb.IOException as exc:
        raise DataUnavailableError(f"database busy (a writer holds it): {exc}") from exc
    body, method = _maybe_rephrase(text)
    resp = ChatResponse(
        answer=_finish(body),
        citations=cites,
        method=method,  # type: ignore[arg-type]
        intent=parsed.intent,
        ticker=parsed.ticker,
        as_of=as_of,
        tools_used=list(outputs),
    )
    return ChatResult(resp, parsed, outputs, window)
