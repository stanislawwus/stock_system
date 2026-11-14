"""Pure helpers that turn API responses into tables for the Streamlit pages (no I/O, testable)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.schemas import (
    Brief,
    EvalResult,
    EventsResponse,
    ModelInfo,
    NewsResponse,
    SignalsResponse,
)


def num(x: float | int | None, nd: int = 3) -> str:
    """A number for display; None is shown as n/a, never as 0."""
    if x is None:
        return "n/a"
    return str(x) if isinstance(x, int) else f"{x:.{nd}f}"


def ci_text(ci: tuple[float, float] | list[float] | None, nd: int = 2) -> str:
    return "n/a" if not ci else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def llm_budget_exhausted(models: list[ModelInfo]) -> bool:
    """True when today's LLM budget is used up. The app is then degraded, not broken: FinBERT,
    the rules and the template chat answers do not need the budget (D-46, D-48)."""
    return bool(models) and models[0].calls_today >= models[0].daily_budget


DEGRADED_BUDGET = (
    "Degraded mode: the daily LLM call budget is used up. Sentiment (FinBERT), event rules, "
    "signals and template chat answers keep working; LLM-assisted steps use their deterministic "
    "fallbacks."
)


def signals_table(resp: SignalsResponse) -> pd.DataFrame:
    """One row per strategy: score, position it implies (long-only), reasons."""
    rows = [
        {
            "strategy": s.strategy,
            "score": round(s.score, 3),
            "position": "long" if s.score > 0 else "cash",
            "reasons": ", ".join(s.reasons),
        }
        for s in resp.signals
    ]
    return pd.DataFrame(rows, columns=["strategy", "score", "position", "reasons"])


def equity_frame(rec: EvalResult) -> pd.DataFrame:
    """Equity curve stored in a backtest eval record's details, indexed by date."""
    curve = (rec.details or {}).get("equity_curve") or []
    df = pd.DataFrame(curve, columns=["date", "equity"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def backtest_summary(rec: EvalResult) -> dict[str, float | int | None]:
    d = rec.details or {}
    return {
        "Sharpe": rec.value,
        "buy-and-hold Sharpe": rec.target,
        "total return": d.get("total_return"),
        "max drawdown": d.get("max_drawdown"),
        "trades": d.get("n_trades"),
    }


def brief_table(brief: Brief) -> pd.DataFrame:
    """One row per watchlist stock. Missing data stays None so it renders blank, not 0."""
    rows = [
        {
            "ticker": i.ticker,
            "last close": None if i.last_close is None else round(i.last_close, 2),
            "change %": None if i.change_pct is None else round(i.change_pct, 2),
            "sentiment index": None if i.sentiment_index is None else round(i.sentiment_index, 3),
            "signal": None if i.signal is None else i.signal.strategy,
            "score": None if i.signal is None else round(i.signal.score, 3),
            "position": None if i.signal is None else ("long" if i.signal.score > 0 else "cash"),
            "8-K filings (7d)": i.filings_7d,
            "news (7d)": i.news_7d,
            "classified events (7d)": len(i.events),
        }
        for i in brief.items
    ]
    return pd.DataFrame(rows)


def filings_table(resp: EventsResponse) -> pd.DataFrame:
    """Filings with their event labels (joined on accession number); newest first."""
    labels = {e.source_ref: e for e in resp.events}
    rows = []
    for f in sorted(resp.filings, key=lambda f: f.available_at, reverse=True):
        ev = labels.get(f.accession_no)
        rows.append(
            {
                "available (UTC)": f.available_at.strftime("%Y-%m-%d %H:%M"),
                "form": f.form,
                "8-K items": ", ".join(f.items),
                "event labels": ", ".join(ev.labels) if ev else "",
                "label confidence": None if ev is None else round(ev.confidence, 2),
                "method": ev.method if ev else "",
                "accession": f.accession_no,
                "link": f.url,
            }
        )
    return pd.DataFrame(rows)


def facts_table(resp: EventsResponse) -> pd.DataFrame:
    rows = [
        {
            "fact": f.name,
            "value": f.value,
            "unit": f.unit,
            "period end": f.period_end,
            "fiscal period": f.fiscal_period,
            "method": f.method,
            "available (UTC)": f.available_at.strftime("%Y-%m-%d"),
            "source": f.source_ref,
        }
        for f in resp.facts
    ]
    return pd.DataFrame(rows)


def news_markdown(resp: NewsResponse) -> list[str]:
    """One citable line per item: headline link, source, availability time, sentiment, id."""
    lines = []
    for e in resp.items:
        n = e.news
        s = "unscored" if e.sentiment is None else f"{e.sentiment.method} {e.sentiment.score:+.2f}"
        head = n.headline.replace("[", "(").replace("]", ")")
        lines.append(
            f"- [{head}]({n.url}) · {n.source} · available {n.available_at:%Y-%m-%d %H:%M} UTC"
            f" · {s} · id `{n.id[:10]}`"
        )
    return lines


# --------------------------------------------------------------------------- evals


def evals_table(recs: list[EvalResult]) -> pd.DataFrame:
    """The Evals page table: value vs target, the CI where one is stored, the in-sample flag."""
    rows = []
    for r in recs:
        met = r.value >= r.target
        rows.append(
            {
                "component": r.component,
                "metric": r.metric,
                "value": round(r.value, 4),
                "target": round(r.target, 4),
                "95% CI": ci_text(r.ci95, 3),
                "meets target": "yes" if met else "no",
                "in-sample": "IN-SAMPLE" if r.in_sample else "no",
                "split": r.split,
                "n": r.n,
                "model": r.model,
                "run at": r.run_at.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return pd.DataFrame(rows)


def by_component(recs: list[EvalResult]) -> dict[str, EvalResult]:
    return {r.component: r for r in recs}


def arena_rows(details: dict[str, Any]) -> pd.DataFrame:
    """Leaderboard of the stored arena (details.leaderboard) with CIs and in-sample flags."""
    rows = []
    for r in details.get("leaderboard", []):
        rows.append(
            {
                "strategy": r["strategy"],
                "Sharpe": r.get("sharpe"),
                "Sharpe 95% CI": ci_text(r.get("sharpe_ci95")),
                "Sharpe val_b": r.get("sharpe_val_b"),
                "val_b 95% CI": ci_text(r.get("sharpe_val_b_ci95")),
                "return": r.get("total_return"),
                "max drawdown": r.get("max_drawdown"),
                "round trips": r.get("round_trips"),
                "exposure": r.get("mean_exposure"),
                "in-sample on validation": "IN-SAMPLE" if r.get("validation_in_sample") else "no",
                "chosen on": r.get("choice_period", ""),
            }
        )
    return pd.DataFrame(rows)


def equity_curves(recs: list[EvalResult], names: list[str]) -> pd.DataFrame:
    """Equity of several strategies on one date index, each rebased to 1.0 on its first day."""
    cols = {}
    for r in recs:
        name = (r.details or {}).get("strategy", r.component)
        df = equity_frame(r)
        if name in names and not df.empty:
            cols[name] = df["equity"] / df["equity"].iloc[0]
    return pd.DataFrame(cols).sort_index()


def replay_table(curves: pd.DataFrame, upto: pd.Timestamp) -> pd.DataFrame:
    """Standing of every strategy on one day of the replay: cumulative return, drawdown so far."""
    part = curves[curves.index <= upto]
    rows = []
    for name in part.columns:
        s = part[name].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "strategy": name,
                "cumulative return": float(s.iloc[-1] - 1),
                "max drawdown so far": float((s / s.cummax() - 1).min()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- in-sample status


def sample_status(
    day: date, strategy: str, evals: dict[str, EvalResult]
) -> tuple[bool | None, str]:
    """Is `strategy`'s signal on decision day `day` in-sample? (flag or None, one sentence)

    Read from the stored records only: the arena leaderboard flags strategies whose parameters
    were chosen on val_a; a day before validation is checked against the strategy's train backtest
    flag. None means the records do not say.
    """
    arena = evals.get("arena")
    det = (arena.details or {}) if arena else {}
    if det.get("start") and day >= date.fromisoformat(det["start"]):
        row = next((r for r in det.get("leaderboard", []) if r["strategy"] == strategy), None)
        if row is None:
            return None, f"no arena record for {strategy}: in-sample status unknown"
        vb = date.fromisoformat(det["val_b_start"]) if det.get("val_b_start") else None
        if row.get("validation_in_sample") and (vb is None or day < vb):
            return True, f"IN-SAMPLE: {strategy} was {row.get('choice_period')} (days before {vb})"
        return False, "validation day no choice was fitted on (val_b or a fixed-a-priori strategy)"
    rec = evals.get(f"backtest_{strategy}_train")
    if rec is None:
        return None, "no train record for this strategy: in-sample status unknown"
    if rec.in_sample:
        return True, f"IN-SAMPLE: {rec.in_sample_note}"
    return False, "train day, strategy has no fitted parameters"
