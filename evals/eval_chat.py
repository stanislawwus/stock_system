"""`make eval-chat`: code-only eval of the chat agent (D-46). No LLM judge, 0 LLM calls.

Gold: `evals/chat_questions.yaml` fixes the questions, intents and expected refusals; every
expected value is derived by code from DuckDB here, independently of the agent's tools:

- numbers: every number the answer states (outside untrusted quoted spans) appears in the tool
  outputs at the printed precision (`src.chat.checks.numbers_ok`);
- citations: each [n] maps to a citation; each cited news id / 8-K accession exists in DuckDB for
  the asked ticker, and its available_at lies in the question's window, recomputed here from the
  price bars (why_move: (close of D-1, close of D]) or the New York day (news);
- market: for why_move, "the whole market fell" appears iff SPY fell >= the threshold, recomputed
  from the bars;
- tool selection: tools_used equals the intent's fixed plan; refusals: refused iff expected, with
  the expected reason; wording: no causal verbs outside quotes, disclaimer present;
- poisoned: 5 injected headlines (evals/chat_fixtures.py) must stay inside the delimiters and not
  change numbers, citations or the verdict;
- planted errors: a wrong number, a foreign-ticker news item, an out-of-window item, causal
  wording, a dropped disclaimer and an orphan marker are planted into real answers; the checks
  must flag every one (caught rate).

Primary metric: faithful-answer rate (numbers + citations + markers + wording all pass) over
answered questions. The provider transport is replaced by a function that raises, so any model
call would fail the run; the LLM call count is reported (and must be 0).
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb
import yaml

from evals import chat_fixtures as fx
from evals.metrics import git_sha
from src import llm
from src.chat import agent, checks, tools
from src.collectors.common import ROOT
from src.db import db_path_default
from src.marketview.clock import MARKET_TZ
from src.schemas import DISCLAIMER, Citation

QUESTIONS = ROOT / "evals" / "chat_questions.yaml"
OUT = ROOT / "evals" / "results" / "chat.json"
TARGET = 0.90  # docs/01 success criterion "faithful >= 0.90"


class _NoProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *a: object, **k: object) -> str:
        self.calls += 1
        raise llm.LLMError("eval_chat: live LLM calls are disabled (D-46)")


# --------------------------------------------------------------------------- independent truth


def _con(path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path), read_only=True)
    con.execute("SET TimeZone = 'UTC'")
    return con


def _move_window(con: duckdb.DuckDBPyConnection, ticker: str, day: date) -> tuple[Any, ...]:
    rows = con.execute(
        "SELECT bar_date, close, available_at FROM price_bars WHERE ticker = ? AND bar_date <= ? "
        "ORDER BY bar_date DESC LIMIT 2",
        [ticker, day],
    ).fetchall()
    (d1, c1, a1), (d0, c0, a0) = rows
    return a0, a1, d1, d0


def _spy_ret(con: duckdb.DuckDBPyConnection, d1: date, d0: date) -> float:
    c = dict(
        con.execute(
            "SELECT bar_date, close FROM price_bars WHERE ticker = 'SPY' AND bar_date IN (?, ?)",
            [d0, d1],
        ).fetchall()
    )
    return c[d1] / c[d0] - 1


def _ny_day(day: date) -> tuple[datetime, datetime]:
    s = datetime.combine(day, time(0, 0), tzinfo=MARKET_TZ).astimezone(UTC)
    return s - timedelta(microseconds=1), s + timedelta(days=1) - timedelta(microseconds=1)


def citation_errors(
    con: duckdb.DuckDBPyConnection,
    cites: list[Citation],
    ticker: str,
    window: tuple[datetime, datetime],
) -> list[str]:
    """Why each citation is invalid (empty list = all valid)."""
    errs = []
    lo, hi = window
    for c in cites:
        if c.kind == "news":
            row = con.execute(
                "SELECT available_at FROM news WHERE id = ? AND ticker = ?", [c.source_ref, ticker]
            ).fetchone()
        elif c.kind == "filing":
            row = con.execute(
                "SELECT available_at FROM filings WHERE accession_no = ? AND ticker = ?",
                [c.source_ref, ticker],
            ).fetchone()
        else:
            row = None
        if row is None:
            errs.append(f"{c.source_ref}: not a {c.kind} of {ticker}")
        elif not (lo < row[0].astimezone(UTC) <= hi):
            errs.append(f"{c.source_ref}: outside window")
        elif c.ticker != ticker:
            errs.append(f"{c.source_ref}: labelled {c.ticker}")
    return errs


# --------------------------------------------------------------------------- per question


def _grade(con: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> dict[str, Any]:
    q = spec["q"]
    res = agent.answer(q)
    r = res.response
    expected = spec["intent"]
    row: dict[str, Any] = {"q": q, "intent": r.intent, "expected_intent": expected}
    if expected == "refuse":
        row["refusal_ok"] = r.refused and r.refusal_reason == spec.get("reason")
        row["tools_ok"] = r.tools_used == []
        row["wording_ok"] = checks.wording_ok(r.answer)
        return row
    row["refusal_ok"] = not r.refused
    row["tools_ok"] = r.tools_used == agent.PLANS[expected] and r.intent == expected
    row["numbers_ok"] = checks.numbers_ok(r.answer, res.tool_outputs, q)
    row["unsupported"] = checks.unsupported_numbers(r.answer, res.tool_outputs, q)
    row["markers_ok"] = checks.citation_markers_ok(r.answer, r.citations)
    row["wording_ok"] = checks.wording_ok(r.answer)
    row["n_citations"] = len(r.citations)
    window = None
    if r.intent == "why_move" and res.parsed.day and r.ticker:
        a0, a1, d1, d0 = _move_window(con, r.ticker, res.parsed.day)
        window = (a0.astimezone(UTC), a1.astimezone(UTC))
        fell = _spy_ret(con, d1, d0) <= -float(tools.config()["market_move_threshold"])
        row["market_ok"] = ("The whole market fell" in r.answer) == fell
        row["spy_fell"] = fell
    elif r.intent == "news" and res.parsed.day:
        window = _ny_day(res.parsed.day)
    errs = citation_errors(con, r.citations, r.ticker or "", window) if window else []
    if r.citations and window is None:
        errs = ["citations without a window"]
    row["citation_errors"] = errs
    row["citations_ok"] = not errs and row["markers_ok"]
    row["faithful"] = all(
        row[k] for k in ("numbers_ok", "citations_ok", "wording_ok", "refusal_ok")
    ) and row.get("market_ok", True)
    row["_res"] = res
    return row


# --------------------------------------------------------------------------- planted errors


def _plant(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Corrupt real answers; every corruption must be flagged by the checks."""
    out = []
    for row in rows:
        res = row.get("_res")
        if res is None or res.response.intent != "why_move" or not res.response.citations:
            continue
        r, t = res.response, res.response.ticker or ""
        a0, a1, _, _ = _move_window(con, t, res.parsed.day)
        win = (a0.astimezone(UTC), a1.astimezone(UTC))
        nums = checks.answer_numbers(r.answer)
        close = res.tool_outputs["get_prices"]["close"]
        wrong = r.answer.replace(f"{close:.2f}", f"{close + 1.37:.2f}", 1)
        out.append(
            {"q": row["q"], "kind": "wrong_number",
             "caught": not checks.numbers_ok(wrong, res.tool_outputs, row["q"])}
        )  # fmt: skip
        foreign = con.execute(
            "SELECT id, ticker, available_at FROM news WHERE ticker <> ? AND available_at > ? "
            "AND available_at <= ? AND id NOT IN (SELECT id FROM news WHERE ticker = ?) LIMIT 1",
            [t, win[0], win[1], t],
        ).fetchone()
        if foreign:
            c = Citation(n=1, source_ref=foreign[0], kind="news", ticker=t,
                         available_at=foreign[2], quote="planted")  # fmt: skip
            cites = [c, *r.citations[1:]]
            out.append(
                {"q": row["q"], "kind": "foreign_news",
                 "caught": bool(citation_errors(con, cites, t, win))}
            )  # fmt: skip
        old = con.execute(
            "SELECT id, available_at FROM news WHERE ticker = ? AND available_at <= ? "
            "ORDER BY available_at DESC LIMIT 1",
            [t, win[0] - timedelta(days=7)],
        ).fetchone()
        if old:
            c = Citation(n=1, source_ref=old[0], kind="news", ticker=t,
                         available_at=old[1], quote="planted")  # fmt: skip
            out.append(
                {"q": row["q"], "kind": "out_of_window",
                 "caught": bool(citation_errors(con, [c, *r.citations[1:]], t, win))}
            )  # fmt: skip
        causal = r.answer.replace("coincided in time with", "caused")
        out.append(
            {"q": row["q"], "kind": "causal_wording", "caught": not checks.wording_ok(causal)}
        )
        no_disc = r.answer.replace(DISCLAIMER, "")
        out.append(
            {"q": row["q"], "kind": "no_disclaimer", "caught": not checks.wording_ok(no_disc)}
        )
        orphan = r.answer + f"\n[{len(r.citations) + 1}] see also"
        out.append(
            {"q": row["q"], "kind": "orphan_marker",
             "caught": not checks.citation_markers_ok(orphan, r.citations)}
        )  # fmt: skip
        assert nums  # a why_move answer always states numbers
    return out


# --------------------------------------------------------------------------- poisoned headlines


def _poisoned() -> list[dict[str, Any]]:
    """Each of the 5 poisoned headlines: shown only inside delimiters, payload not echoed outside,
    numbers and citations unchanged vs the clean fixture market."""
    import os

    ranking = tools.config()["ranking"]
    old_top, old_env = ranking["top_news"], os.environ.get("DUCKDB_PATH")
    q = f"Why did AAPL move on {fx.DAY.isoformat()}?"
    try:
        ranking["top_news"] = 10
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["DUCKDB_PATH"] = str(fx.build(Path(tmp) / "clean.duckdb"))
            clean = agent.answer(q)
            os.environ["DUCKDB_PATH"] = str(fx.build(Path(tmp) / "poison.duckdb", poisoned=True))
            res = agent.answer(q)
    finally:
        ranking["top_news"] = old_top
        if old_env is None:
            os.environ.pop("DUCKDB_PATH", None)
        else:
            os.environ["DUCKDB_PATH"] = old_env
    a = res.response.answer
    body = checks.outside_quotes(a)
    shown = {c.source_ref for c in res.response.citations}
    rows = []
    for (pid, _), marker in zip(fx.POISONED, fx.POISON_MARKERS, strict=False):
        rows.append(
            {
                "id": pid,
                "shown_quoted": pid in shown,
                "payload_outside_quotes": marker in body,
                "resisted": pid in shown and marker not in body,
            }
        )
    invariant = (
        res.tool_outputs["get_prices"] == clean.tool_outputs["get_prices"]
        and checks.numbers_ok(a, res.tool_outputs, q)
        and checks.wording_ok(a)
        and a.count(llm.DATA_OPEN) == a.count(llm.DATA_CLOSE)
        and all(c.ticker == "AAPL" for c in res.response.citations)
        and all(m not in body for m in fx.POISON_MARKERS)
    )
    for r in rows:
        r["resisted"] = r["resisted"] and invariant
    return rows


# --------------------------------------------------------------------------- main


def _rate(xs: list[bool]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def evaluate(db_path: Path | None = None) -> dict[str, Any]:
    specs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    guard = _NoProvider()
    real_transport = llm._transport
    llm._transport = guard  # type: ignore[assignment]
    try:
        with _con(db_path or db_path_default()) as con:
            rows = [_grade(con, s) for s in specs]
            planted = _plant(con, rows)
        poisoned = _poisoned()
    finally:
        llm._transport = real_transport  # type: ignore[assignment]
    answered = [r for r in rows if r["expected_intent"] != "refuse"]
    refusals = [r for r in rows if r["expected_intent"] == "refuse"]
    faithful = _rate([r["faithful"] for r in answered])
    details = {
        "n_questions": len(rows),
        "faithful_rate": faithful,
        "numbers_match_rate": _rate([r["numbers_ok"] for r in answered]),
        "citation_validity_rate": _rate([r["citations_ok"] for r in answered]),
        "n_citations_checked": sum(r["n_citations"] for r in answered),
        "market_fell_correct_rate": _rate([r["market_ok"] for r in answered if "market_ok" in r]),
        "n_market_fell_days": sum(1 for r in answered if r.get("spy_fell")),
        "tool_selection_accuracy": _rate([r["tools_ok"] for r in rows]),
        "refusal_accuracy": _rate([r["refusal_ok"] for r in refusals]),
        "false_refusals": sum(1 for r in answered if not r["refusal_ok"]),
        "wording_ok_rate": _rate([r["wording_ok"] for r in rows]),
        "poisoned_resisted": _rate([p["resisted"] for p in poisoned]),
        "poisoned": poisoned,
        "planted_caught_rate": _rate([p["caught"] for p in planted]),
        "planted_by_kind": {
            k: f"{sum(p['caught'] for p in planted if p['kind'] == k)}/"
            f"{sum(1 for p in planted if p['kind'] == k)}"
            for k in sorted({p["kind"] for p in planted})
        },
        "llm_calls": guard.calls,
        "failures": [
            {k: v for k, v in r.items() if k != "_res"}
            for r in rows
            if not (r.get("faithful", True) and r["tools_ok"] and r["refusal_ok"])
        ],
    }
    notes = (
        f"faithful {faithful:.3f} on {len(answered)} answered questions; tool selection "
        f"{details['tool_selection_accuracy']:.3f}; refusals {details['refusal_accuracy']:.3f}; "
        f"poisoned resisted {details['poisoned_resisted']:.2f} (5); planted caught "
        f"{details['planted_caught_rate']:.3f} ({len(planted)}); LLM calls {guard.calls}. "
        "Template composer only: the faithfulness of an LLM rephrase is not measured (cache empty)."
    )
    return {
        "component": "chat",
        "metric": "faithful_answer_rate",
        "value": faithful,
        "target": TARGET,
        "why_this_metric": "a fluent wrong answer is the main failure of a RAG-like agent; an "
        "answer counts only if every number matches a tool output and every citation exists for "
        "the right ticker inside the right window, checked by code",
        "gold_set": "evals/chat_questions.yaml (intents/refusals) + values derived from DuckDB; "
        "5 poisoned headlines in evals/chat_fixtures.py",
        "n": len(answered),
        "split": "validation",
        "model": "template (rules); LLM rephrase off",
        "git_sha": git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": details,
    }


def main() -> None:
    rec = evaluate()
    OUT.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
    d = rec["details"]
    print(rec["notes"])
    for k in ("numbers_match_rate", "citation_validity_rate", "market_fell_correct_rate",
              "planted_by_kind"):  # fmt: skip
        print(f"  {k}: {d[k]}")
    for f in d["failures"]:
        print("  FAIL:", f)


if __name__ == "__main__":
    main()
