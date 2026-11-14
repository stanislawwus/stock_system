"""Eval: fact extraction from EX-99.1 earnings releases (src/nlp/facts.py), regex vs LLM.

- **Metric**: share of gold (filing, metric) pairs whose extracted value is within +/- 1 % of
  XBRL (`within_1pct`); also exact match at the stated precision (XBRL rounded to the decimals
  the release wrote equals the extracted number), grounding rate (share of extracted facts whose
  quote is in the source and holds the value, `is_grounded`), and period match (quarter number).
- **Gold set**: derived, XBRL companyfacts (`facts` table, method xbrl) for 30 dev-split 8-K
  item 2.02 filings: one per ticker first, fixed seed; gold = the 3-month revenue / net income /
  diluted EPS of the latest fiscal quarter (Q1-Q3; XBRL has no Q4 quarter) that ended at most
  `gold_max_days_after_period` days before the 8-K was accepted.
- **Target**: >= 0.90 within 1 %, 100 % grounded (docs/03_eval_plan.md).
- **Why**: a number is right or useless; 1 % covers the release's rounding ("$124.3 billion");
  grounding makes every wrong value inspectable.

`value` is the regex baseline on all 30 (deterministic; also the LLM fallback). The LLM runs on
the first `llm_n_filings` (15) through src/llm.py, batched, cached; both methods are compared on
those 15 in `details.head_to_head`. Without an API key or budget the LLM part is the fallback and
says so.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from evals.metrics import git_sha
from src import db
from src.collectors.common import ROOT, load_periods
from src.marketview.clock import cutoff_for
from src.nlp.facts import FilingFacts, config, extract_llm, extract_regex, is_grounded

TARGET = 0.90
METRICS = ("revenue", "net_income", "eps_diluted")


def gold_filings(db_path: Path | None = None) -> list[dict[str, Any]]:
    """The 30 gold filings with their EX-99.1 text and XBRL values (deterministic)."""
    c = config()
    (validation,) = load_periods(["validation"])
    con = db.connect(db_path)
    try:
        fil = con.execute(
            "SELECT accession_no, ticker, accepted_at, text_path FROM filings "
            "WHERE form = '8-K' AND list_contains(items, '2.02') AND text_path IS NOT NULL "
            "AND accepted_at <= ? ORDER BY ticker, accepted_at",
            [cutoff_for(validation.end)],  # dev splits only (R3)
        ).fetchall()
        facts = con.execute(
            "SELECT ticker, name, value, period_end, fiscal_period FROM facts "
            "WHERE method = 'xbrl' AND fiscal_period LIKE 'Q%'"
        ).fetchall()
    finally:
        con.close()
    by_ticker: dict[str, list[tuple]] = {}
    for f in facts:
        by_ticker.setdefault(f[0], []).append(f)
    max_days = timedelta(days=int(c["gold_max_days_after_period"]))
    cands = []
    for acc, ticker, at, text_path in fil:
        day = at.astimezone(UTC).date()
        q = [f for f in by_ticker.get(ticker, []) if day - max_days <= f[3] < day]
        if not q:
            continue
        end = max(f[3] for f in q)
        gold = {f[1]: f[2] for f in q if f[3] == end}
        if len(gold) < 2:
            continue
        cands.append(
            {
                "accession": acc,
                "ticker": ticker,
                "accepted_at": at.astimezone(UTC).isoformat(),
                "text_path": text_path,
                "period_end": end.isoformat(),
                "fiscal_period": next(f[4] for f in q if f[3] == end),
                "gold": gold,
            }
        )
    rng = random.Random(int(c["sample_seed"]))
    rng.shuffle(cands)
    picked, seen = [], set()
    for rnd in (0, 1):  # one per ticker first, then fill
        for cand in cands:
            if len(picked) >= int(c["gold_n_filings"]):
                break
            if (rnd == 0 and cand["ticker"] in seen) or cand in picked:
                continue
            picked.append(cand)
            seen.add(cand["ticker"])
    for p in picked:
        p["text"] = (ROOT / p["text_path"]).read_text(encoding="utf-8", errors="replace")
    return picked


def _decimals(x: float) -> int:
    s = repr(float(x))
    return 0 if s.endswith(".0") else len(s.split(".")[1])


def score(filings: list[dict[str, Any]], outputs: list[FilingFacts]) -> dict[str, Any]:
    tol = float(config()["value_rel_tol"])
    scales = {"one": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9}
    pairs = within = exact = period_ok = n_extracted = n_grounded = found = 0
    misses: list[dict[str, Any]] = []
    for f, out in zip(filings, outputs, strict=True):
        n_extracted += len(out.facts)
        n_grounded += sum(is_grounded(f["text"], x) for x in out.facts)
        pred = {}
        for x in out.facts:
            pred.setdefault(x.metric, x)  # first figure per metric
        q = f["fiscal_period"].split("-")[0]
        for metric, gv in f["gold"].items():
            pairs += 1
            x = pred.get(metric)
            if x is None:
                misses.append({"accession": f["accession"], "metric": metric, "gold": gv})
                continue
            found += 1
            ok = abs(x.value - gv) <= tol * abs(gv)
            within += ok
            exact += round(gv / scales[x.scale], _decimals(x.number)) == x.number
            period_ok += x.period.upper().startswith(q)
            if not ok:
                misses.append(
                    {
                        "accession": f["accession"],
                        "metric": metric,
                        "gold": gv,
                        "pred": x.value,
                        "quote": x.quote[:160],
                    }  # fmt: skip
                )
    return {
        "n_filings": len(filings),
        "gold_pairs": pairs,
        "within_1pct": round(within / pairs, 4) if pairs else 0.0,
        "exact_match": round(exact / pairs, 4) if pairs else 0.0,
        "coverage": round(found / pairs, 4) if pairs else 0.0,
        "grounding_rate": round(n_grounded / n_extracted, 4) if n_extracted else 0.0,
        "n_extracted": n_extracted,
        "period_quarter_match_of_found": round(period_ok / found, 4) if found else 0.0,
        "errors": misses[:20],
    }


def evaluate(db_path: Path | None = None) -> dict[str, Any]:
    c = config()
    filings = gold_filings(db_path)
    regex_out = [extract_regex(f["text"]) for f in filings]
    base = score(filings, regex_out)
    k = int(c["llm_n_filings"])
    sub = filings[:k]
    res = extract_llm([f["text"] for f in sub])
    statuses = [r.status for r in res.records]
    llm_block: dict[str, Any] = {
        "model": res.model,
        "used_fallback": res.used_fallback,
        "calls": {s: statuses.count(s) for s in sorted(set(statuses))},
        "cache_hits": res.cache_hits,
        **score(sub, res.outputs),
    }
    head = {"regex": score(sub, regex_out[:k]), "llm": llm_block}
    value = base["within_1pct"]
    notes = ""
    if value < TARGET or base["grounding_rate"] < 1.0:
        notes = (
            f"regex within-1% {value} (exact {base['exact_match']}, grounded "
            f"{base['grounding_rate']}) vs target {TARGET}; LLM on {k}: within-1% "
            f"{llm_block['within_1pct']}, grounded {llm_block['grounding_rate']}"
            f"{' (LLM not run: regex fallback)' if res.model is None else ''}. "
            "Explanation in docs/evals.md (facts)."
        )
    return {
        "component": "facts",
        "metric": "within_1pct",
        "value": value,
        "target": TARGET,
        "why_this_metric": "a number is right or useless; +/-1 % covers the release's rounding; "
        "grounding makes every wrong value inspectable",
        "gold_set": "derived: XBRL companyfacts (3-month Q1-Q3 values) for 30 dev 8-K 2.02 filings",
        "n": base["gold_pairs"],
        "split": "warmup+train+validation (filings accepted by the last validation cutoff)",
        "model": f"regex (LLM {res.model or 'fallback'} on {k} in details)",
        "git_sha": git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            "regex_all": base,
            "head_to_head_first_k": head,
            "filings": [
                {kk: f[kk] for kk in ("accession", "ticker", "fiscal_period", "period_end")}
                for f in filings
            ],
            "gold_metric_counts": {
                m: int(np.sum([m in f["gold"] for f in filings])) for m in METRICS
            },
        },
    }


if __name__ == "__main__":
    import json

    from dotenv import load_dotenv

    load_dotenv()
    out = evaluate()
    path = ROOT / "evals" / "results" / "facts.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=2))
