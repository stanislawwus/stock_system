"""Eval: free-LLM bake-off through src/llm.py (R5), to pick the primary model and fallback order.

- **Metric**: JSON-schema adherence = share of HTTP calls whose output validated against the
  pydantic schema on that attempt (per model); latency per call, sentiment macro-F1 / events
  exact match on 20 headlines and facts within-1 % on 10 releases reported alongside.
- **Gold set**: 20 headlines of `evals/labels/news_100.csv` (fixed seed) and the first 10
  filings of the facts gold (`evals.eval_facts.gold_filings`, XBRL values).
- **Target**: 1.0 adherence for the chosen model (every call parseable; retries cost budget).
- **Why**: on free tiers a model that breaks the schema burns retries and lands on the fallback,
  so adherence and latency decide usefulness before quality does; n = 30 items is too small to
  rank quality, so quality is a tie-breaker only.

Budget: 3 calls per model (2 x 10 headlines, 1 x 10 releases), at most `max_calls` HTTP calls
in total, enforced by the gateway's daily counter (`LLM_DAILY_BUDGET`). Responses are cached on
disk, so a rerun costs nothing; call latencies of the live run are kept in
`evals/results/llm_bakeoff.calls.jsonl` because cache hits have none.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from evals.eval_events import load_gold as load_events_gold
from evals.eval_facts import gold_filings, score
from evals.metrics import classification_report, git_sha
from src import llm
from src.collectors.common import ROOT
from src.nlp.facts import PROMPT as FACTS_PROMPT
from src.nlp.facts import FilingFacts, extract_regex, snippet

MODELS = ["cohere/north-mini-code:free", "nvidia/nemotron-3-ultra-550b-a55b:free"]
N_HEADLINES, N_RELEASES, MAX_CALLS, SEED = 20, 10, 15, 20261114
CALLS_LOG = ROOT / "evals" / "results" / "llm_bakeoff.calls.jsonl"
NewsEvent = Literal[
    "1.01", "2.01", "2.02", "2.05", "3.02", "5.02", "5.07", "7.01", "8.01", "other_item", "none"
]


class HeadlineLabel(BaseModel):
    """LLM structured output for one headline (sentiment + events, D-10 without 9.01)."""

    sentiment: Literal["pos", "neu", "neg"]
    events: list[NewsEvent] = Field(min_length=1, max_length=4)


HEADLINE_PROMPT = (
    "Each item is a news headline and the stock ticker it was tagged with. Label it from the view "
    "of a holder of THAT ticker, using the headline only. sentiment: pos / neg / neu (neu when "
    "there is no clear direction for that holder, when it is about another company, or mixed). "
    "events: the 8-K item types the headline reports for that company: 1.01 material agreement, "
    "2.01 acquisition/disposal completed, 2.02 quarterly results, 2.05 restructuring/layoffs, "
    "3.02 unregistered equity sale, 5.02 executive/director change, 5.07 shareholder vote, "
    "7.01 Reg FD / investor presentation, 8.01 other material news, other_item another 8-K item; "
    "'none' (alone) when no corporate event is reported (opinion, price move, list, analyst note)."
)


def _headline_fallback(batch: list[dict]) -> list[HeadlineLabel]:
    return [HeadlineLabel(sentiment="neu", events=["none"]) for _ in batch]


def run(models: list[str] = MODELS) -> dict[str, Any]:
    rows = load_events_gold()
    rng = random.Random(SEED)
    heads = rng.sample(rows, N_HEADLINES)
    rels = gold_filings()[:N_RELEASES]
    h_items = [{"ticker": r["ticker"], "headline": r["headline"]} for r in heads]
    r_items = [{"release": snippet(f["text"])} for f in rels]
    out: dict[str, Any] = {}
    live_records: list[dict] = []
    for model in models:
        llm.reset_run_state()
        h = llm.call_json(
            "events", HEADLINE_PROMPT, h_items, HeadlineLabel,
            fallback=_headline_fallback, models=[model],
        )  # fmt: skip
        r = llm.call_json(
            "facts", FACTS_PROMPT, r_items, FilingFacts,
            fallback=lambda b: [extract_regex(x["release"]) for x in b], models=[model],
            params=llm.LLMParams(max_tokens=2048),
        )  # fmt: skip
        recs = [x for x in (*h.records, *r.records) if x.status != "fallback"]
        for x in recs:
            if not x.cache_hit:
                live_records.append({**json.loads(x.model_dump_json()), "model": model})
        http = [x for x in recs if not x.cache_hit and x.status != "budget_exceeded"]
        ok = [x for x in http if x.status == "ok"]
        sent = classification_report(
            [r_["sentiment"] for r_ in heads], [o.sentiment for o in h.outputs],
            ["pos", "neu", "neg"],
        )  # fmt: skip
        ev_exact = sum(
            set(o.events) == r_["gold"] for o, r_ in zip(h.outputs, heads, strict=True)
        ) / len(heads)
        out[model] = {
            "http_calls": len(http),
            "schema_valid_calls": len(ok),
            "schema_adherence": round(len(ok) / len(http), 4) if http else None,
            "status_counts": dict(Counter(x.status for x in recs)),
            "latency_ms": sorted(x.latency_ms for x in http),
            "headlines_used_fallback": h.used_fallback,
            "releases_used_fallback": r.used_fallback,
            "sentiment_macro_f1_20": sent["macro_f1"],
            "events_exact_set_match_20": round(ev_exact, 4),
            "facts_10": {k: v for k, v in score(rels, r.outputs).items() if k != "errors"},
        }
    return {"per_model": out, "live_records": live_records}


def evaluate() -> dict[str, Any]:
    res = run()
    per = res["per_model"]
    if res["live_records"]:
        with CALLS_LOG.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in res["live_records"]:
                fh.write(json.dumps(rec, default=str) + "\n")
    logged = []
    if CALLS_LOG.exists():
        logged = [json.loads(x) for x in CALLS_LOG.read_text(encoding="utf-8").splitlines() if x]
    for model, block in per.items():
        mine = [x for x in logged if x["model"] == model and x["status"] != "budget_exceeded"]
        if mine:  # adherence/latency from the live run (reruns are cache hits)
            ok = [x for x in mine if x["status"] == "ok"]
            block["http_calls"] = len(mine)
            block["schema_valid_calls"] = len(ok)
            block["schema_adherence"] = round(len(ok) / len(mine), 4)
            block["latency_ms"] = sorted(x["latency_ms"] for x in mine)
            block["median_latency_ms_ok"] = (
                sorted(x["latency_ms"] for x in ok)[len(ok) // 2] if ok else None
            )
    total_calls = len(logged)
    ranked = sorted(
        per,
        key=lambda m: (
            -(per[m]["schema_adherence"] or 0),
            per[m].get("median_latency_ms_ok") or 10**9,
        ),
    )
    best = ranked[0]
    value = per[best]["schema_adherence"] or 0.0
    return {
        "component": "llm_bakeoff",
        "metric": "schema_adherence",
        "value": value,
        "target": 1.0,
        "why_this_metric": "on free tiers a model that breaks the JSON schema burns retries and "
        "falls back; adherence and latency decide usefulness before quality on n=30",
        "gold_set": "20 headlines of evals/labels/news_100.csv + 10 releases of the facts gold "
        "(XBRL)",
        "n": N_HEADLINES + N_RELEASES,
        "split": "train+validation (gold), dev filings",
        "model": best,
        "git_sha": git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": "" if value >= 1.0 else f"best model adherence {value}; see docs/evals.md",
        "details": {
            "ranking": ranked,
            "per_model": per,
            "http_calls_total_live_run": total_calls,
            "max_calls": MAX_CALLS,
            "budget_env": os.environ.get("LLM_DAILY_BUDGET"),
        },
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    out = evaluate()
    path = ROOT / "evals" / "results" / "llm_bakeoff.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2)[:4000])
