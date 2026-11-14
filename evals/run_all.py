"""`make eval`: run every component eval, write evals/results/<component>.json, refresh the table
between the markers in docs/evals.md (skill run-evals). Components are added here as they land."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import partial
from typing import Any

from evals import (
    eval_collectors,
    eval_events,
    eval_facts,
    eval_relevance,
    eval_risk,
    eval_sentiment,
    eval_simulator,
)
from src.collectors.common import ROOT
from src.sim import run as backtest
from src.strategies.registry import BUILDERS

EVALS: dict[str, Callable[[], dict[str, Any]]] = {
    "collectors": eval_collectors.evaluate,
    "simulator": eval_simulator.evaluate,
    "risk": eval_risk.evaluate,
    "relevance": eval_relevance.evaluate,
    "sentiment": eval_sentiment.evaluate,
    "events": eval_events.evaluate,
    "facts": eval_facts.evaluate,
    **{
        f"backtest_{name}_{split}": partial(backtest.evaluate, name, split)
        for split in ("train", "validation")
        for name in sorted(BUILDERS)
    },
}
DOC = ROOT / "docs" / "evals.md"
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def table(results: list[dict[str, Any]]) -> str:
    rows = [
        "| Component | Metric | Value | Target | Pass | Gold set | n | Model | git |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ok = "yes" if r["value"] >= r["target"] else "**no**"
        rows.append(
            f"| {r['component']} | {r['metric']} | {r['value']} | {r['target']} | {ok} | "
            f"{r['gold_set']} | {r['n']} | {r['model']} | {r['git_sha']} |"
        )
    return "\n".join(rows)


def refresh_doc() -> None:
    results = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "evals" / "results").glob("*.json"))
    ]
    text = DOC.read_text(encoding="utf-8")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(f"{head}{BEGIN}\n{table(results)}\n{END}{tail}", encoding="utf-8", newline="\n")


def main() -> int:
    # `make eval` never spends LLM budget: cached responses are served (cache hits cost nothing),
    # misses fall back deterministically. Live LLM evals run on purpose, e.g.
    # `EVAL_LLM_LIVE=1 LLM_DAILY_BUDGET=<used + 15> uv run python -m evals.eval_llm_bakeoff` (D-39).
    if not os.environ.get("EVAL_LLM_LIVE"):
        os.environ["LLM_DAILY_BUDGET"] = "0"
    for name, fn in EVALS.items():
        try:
            result = fn()
        except FileNotFoundError as exc:  # a gold set not on disk (e.g. a clean clone)
            print(f"{name}: skipped ({exc})")
            continue
        path = ROOT / "evals" / "results" / f"{name}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"{name}: {result['metric']} = {result['value']} (target {result['target']})")
    refresh_doc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
