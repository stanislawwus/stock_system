"""`make eval-api`: the API layer's own eval (docs/03 row "API"); no live LLM call (D-48).

Metric: contract-test pass rate (target 1.00). `tests/api` is run in a subprocess; xfail
(design-phase) tests are reported separately, never counted as passes.

Also reported, in `details`:
- LLM response validity against the pydantic schema, from the committed bake-off trace
  `evals/results/llm_bakeoff.calls.jsonl` (one record per HTTP attempt; a batch is one
  (model, prompt) pair): share valid on the first attempt and after retries. The transport
  categories `empty_content` / `http_error` count as invalid (no schema-valid response arrived).
- latency p50 / p95 over `N_REPS` repeated calls: LLM cache replay (the cached path), the
  endpoints through FastAPI's TestClient on a database rebuilt from the demo slice (`first` = the
  first call of the process, i.e. cold: opens the database, loads FinBERT; `warm` = the rest).
  Cold LLM calls are `n/a`: nothing here calls a provider. Their recorded latencies from the
  bake-off are shown for context only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.metrics import git_sha
from src import demo, llm
from src.collectors.common import ROOT

OUT = ROOT / "evals" / "results" / "api.json"
TRACE = ROOT / "evals" / "results" / "llm_bakeoff.calls.jsonl"
N_REPS = 20
TARGET = 1.0
CALLS: list[tuple[str, str, str, dict[str, Any] | None]] = [
    ("health", "GET", "/api/v1/health", None),
    ("signals", "GET", "/api/v1/signals?ticker=AAPL&as_of=2026-08-14", None),
    ("brief", "GET", "/api/v1/brief?day=2026-08-31", None),
    ("chat_why_move", "POST", "/api/v1/chat", {"question": "Why did NVDA move on 2026-08-27?"}),
    (
        "analyze_finbert",
        "POST",
        "/api/v1/analyze",
        {"text": "Boeing shares plunge", "ticker": "BA"},
    ),
]


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))], 2)


def contract_tests() -> dict[str, Any]:
    env = {**os.environ, "LLM_MODE": "mock", "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/api",
            "-o", "addopts=", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    n = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|xfailed|error|errors)", tail)}
    passed, failed = n.get("passed", 0), n.get("failed", 0) + n.get("error", 0) + n.get("errors", 0)
    return {"passed": passed, "failed": failed, "xfailed": n.get("xfailed", 0), "summary": tail}


def llm_validity(path: Path = TRACE) -> dict[str, Any]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    batches: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        batches[(r["model"], r["prompt_sha256"])].append(r)
    first = after = 0
    for recs in batches.values():
        recs.sort(key=lambda r: r["attempt"])
        first += recs[0]["status"] == "ok"
        after += any(r["status"] == "ok" for r in recs)
    n = len(batches)
    lat = [r["latency_ms"] / 1000 for r in rows if not r["cache_hit"]]
    return {
        "n_attempts": len(rows),
        "n_batches": n,
        "valid_first_attempt": round(first / n, 4) if n else None,
        "valid_after_retries": round(after / n, 4) if n else None,
        "attempt_statuses": dict(sorted(Counter(r["status"] for r in rows).items())),
        "recorded_cold_latency_s": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "n": len(lat)},
    }  # fmt: skip


def cache_replay_latency() -> dict[str, Any]:
    cache = llm.DiskCache(demo.DEMO_DIR / "llm_cache")
    keys = [p.stem for p in (demo.DEMO_DIR / "llm_cache").rglob("*.json")]
    ms = []
    for _ in range(N_REPS):
        for k in keys:
            t0 = time.perf_counter()
            assert cache.get(k) is not None
            ms.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95), "n": len(ms), "entries": len(keys)}


def endpoint_latency() -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from src.api.main import app

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DUCKDB_PATH"] = str(Path(tmp) / "demo.duckdb")
        demo.load_slice(Path(os.environ["DUCKDB_PATH"]), copy_artifacts=False)
        client = TestClient(app)
        try:
            for name, method, url, body in CALLS:
                ms = []
                for _ in range(N_REPS + 1):
                    t0 = time.perf_counter()
                    r = client.request(method, url, json=body)
                    reply = r.json()
                    ms.append((time.perf_counter() - t0) * 1000)
                    assert r.status_code == 200, (name, r.status_code, r.text[:200])
                out[name] = {
                    "first_ms": round(ms[0], 1),
                    "warm_p50_ms": pct(ms[1:], 0.5),
                    "warm_p95_ms": pct(ms[1:], 0.95),
                    "n_warm": N_REPS,
                    "method": reply.get("method") if isinstance(reply, dict) else None,
                }  # fmt: skip
        finally:
            os.environ.pop("DUCKDB_PATH", None)
    return out


def evaluate() -> dict[str, Any]:
    ct = contract_tests()
    total = ct["passed"] + ct["failed"]
    rate = round(ct["passed"] / total, 4) if total else 0.0
    validity = llm_validity()
    replay = cache_replay_latency()
    endpoints = endpoint_latency()
    notes = (
        f"contract tests {ct['passed']}/{total} pass ({ct['xfailed']} design-phase xfail not "
        f"counted); LLM responses schema-valid {validity['valid_first_attempt']} on the first "
        f"attempt, {validity['valid_after_retries']} after retries (n = {validity['n_batches']} "
        f"batches, free models, bake-off trace); cache replay p50 {replay['p50_ms']} ms, p95 "
        f"{replay['p95_ms']} ms; cold LLM latency n/a (no live call; recorded bake-off p50 "
        f"{validity['recorded_cold_latency_s']['p50']} s). `method` shows which endpoint answered "
        "with FinBERT or the template composer; no endpoint waits for an LLM."
    )
    return {
        "component": "api",
        "metric": "contract_test_pass_rate",
        "value": rate,
        "target": TARGET,
        "why_this_metric": "the grader's experience is the API: every documented endpoint must "
        "answer with its documented status and schema, offline",
        "gold_set": "tests/api/ (contract tests); evals/results/llm_bakeoff.calls.jsonl; data/demo",
        "n": total,
        "split": "n/a",
        "model": "template (rules) + FinBERT; no LLM call",
        "git_sha": git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            "contract": ct,
            "llm_schema_validity": validity,
            "latency": {
                "llm_cache_replay": replay,
                "llm_cold": "n/a",
                "endpoints_testclient_demo_db": endpoints,
                "n_reps": N_REPS,
            },
        },
    }  # fmt: skip


def main() -> None:
    rec = evaluate()
    OUT.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(rec["notes"])
    for k, v in rec["details"]["latency"]["endpoints_testclient_demo_db"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
