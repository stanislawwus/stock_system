"""Intake eval (src/nlp/intake.py, D-45; docs/03_eval_plan.md row "Intake").

- metric: accuracy of `resolve_name` on the official SEC titles of the 77 universe companies
  ("JPMORGAN CHASE & CO" -> JPM); target 0.95; 95 % bootstrap CI over companies.
- details: false-positive rate on non-universe SEC titles (hard ones contain an alias word, e.g.
  "Apple Hospitality REIT"; random ones do not), and exact-set accuracy of `extract_tickers` on
  seeded watchlist requests ("add JPM, NVIDIA CORP and Walmart Inc. to my watchlist"), target
  0.90.
- gold set: derived from `data/raw/edgar/company_tickers.json` (SEC, cached by the collectors);
  the resolver never reads SEC titles (only tickers and the relevance aliases), so the gold is
  independent of the table it grades. Its sha256 is in the record.
- why this metric: a wrong ticker poisons every downstream step, so exact accuracy is the only
  acceptable outcome; false positives on look-alike names are the failure that matters next.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np

from evals.metrics import bootstrap_ci, eval_config, git_sha
from src.collectors.common import ROOT, load_universe
from src.nlp.intake import extract_tickers, normalise, resolve_name, tables

SEC_TAG = re.compile(r"\s*/[A-Z]{2,3}/?\s*$")
TEMPLATES = (
    "add {a}, {b} and {c} to my watchlist",
    "please track {a} plus {b}, and {c}",
    "I want news on {a}; also {b} and {c}",
)


def build_gold() -> dict[str, Any]:
    cfg = eval_config()["intake"]
    path = ROOT / cfg["sec_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (SEC company_tickers.json, make collect-dev)")
    raw = path.read_bytes()
    rows = json.loads(raw)
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    universe = {c.ticker for c in load_universe()[0]}
    title = {}
    for r in rows:  # first title per ticker (SEC lists the primary listing first)
        title.setdefault(r["ticker"], r["title"])
    positives = [(title[t], t) for t in sorted(universe) if t in title]
    alias_words = {w for alias, _ in tables()[2] for w in alias if len(w) >= 4}
    ciks = {int(c.cik) for c in load_universe()[0]}  # same company, other symbol: not negative
    others = sorted({r["title"] for r in rows if int(r["cik_str"]) not in ciks})
    hard = [t for t in others if alias_words & set(normalise(t))]
    rng = random.Random(cfg["seed"])
    hard = rng.sample(hard, min(int(cfg["n_hard_negatives"]), len(hard)))
    rest = [t for t in others if t not in set(hard)]
    rand = rng.sample(rest, int(cfg["n_random_negatives"]))
    phrases = []
    for i in range(int(cfg["n_phrases"])):
        picks = rng.sample(positives, int(cfg["companies_per_phrase"]))
        # SEC state tags ("/MA/", "/NEW") are filing artefacts nobody types in a request
        forms = [t if rng.random() < float(cfg["ticker_share"]) else SEC_TAG.sub("", name).strip()
                 for name, t in picks]  # fmt: skip
        text = TEMPLATES[i % len(TEMPLATES)].format(a=forms[0], b=forms[1], c=forms[2])
        phrases.append((text, sorted(t for _, t in picks)))
    return {"positives": positives, "hard": hard, "random": rand, "phrases": phrases,
            "sha256": hashlib.sha256(raw).hexdigest(), "path": cfg["sec_file"]}  # fmt: skip


def _acc(hits: list[bool]) -> tuple[float, tuple[float, float] | None]:
    a = np.asarray(hits, dtype=float)
    if not a.size:
        return 0.0, None
    return float(a.mean()), bootstrap_ci(a.size, lambda idx: float(a[idx].mean()))


def evaluate() -> dict[str, Any]:
    cfg = eval_config()["intake"]
    g = build_gold()
    pos_hits = [resolve_name(n) == t for n, t in g["positives"]]
    misses = [(n, t, resolve_name(n)) for n, t in g["positives"] if resolve_name(n) != t]
    fp_hard = [(n, resolve_name(n)) for n in g["hard"] if resolve_name(n) is not None]
    fp_rand = [(n, resolve_name(n)) for n in g["random"] if resolve_name(n) is not None]
    ph_hits = [sorted(extract_tickers(p)) == exp for p, exp in g["phrases"]]
    ph_miss = [(p, exp, extract_tickers(p)) for p, exp in g["phrases"]
               if sorted(extract_tickers(p)) != exp]  # fmt: skip
    acc, ci = _acc(pos_hits)
    ph_acc, ph_ci = _acc(ph_hits)
    target = float(cfg["target_name_accuracy"])
    notes = (
        f"name->ticker {acc:.3f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}) on {len(pos_hits)} SEC titles; "
        f"{len(misses)} misses; false positives {len(fp_hard)}/{len(g['hard'])} on look-alike "
        f"titles, {len(fp_rand)}/{len(g['random'])} on random ones; phrase exact-set "
        f"{ph_acc:.3f} (target {cfg['target_phrase_exact']})."
    )
    if acc < target:
        notes += " Below target: SEC titles that do not start with an alias (see misses)."
    return {
        "component": "intake",
        "metric": "name_to_ticker_accuracy",
        "value": round(acc, 4),
        "target": target,
        "why_this_metric": "a wrong ticker poisons every downstream step; exact match only",
        "gold_set": f"derived: SEC titles in {g['path']} (sha256 {g['sha256'][:12]})",
        "n": len(pos_hits),
        "split": "n/a",
        "model": "table (tickers + relevance aliases)",
        "git_sha": git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            "name_accuracy_ci95": ci,
            "misses": misses,
            "false_positive_rate_hard": round(len(fp_hard) / max(len(g["hard"]), 1), 4),
            "false_positives_hard": fp_hard,
            "false_positive_rate_random": round(len(fp_rand) / max(len(g["random"]), 1), 4),
            "false_positives_random": fp_rand,
            "phrase_exact": round(ph_acc, 4),
            "phrase_exact_ci95": ph_ci,
            "phrase_target": cfg["target_phrase_exact"],
            "phrase_misses": ph_miss,
            "n_phrases": len(ph_hits),
            "sha256": g["sha256"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
