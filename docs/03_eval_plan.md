# 03 — Evaluation plan

One row per component (AGENTS.md R4): metric, gold set, target, why this metric. Results land in
`evals/results/<component>.json` (format in AGENTS.md section 5) and the table in `docs/evals.md` is
generated from them by `make eval`. Evals run on `train` / `validation` only; the test period is used
once by `make test-final`.

## Labels used below

Defined in `docs/02_architecture.md`, repeated here because every impact and arena number depends on
them. For an event with `available_at = a`: D = first trading day with `cutoff(D) >= a`
(`cutoff` = 09:00 ET), D+1 = next trading day, `r(x→y) = price_y/price_x − 1`, SPY as the market.

| Label | Definition | Realised when | Used for |
|---|---|---|---|
| `gap_ar` | `r(close(D−1)→open(D)) − r_SPY(close(D−1)→open(D))` | at our entry (open of D) | how much was priced in before entry; diagnostic, never a strategy input |
| `drift_ar` | `r(open(D)→close(D+1)) − r_SPY(open(D)→close(D+1))` | close of D+1 | the tradable target; impact model predicts `P(drift_ar > 0)` |

## Components

| Component | Metric | Gold set | Target | Why this metric |
|---|---|---|---|---|
| Intake (phrase → tickers, risk profile) | accuracy of name→ticker; exact-match accuracy of profile fields | `evals/labels/intake_dialogues.jsonl` (25 synthetic dialogues; known name→ticker pairs) | ≥ 0.95 / ≥ 0.90 | one wrong ticker poisons everything downstream; exact match is the only acceptable outcome |
| Collectors | completeness of trading days per ticker; duplicate share; `available_at >= event_time` violations | exchange calendar; dedup keys; schema check | ≥ 99 %; 0; 0 | gaps and duplicates silently bias indices and backtests |
| Time rules (MarketView) | three tests pass: canary, random-future, bypass raises; shift test (signals unchanged when future rows are appended) | `tests/test_marketview.py`; strategy fixture | 100 % | a single leak invalidates every arena number; this is a gate, not a score |
| Relevance + dedup | precision / recall on pairs (same story? on-topic?) | auto pairs from the same 8-K event + ~30 hand-labelled pairs | P ≥ 0.90, R ≥ 0.80 | precision keeps noise out of the sentiment index; recall shows what is lost |
| Sentiment | macro-F1; lexicon vs LLM; schema-valid rate | Financial PhraseBank sample, agree ≥ 75 % (n = 500) | ≥ 0.80 | classes are imbalanced; macro-F1 weights negative and neutral as much as positive |
| Events | macro-F1 per label (multi-label, micro-F1 reported too); news subset kappa vs human | 8-K item numbers from the EDGAR index (automatic, all dev filings); 60 hand-labelled news with LLM suggestion column | ≥ 0.75 macro-F1 | item classes are rare and unequal; accuracy would reward predicting "8.01" |
| Facts | share of values within ±1 % of XBRL; grounding validity (span text equals the quote and contains the number) | EDGAR companyfacts for the same fiscal period (automatic) | ≥ 0.90; 100 % grounded | a number is right or useless; tolerance covers rounding; the span makes errors inspectable |
| Impact | AUC and directional accuracy of `calibrated_p` vs `sign(drift_ar)`; Brier score; `gap_ar` reported as "priced-in share" | `impact_labels` on train/validation | AUC > 0.55; Brier < 0.25 | AUC is threshold-free with chance at 0.5; Brier checks the probability itself |
| Strategies (each) | total return, Sharpe, max drawdown, hit rate, trade count, cost share; rank vs 1000 random long-only strategies with the same trade count | buy-and-hold; random strategies; SPY | each explained vs baselines; ≥ 1 strategy beats 95 % of random | one number hides risk; the random control separates skill from luck |
| Aggregator + calibration | Sharpe on validation vs best single strategy (chosen on train), equal weights, buy-and-hold; deflated Sharpe counting every attempt; Brier and reliability diagram of `calibrated_p`; ablation (drop each strategy) | validation period; attempt log | Sharpe > buy-and-hold and > 95 % random; reliability slope within 0.8–1.2 | deflated Sharpe punishes trying many things; calibration makes confidence meaningful for the threshold |
| Risk manager | share of rule violations after veto (position > limit, trade during halt, stop not applied); planted scenarios caught | synthetic portfolios with planted violations (`evals/labels/planted_errors.jsonl`) | 0 violations; 100 % planted caught | the risk layer is a safety property; any violation is a bug |
| Simulator | fills reproduce hand-computed P&L on a fixture (commissions, slippage, open-of-D fills); equity-curve identity: final equity = cash + positions | 3 hand-worked fixtures | exact to the cent | correctness is binary here; the fixtures double as documentation |
| Chat agent | faithfulness (every number and claim traceable to a cited record); citation validity (ids exist and are visible at `as_of`); LLM-judge kappa vs ~30 human labels; injection test | `evals/labels/chat_questions.jsonl` (30 questions with expected values); DuckDB | ≥ 0.90 faithful; 100 % citations valid; kappa ≥ 0.6; 0 injections followed | a fluent wrong answer is the main failure mode; citations are checkable without an LLM |
| LLM layer | schema-valid rate before / after retries; calls per task per day; cache hit rate; fallback share | JSONL traces | ≥ 98 % valid after retries | separates model reliability from component quality; budget use shows the cost |
| API | contract tests pass; p50 / p95 latency of `/analyze`, `/signals`, `/chat`; 501/500 count | pytest + TestClient; traces | 100 %; p95 ≤ 10 s; 0 500s | the grader's experience is the API |
| End-to-end | smoke scenarios through the API from `/collect` to `/chat` | 10 scripted scenarios on the demo slice | 10 / 10 | the demo must not break |
| All LLM-using components | planted errors caught | `evals/labels/planted_errors.jsonl` (wrong number, foreign-company news, shifted date, injected instruction) | ≥ 90 % | proves the metric can tell good from bad |

## Procedure

1. `make eval` runs `evals/run_all.py`, which imports each `evals/eval_<component>.py`, calls the
   service functions directly (no HTTP), and writes one JSON per component.
2. Every LLM-scored metric is accompanied by a human-labelled subset and Cohen's kappa; kappa < 0.4
   marks the metric as unreliable in the table.
3. Every score below target gets 2–3 sentences in `docs/evals.md`: where the errors are, why, what
   changes next. A missed target with an explanation is a valid result.
4. The masked-prompt experiment (company names and dates replaced by placeholders) runs for sentiment,
   events and the chat agent on `validation`; if unmasked scores are clearly higher, the report says
   the model remembers rather than analyses.
5. `make test-final` runs the arena and the simulator on `test` once; its numbers go into the report
   as-is, with the lock file as proof.
