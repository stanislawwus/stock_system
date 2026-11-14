# Gate 2 review (independent, 2026-11-14)

Scope: time-leak audit, simulator, evals, top risks, and the four questions asked for this
review (the D-40 validation halves, training code in `src/sim`, the test-period artefacts, and
`make train` on a fresh checkout). Repo state reviewed: `daceab7` (clean tree). Baseline before
any change: `make test` green (2 design-phase xfails), `make lint` clean. Product code was not
changed; the only code change is a Makefile guard plus its test (D-41). Everything that needs
more than a quick fix is in `docs/known_limitations.md` (`L-nn`).

## 1. Time-leak audit

Method: the `time-leak-check` skill steps on the code at `daceab7`, plus the artefact timestamps
under `data/models/` and `evals/results/`.

| Step | Result | Evidence |
|---|---|---|
| 1 Static read paths | pass | `connect()` / `read_parquet` only in `src/db.py`, `src/marketview`, `src/sim/execution.py`, `src/sim/impact_train.py` (feature cache), `src/nlp/features*.py`, `src/eda`, collectors. Strategies, impact and API read through `MarketView`; `tests/sim/test_signal_stack.py::test_impact_and_strategies_never_import_execution_or_labels` pins it. |
| 2 Execution prices | pass | `open_price` / `close_price` exist only on `ExecutionPrices`; imported by `src/sim/{engine,labels,impact_train,select}.py` only. |
| 3 Clock | pass | no `datetime.now` / `date.today` in decision paths (`datetime.now(UTC)` only stamps eval records). |
| 4 Schema | pass (by test) | collectors eval and `tests/collectors/test_quality.py` check `available_at` not null and not before the event time. |
| 5 Mapping | pass | price bar = 16:00 ET bar close (`prices.py:141`); news = Finnhub epoch `datetime`, parsed as UTC (`news.py:44`); filing = `acceptanceDateTime`, verified UTC (D-29); XBRL fact = acceptance of the carrying 10-Q/10-K, fallback filed date 22:00 ET (`xbrl.py:38`, D-30). Sentiment rows inherit the news `available_at` (`features.py:73`). Note: the fact `available_at` applies to the dev raw cache bucketing too, so a fact accepted in the test period never enters the dev cache. |
| 6 Probe | pass | `tests/test_marketview_leak.py`, `tests/test_test_period_lock.py`. |
| 7 Shift test | **partial** | covers momentum and mean reversion only; `finbert_v0` and the three stacked strategies are not shifted (L-03). |
| 8 Labels | pass | `drift_ar` is built in `src/sim/labels.py` from `ExecutionPrices` and consumed by `impact_train` as `y` only; `impact_labels` is never joined by a strategy or feature. The label of the last session of each split is dropped, so no label crosses a split boundary; inside validation the last val_a label uses the first val_b close (L-02). |
| 9 Splits | pass with notes | strategy parameters: `event` grid on train; `impact` / `aggregate_equal` grid on val_a (R3 says the confidence threshold belongs to validation); impact model fit on train; model choice and Platt on val_a; val_b report only. `TEST_FINAL.lock` absent, `final_run()` opened nowhere yet. Notes: L-02, L-05, L-10. |
| 10 LLM memory | n/a | no LLM strategy; masked-prompt experiment is a Should. |

Specific traces:

- **Features** (`src/impact/features.py`): every value is from `MarketView(cutoff(D))`: FinBERT
  scores of news with `available_at` in (previous weekday cutoff, D 09:00], 8-K items in the same
  window, XBRL growth from facts with `available_at <= as_of`, prior-day abnormal return from bars
  with `available_at <= as_of` (newest = close of D-1), volatility likewise. Clean. Holiday skew:
  L-04.
- **Dedup**: `relevance.dedup` sorts by `(available_at, id)`, keeps the earliest copy and compares
  an item only with earlier kept items inside `window_hours`, so the verdict for an item depends on
  the past only; the daily window loads 24 h extra so yesterday's original suppresses today's copy.
  No future duplicates.
- **News time zones**: Finnhub `datetime` is a Unix epoch, converted with `fromtimestamp(..., UTC)`;
  the stored value is tz-aware UTC; `MarketView` sets the session time zone to UTC before comparing.
  Correct. What cannot be checked offline is whether the provider's stamp is the publication time
  (L-08).
- **XBRL**: acceptance time by accession; comparative prior-year values skipped; test-period
  acceptances excluded from the dev cache by period bucketing. Clean.
- **Calibration**: `fit_impact` fits Platt on `val_a` predictions of the model chosen by val_a AUC
  and reports val_b; isotonic is reported, not used. `calibrator: platt` and `val_fit_fraction` sit
  in `config/strategies.yaml`, last modified 21:02 local, before `make train` wrote
  `impact_report.json` at 21:11. Clean.

## 2. Simulator

- **Lookahead**: decisions use `MarketView(09:00 ET of D)`; fills at open(D) from `ExecutionPrices`,
  loaded once for the split and never passed to a strategy (`collect_signals` runs inside
  `guarded()`). Score exits at the open use the 09:00 signal; stop-loss is observed at a close and
  filled at the next open; time exits and the end-of-split sale use the close of that day. The
  drawdown halt uses the previous close's equity. No lookahead found. The entry filter "has a bar
  on D" uses knowledge available at the open.
- **Survivorship**: present by construction (current S&P 100 names, watchlist chosen in hindsight);
  not fixable offline (L-07).
- **Costs**: buy spends exactly the budget (`qty = budget / (px (1 + c))`, fee `qty px c`); sell
  fee `gross c`; 5 bps per side (D-35), slippage 0, no dividends, fractional shares, unlimited
  liquidity at the open. Internally consistent, optimistic in level (L-09). Costs paid and turnover
  are in every record, which is what makes the finbert_v0 and aggregate_equal losses legible.
- **Risk manager**: cap, stop, halt, cooldown from `config/risk.yaml`; receives marks from the
  engine only. Eval `risk` 4/4 planted violations.

## 3. Evals

| Component | Metric fits? | Gold independent of the model? | CI? |
|---|---|---|---|
| collectors | yes (session completeness + zero-tolerance checks) | derived from SPY calendar; planted errors by the author (L-13) | no (n/a) |
| simulator, risk | yes (exact hand fixtures) | author-written fixtures (L-13) | n/a |
| relevance (dedup F1) | yes | planted edits by the author: measures robustness to imagined edits (L-13) | **no** |
| sentiment (macro-F1) | yes | human gold, FinBERT not tuned on it; second public set | yes |
| events (macro-F1) | yes, but 7 of 10 classes have 1-4 rows | human gold; sampling strata share the rule's window (documented) | yes |
| facts (within 1 %) | yes | XBRL, independent | **no** |
| impact (AUC) | yes | price labels, independent | **no** |
| calibration (Brier skill) | yes | same | **no** |
| aggregator (Sharpe val_b) | yes, but no random control / deflated Sharpe (L-10) | buy-and-hold target | yes |
| backtests (Sharpe) | yes | buy-and-hold target; `backtest_impact_train` is in-sample yet shown as a pass (L-11) | yes |

No circular gold set was found (no eval grades a model with labels that model produced). The
`events` TF-IDF model is trained on weak labels from the rule method, and both are graded against
the human set with gold ids excluded from training, which is correct.

## 4. The four questions

**D-40 validation halves.** Verified in code: `split_validation` cuts the labelled panel in time
order; `selection.period` routes `event` to train and `impact` / `aggregate_equal` to val_a;
`choose` never sees val_b; the calibrator is fitted on val_a. Evidence that no choice was made on
val_b: `config/strategies.yaml` and `config/risk.yaml` were last modified at 21:02, before
`make train` (21:11), `src.sim.select` (21:13) and the evals (21:15-21:22); `git diff 90efd68 HEAD
-- config src` is empty; `selected.json` holds the whole grid and its chosen rows are the val_a /
train maxima. Caveats: one val_b close enters the last val_a label (L-02); val_b figures of the
rejected model and calibrator were visible from 21:11 on, so the claim holds for this session
and needs a rule to keep holding (L-05); the `impact` choice rests on exactly 10 round trips
(L-10).

**Training code in `src/sim`.** `src/sim/labels.py`, `src/sim/impact_train.py` and
`src/sim/select.py` are the only modules that import `ExecutionPrices` besides the engine; the
impact package exposes pure functions and `MarketView`-only features; the stacked strategies
import no sim module. Pinned by `test_impact_and_strategies_never_import_execution_or_labels`.
Correct placement under R2.

**`data/test_scores.parquet` and the lock.** The parquet is written only by
`src/nlp/features_test.py` and read only by `MarketView.sentiment()` through
`splits.final_scores_for`, which returns `None` for any view before the first test cutoff and
otherwise requires `final_run()`. The manifest carries counts only (25,750 rows, 22,417 articles,
FinBERT 0bc2ac1, window from 2026-08-25 13:00 UTC); no eval, test or script reads the real file.
`TEST_FINAL.lock` is absent; `final_run()` refuses when it exists and is opened nowhere yet
(`test_final_run_is_opened_only_where_the_lock_is_defined_for_now`); the Makefile refuses too.
Open: the lock is written after the runs, not before (L-01).

**`make train` on a fresh checkout.** `data/models/` and `data/features/` are git-ignored, so on
a fresh clone `make backtest` (`STRATEGY=all` or a stacked name) crashed with
`FileNotFoundError: ... run make train first`, and `make eval` printed `skipped` for the
aggregator and the six stacked backtest records because `run_all` catches `FileNotFoundError`
as "gold set missing". Fixed in this review: a `need-train` guard target that `eval`, `arena`
and the stacked `backtest` targets depend on, with a `make train` hint (D-41, test in
`tests/test_repo_rules.py`); README lists `make train` in the walking-skeleton steps. The
Python-side conflation remains (L-12).

## 5. Top 10 risks

| # | Risk | Fix | Status |
|---|---|---|---|
| 1 | `make test-final` can read the test period once without writing the lock (lock after the runs; arena module missing) | write the lock before the `--final` runs | L-01 |
| 2 | Fresh checkout: evals silently skip stacked components, backtests crash | Makefile guard + README | **applied** (D-41) |
| 3 | Shift test misses finbert_v0 and the stacked strategies | extend the fixture and parametrisation | L-03 |
| 4 | One val_b close inside val_a labels; cut computed twice | single cut function, drop the last val_a session, pin by test | L-02 |
| 5 | 20-point grid, max Sharpe, no deflation; `impact` at the 10-trade minimum | deflated Sharpe, random control | L-10 |
| 6 | val_b of rejected alternatives visible | freeze val_b by rule; log every `make train` | L-05 |
| 7 | Survivorship in universe and watchlist | document; historical constituents | L-07 |
| 8 | News not point-in-time | `collected_at`, live vs backfill comparison | L-08 |
| 9 | No CI for impact, calibration, facts, relevance; in-sample row shown as pass | bootstrap over rows; `in_sample` flag | L-11 |
| 10 | Holiday skew between live window and item table | session-aware `previous_cutoff` | L-04 |

## 6. Gate 2 verdict

Gate 2 asks that every Must component up to the aggregator has an eval. Status:

- with an eval file and a result: collectors, MarketView/splits (tests), relevance, sentiment,
  events, facts, impact, calibration, strategies (buy-and-hold, momentum, mean reversion,
  finbert_v0, event, impact), simulator, risk manager, equal-weight aggregator;
- **without**: the aggregator *as specified* (Hedge with online weights and a confidence threshold
  chosen on validation, D-17 and docs/01) is not built; `aggregator.confidence_threshold` is
  `null`; the eval plan's random-strategy control, deflated Sharpe and ablation are missing;
  `make arena` fails. The Must item "intake: phrase -> tickers" has no service and no eval.

Verdict: **Gate 2 is not passed for the aggregator as specified; no Should work.** The next
sessions stay on Must items: Hedge + threshold + arena (L-14), the `--final` entry points with an
atomic lock (L-01), then API/chat/UI. Every number in `docs/evals.md` currently rests on dev
splits; nothing in this review touched the test period.
