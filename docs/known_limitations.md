# Known limitations

Items found in reviews that need more than a quick fix. Each entry names the risk, the evidence
and the concrete fix. Source of most rows: `docs/gate2_review.md` (2026-11-14). Remove a row in the
commit that fixes it and cite the row in the commit message.

## Time and splits

**L-02 — One session of val_b is used by val_a.** `drift_ar` for the last val_a session needs the
close of the first val_b session (`src/sim/labels.py`), so the calibrator and the hold/threshold
grid see one val_b close. The cut is also computed twice: `impact_train.split_validation` from the
labelled panel (62 days, drops first and last session) and `val_a_range` / `val_b_range` from the
session list (64 days). Both land on session 32 today only because 62/2 and 64/2 round the same
way. *Fix*: one function returning the cut date from the session list, used by both; exclude the
last val_a session from the fit set (or shift the cut by one); a test with a fixture DB that the
panel cut equals the session cut.

**L-03 — The shift test covers only momentum and mean reversion.**
`tests/sim/test_no_lookahead.py::test_signals_for_d_ignore_bars_of_d_and_later` corrupts future
bars; `finbert_v0`, `event`, `impact` and `aggregate_equal` are protected only by the static
import scan and by `MarketView` filtering. *Fix*: a fixture DB with news, sentiment, filings and
facts; run the four strategies (with an in-memory item table and a constant model) for days
D1..Dn, then append future news/filings/facts/sentiment rows and bars and assert equality.

**L-04 — Holiday skew between the live window and the item table.** `src/nlp/daily.previous_cutoff`
steps back to the previous *weekday*, not the previous *session*: on the day after a market
holiday the window starts at the holiday's 09:00 cutoff, so news and 8-Ks accepted on the eve of
the holiday are not counted, while the training-time item table maps them to that session via
`decision_day` (sessions). *Fix*: give `previous_cutoff` the session calendar (the view can serve
SPY bar dates) and use it in `src/impact/features.py` too.

**L-05 — val_b figures of rejected alternatives are visible.** `data/models/impact_report.json`,
`evals/results/calibration.json` and `evals/results/aggregator.json` report val_b for both models,
both calibrators and for `event` / `impact` / `finbert_v0`. Nothing was changed after they were
produced (config mtimes precede the training run; no config/src commit after `90efd68`), but any
future change of model, calibrator or weights is now informed by val_b. *Fix*: treat val_b like
the test period from here on: changes to the stack are judged on val_a only; say so in a `D-nn`,
and log every re-run of `make train` in `agent_log.md`.

**L-06 — Feature cache is keyed by config, not by data.** `data/features/impact_<split>_<hash>.parquet`
is reused as long as the feature config is unchanged, so a re-collection or a new `make features`
run does not refresh the panel. *Fix*: add the DuckDB row counts (news, sentiment, filings, facts)
to the hash or pass `--refresh` from `make train`.

## Simulator and data

**L-07 — Survivorship / selection bias.** The 77-name universe is the S&P 100 as of November 2026
and the 10-name watchlist was chosen with the whole sample known (BA explicitly for its events).
Buy-and-hold targets, the 8-K item table and every backtest inherit this. *Fix*: document in the
report; for the item table use the constituents as of 2025-12 (not available offline; needs a
historical constituent list).

**L-08 — News is not point-in-time.** Finnhub history was fetched in November 2026 and
`available_at` is the provider's `datetime`. Items may have been added, edited or re-stamped
after publication; 43 % of news carries a pre-open time, which cannot be validated offline.
*Fix*: store `collected_at` per row; on a later live day compare the live feed with a backfill of
the same day and report the share of items that differ.

**L-09 — Cost and fill model is optimistic.** Fixed 5 bps per side, zero slippage, fills at the
official open and close for any size, fractional shares, no dividends (for the strategies and for
the buy-and-hold benchmark alike), no borrowing. *Fix*: a stress run with `slippage_pct` > 0 and
dividend-adjusted benchmark returns; report the cost share next to every Sharpe (already in
`details.costs_paid`).

## Evals

**L-10 — No multiplicity control on the selection grid.** `src/sim/select.py` tries 20 (hold,
threshold) points per strategy and keeps the best Sharpe; `impact` kept a point with exactly
`min_trades` = 10 round trips. *Partly fixed (D-42)*: `make arena` logs every grid point in
`evals/trials.csv`, reports the deflated Sharpe with N = all logged trials, matched random
controls and the drop-one ablation (`docs/05_arena.md`). *Still open*: the aggregator eval
(`evals/eval_impact.evaluate_aggregator`, val_b) has no random control of its own.

**L-11 — Point estimates without intervals.** `impact` (AUC), `calibration` (Brier skill),
`facts` (within 1 %) and `relevance` (dedup F1) report no CI; `backtest_impact_train` shows
Sharpe 7.19 as a pass in `docs/evals.md` although it is in-sample. *Fix*: bootstrap over rows
with `evals.metrics.bootstrap_ci` (AUC: resample (p, y) pairs; facts: resample gold pairs); add
`in_sample: true` to records whose split was used to fit anything and render it in the table.

**L-12 — `make eval` conflates a missing gold set with a missing model.** `evals/run_all.py`
catches `FileNotFoundError` (meant for gold sets absent on a clean clone) and prints `skipped`,
so without `make train` the aggregator and the stacked backtests silently disappear from the
table. The Makefile guard `need-train` (D-41) stops this at the entry point; the Python path
still conflates the two. *Fix*: raise a dedicated `ModelNotTrainedError(FileNotFoundError)` from
`src/strategies/signal_stack._need` and let `run_all` re-raise it.

**L-13 — Gold sets written by the component's author.** The relevance dedup gold (planted edits),
the simulator and risk fixtures and the collectors' planted errors were designed by the same
author as the code they grade, so they measure robustness to the failure modes the author
imagined. The `events` gold was sampled with the rule method's own ±24 h window (noted in
`docs/evals.md`). *Fix*: a second annotator for 30 dedup pairs; a held-out set of real syndicated
copies found by `clean.dedup_key` rather than planted.

## Scope
