---
name: time-leak-check
description: Audit code and data for look-ahead bias (future information reaching a decision at as_of). Use before merging any collector, strategy, feature, impact or simulator change, and before make arena / make backtest / make test-final.
---

# time-leak-check

The time rules are AGENTS.md R2 and R3 and `docs/decisions.md` D-03, D-04, D-16. A leak makes every
backtest number meaningless, so this check runs before any number is reported.

## When to use

- a collector, feature, strategy, impact, aggregator, or simulator module was added or changed
- before `make arena`, `make backtest`, and always before `make test-final`
- a backtest result looks too good (Sharpe > 2 on daily stock data is a leak until proven otherwise)

## Steps

1. **Static scan of read paths.** Grep `src/` for DuckDB connections, `read_parquet`, `read_csv`,
   `pd.read_*` outside `src/marketview/` and `src/collectors/`. Anything found in `strategies`, `nlp`,
   `impact`, `api` is a violation: route it through `MarketView`.
2. **Execution-price scan.** Grep `src/strategies`, `src/nlp`, `src/impact` for the open-price
   accessor (`open_price`, `execution`, `fill`) and for `close` of the same day D. Only `src/sim` may
   read open(D).
3. **Clock scan.** Grep for `datetime.now(`, `date.today(`, `time.time(` in decision paths. Decision
   code receives `as_of` as an argument; it never reads the wall clock. (`ruff` DTZ catches naive ones.)
4. **Schema check.** Open DuckDB read-only and verify every table written by collectors and features
   has `available_at TIMESTAMPTZ NOT NULL`; list rows where `available_at` is null or earlier than the
   event time (publication, acceptance, candle close). Zero rows expected.
5. **Mapping check.** Compare each collector's `available_at` rule to D-04 (prices: close + 30 min;
   news: Finnhub publication time; SEC: acceptance time, next 06:00 ET if accepted after 17:30 ET;
   XBRL: acceptance of the filing that carried the fact; crypto: candle close). Derived features:
   `available_at = max(available_at of inputs)` (D-16).
6. **Probe test.** Run `tests/test_marketview_leak.py` (create it if missing): insert a row with
   `available_at = as_of + 1 second`, assert `MarketView(as_of)` does not return it; a row at exactly
   `as_of` is returned.
7. **Shift test.** For every strategy on a fixture: compute signals for days D1..Dn with
   `MarketView(as_of = 09:00 ET of D)`, then append rows dated after Dn to the fixture and recompute.
   Signals for D1..Dn must be identical. Any difference is a leak.
8. **Labels check.** Forward returns and impact labels live only in `impact_labels`; grep for joins
   between a strategy/feature and that table. None expected.
9. **Splits check.** Any fitted parameter (strategy windows, aggregator weights, thresholds, trained
   models) carries the split it was fitted on; confirm `train` for strategy params and `validation`
   for aggregator/threshold. Confirm `evals/results/TEST_FINAL.lock` is absent unless the final run
   already happened, and no code path reads test-period rows outside `--final` runs.
10. **LLM memory check (when an LLM strategy or chat is involved).** The masked-prompt experiment
    (company name and dates replaced by placeholders) must be present in the eval; if unmasked
    performance is clearly higher, say so in `docs/evals.md`: the model remembers, it does not analyse.

## Output

A checklist in the reply with pass/fail per step and the file:line of every violation. Fix violations
in the same session when small; otherwise open a line in `agent_log.md` and block the backtest.
