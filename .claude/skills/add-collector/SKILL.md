---
name: add-collector
description: Add or change a data source (prices, candles, news, SEC filings, XBRL facts) so it lands in DuckDB with a correct available_at, a raw cache, rate limiting, quality checks, an offline test and a completeness eval. Use for any new or changed collector.
---

# add-collector

Rules: AGENTS.md R2 (available_at), R7 (single writer), R9 (SEC_USER_AGENT), R12 (offline tests), R15
(raw cache, rate limits, existing datasets first). Per-source `available_at` rules: `docs/decisions.md`
D-04.

## When to use

- a new source (e.g. Form 4, XBRL companyfacts, Binance klines)
- an existing collector changes its schema, its dedup key, or its `available_at` rule

## Steps

1. **Check existing datasets first** (assignment requirement). For labelled text prefer Financial
   PhraseBank / FNSPID; for filings use EDGAR directly. Synthetic data only for planted errors.
2. **Contract.** Define the table in `src/collectors/schema.sql` (or the DDL module): primary key /
   dedup key, `event_time TIMESTAMPTZ` (publication, acceptance, candle close) and
   `available_at TIMESTAMPTZ NOT NULL`, `source`, `raw_path`. Write the `available_at` rule as a comment
   and confirm it matches D-04; if the source is new, add it to D-04 in the same commit.
3. **Raw cache.** Every HTTP response is saved under `data/raw/<source>/<key>.json` before parsing; a
   re-run reads the cache and never re-downloads. Cache keys include the date range so dev and test
   collections stay separate (D-13).
4. **Rate limits and headers.** Finnhub ≤ 60/min, SEC ≤ 10/s with `SEC_USER_AGENT` from `.env`,
   Binance public limits; add sleeps and retries with backoff. Keys only from `.env`.
5. **Single writer.** Write through the shared writer helper that takes the file lock
   (`data/.duckdb.write.lock`) and uses `INSERT ... ON CONFLICT DO NOTHING` on the dedup key.
   Collectors are invoked from `src/collectors/run.py --periods ...` (Makefile `collect-dev` /
   `collect-test`) or from the API collect job, never from two processes at once.
6. **Quality checks as code** (`src/collectors/quality.py`): completeness per ticker/day (missing
   trading days), duplicate share, monotonic dates, price outliers (> 20 % day move flagged), timezone
   sanity (no naive timestamps), `available_at >= event_time`. Checks run at the end of every collect
   and fail loudly.
7. **Offline test** in `tests/collectors/test_<source>.py` with a recorded fixture response in
   `tests/fixtures/<source>/`: parse → rows with expected `available_at`; dedup on re-insert; quality
   checks pass; a malformed fixture is rejected.
8. **Eval** `evals/eval_collectors.py` → `evals/results/collectors.json`: completeness, duplicate share,
   missing days per source and split (metric: completeness; gold: the exchange calendar; target ≥ 99 %
   trading days for prices, 0 duplicates; why: downstream signals silently degrade on gaps).
9. **Demo slice.** Add a few rows of the new source to `data/demo/` so the app runs without keys.
10. Run the `time-leak-check` skill (steps 4–5), then `make lint && make test`, commit
    `feat(collectors): add <source>`, log line in `agent_log.md`, update
    `docs/02_data_understanding.md` with coverage figures.

## Output

Collector module, DDL, fixture, test, eval result, D-04 entry if new, demo rows.
