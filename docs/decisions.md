# Decisions

Format: `D-nn (date) — decision · why · alternatives rejected`. Cite ids in code comments, commits,
and `AGENTS.md`. Never delete an entry; supersede it with a new one that says so.

**D-01 (2026-11-13) — Layout follows the task brief, not the plan PDF.** `src/{collectors, marketview,
nlp, impact, strategies, sim, api, ui}` plus `src/llm.py`. Service functions live in each package
(`src/<pkg>/service.py`), schemas next to their component (`src/<pkg>/schemas.py`) and are re-exported
by `src/api/schemas.py`. Why: the brief is the latest instruction; one fewer package to explain.
Rejected: the plan's `src/services`, `src/schemas`, top-level `api/` and `ui/`.

**D-02 (2026-11-13) — Import root is `src`; the project is not an installable package.** `uv` runs with
`package = false`; the Makefile exports `PYTHONPATH=.`; pytest sets `pythonpath = ["."]`. Why: zero
build config, `python -m src.x.y` works from the repo root, graders run `make`. Rejected: hatchling
with `packages = ["src"]` (installs a package literally named `src`).

**D-03 (2026-11-13) — Decision cutoff is 09:00 America/New_York on day D; fills at the open of D.**
The plan said "close of D−1"; the brief says just before the open. Why: pre-market news and filings
(most 8-Ks are accepted after 16:00 or before 09:00) are legitimately known before the open, and the
rule is the same for every source. For crypto (24/7, hourly candles) the decision is at a candle close
and the fill is the next candle's open. Rejected: close of D−1 (throws away overnight filings), fills at
the close of D (the signal would see intraday data it was not given).

**D-04 (2026-11-13) — `available_at` per source.** Stored in UTC, never null.
- daily stock bar for day t: `t 16:00 ET + 30 min` (so the bar of D−1 is visible at 09:00 ET of D; the
  open of D is served only by the simulator's execution accessor);
- Finnhub news: the article `datetime` (publication time);
- SEC filings (8-K, Form 4): EDGAR `acceptanceDateTime`; if accepted after 17:30 ET, then 06:00 ET of
  the next business day (EDGAR dissemination);
- XBRL facts: acceptance time of the filing that carried the fact (10-Q/10-K), i.e. later than the
  8-K press release they are the gold for;
- Binance candle: candle close time;
- derived features (sentiment, events, facts, impact features): `max(available_at of inputs)` (see
  D-16); forward-return labels: `event_date + horizon`, stored in `impact_labels`, never served by
  `MarketView`.

**D-05 (2026-11-13) — Splits.** warmup 2024-01-02..2025-11-28 (prices only, 200-day MA warm-up);
train 2025-12-01..2026-05-29; validation 2026-06-01..2026-08-31; test 2026-09-01..2026-11-06. Why:
Finnhub's free plan serves roughly one year of news, so news-aware periods start in December 2025;
the test window is the most recent one possible and ends a week before today so forward returns and
late filings are complete; knowledge cutoffs of the free models are not stated
(`docs/inputs/free_models.md`), so "test after the cutoff" cannot be guaranteed and the masked-prompt
experiment is the control. If collection shows thinner news coverage than expected, the splits do not
move; the coverage is documented in `docs/02_data_understanding.md`. Rejected: longer train using
FNSPID (ends 2023; prices-only strategies get the warmup period instead).

**D-06 (2026-11-13) — LLM gateway.** OpenRouter free models only, in the priority order of
`docs/inputs/free_models.md`, configured in `config/llm.yaml`; primary `cohere/north-mini-code:free`
(JSON schema output). Hard daily budget of 200 calls (counter in `.llm_budget.json`, UTC day),
`BudgetExceeded` → deterministic fallback per task. On-disk cache keyed by model + prompt + schema +
items. Tracing is a local JSONL file per day in `data/traces/`, shown in the UI and the video; Langfuse
is a Should item. Why: no extra accounts for the grader, offline reproducibility, free tiers are
rate-limited anyway. Rejected: Langfuse as the only trace (needs keys), paid models.

**D-07 (2026-11-13) — One DuckDB file, one writer.** `data/stock.duckdb`; writers (collector CLI,
features CLI, API collect job) take `data/.duckdb.write.lock`; all other code opens `read_only=True`.
Why: DuckDB allows one writing process; the lock gives a clear error instead of a corrupt file.
Rejected: Postgres (setup burden for the grader), SQLite (weaker analytics).

**D-08 (2026-11-13) — Long-only.** Signal ≤ 0 means cash. Why: shorting needs borrow and margin
modelling that would be invented, and the brief requires it. Rejected: symmetric long/short.

**D-09 (2026-11-13) — Sentiment baseline is a Loughran–McDonald lexicon, not FinBERT by default.**
FinBERT is an optional extra (`uv sync --extra finbert`) and is compared in the eval when installed.
Why: torch is ~2 GB and makes `make setup` slow for the grader; the lexicon is a deterministic fallback
that always works. Rejected: FinBERT as the required baseline.

**D-10 (2026-11-13) — Event taxonomy = 8-K item numbers.** Classes: `1.01` material agreement,
`2.01` acquisition/disposition, `2.02` results of operations, `2.05` exit costs, `3.02` unregistered
sales, `5.02` officer/director change, `5.07` shareholder vote, `7.01` Reg FD, `8.01` other events,
`9.01` exhibits only, `none` (no event). Rare items are merged into `other_item` in the eval. Why: the
EDGAR index gives the gold labels for free; news gets the same labels so events line up across sources.

**D-11 (2026-11-13) — Impact label = abnormal return.** Stock return minus SPY return over 1 and 3
trading days starting at the first open after `available_at`; direction = sign. Evaluated with AUC and
directional accuracy. Why: removes the market move; chance level is interpretable (0.5).

**D-12 (2026-11-13) — Repo, docs and UI are in English.** The plan was written in Ukrainian; the
assignment, graders' rubric and code are English. The video may be in either.

**D-13 (2026-11-13) — Test-period data is collected separately.** `make collect-dev` fetches warmup,
train and validation; `make collect-test` fetches the test period and runs only inside
`make test-final` (or right before it). Why: during development the test rows are not even on disk,
so an accidental peek is impossible. Raw-cache keys include the period name.

**D-14 (2026-11-13) — `agent_log.md` at the repo root** (the plan had `docs/agent_log.md`). Why: the
brief says root; it is the file the agent must touch every session, so it should be visible.

**D-15 (2026-11-13) — Repo rules are tested.** `tests/test_repo_rules.py` pins the sha256 of
`config/splits.yaml`, forbids provider access outside `src/llm.py`, checks `.env` is untracked, and
runs the commit-msg hook on a sample message. Why: rules that can be checked by a test are not rules
that depend on memory. This is scaffolding, not product code.

**D-16 (2026-11-13) — NLP features are precomputed per split (`make features`).** Sentiment, event
type, extracted facts and impact features are written once into feature tables with
`available_at = max(available_at of inputs)`; strategies read them through `MarketView`. Why:
backtests become deterministic and repeatable, the LLM budget is spent once, and the arena does not
depend on the model being up. Online calls remain for `/analyze` and `/chat`.

**D-17 (2026-11-13) — Aggregation.** Must: Hedge (multiplicative weights) over strategy signals,
walk-forward, weights from the last 20 trading days, threshold chosen on validation by the
threshold→return curve. Should: stacking (gradient boosting + SHAP) and an LLM researcher that reads the
arena report and proposes rules **as YAML** interpreted by code (never executable code), at most 5
iterations on validation, each counted as an attempt in the deflated Sharpe. Why: Hedge has a regret
bound and no fitting; the YAML rule format honours R5.

**D-18 (2026-11-13) — Watchlist.** AAPL, MSFT, NVDA (technology), JPM, GS (financials), XOM, CVX
(energy), WMT, COST, MCD (consumer), JNJ (healthcare), BA (industrials); SPY as the benchmark; BTCUSDT
and ETHUSDT for crypto. Why: four-plus sectors as the plan asks, liquid names with frequent news and
regular 8-Ks; BA adds a name with non-earnings events (5.02, 8.01).

**D-19 (2026-11-13) — Risk defaults** (`config/risk.yaml`): 20 % max position, 8 % stop-loss, 15 %
drawdown halt, 10 bps commission per side, 5 bps slippage, confidence threshold left `null` until the
validation run fills it. Why: common textbook values; none are tuned on test.

**D-20 (2026-11-13) — Git hygiene is automated.** `.githooks/commit-msg` strips `Co-Authored-By`,
`Claude-Session` and "Generated with Claude" lines; `make setup` runs
`git config core.hooksPath .githooks`; `.claude/settings.json` disables attribution. Why: AGENTS.md R1
should hold even when a prompt forgets it.

**D-21 (2026-11-13) — Makefile uses `.RECIPEPREFIX = >`.** Why: Windows editors replace tabs with
spaces and silently break recipes; GNU make ≥ 3.82 is available on the dev machine and in Git Bash.

**D-23 (2026-11-13) — All schemas in one module `src/schemas.py`** (supersedes the schema placement
in D-01). Why: the brief asks for it; one import for collectors, NLP, API and LLM structured output;
the file is small enough to read in one go. Component packages do not define their own models.

**D-24 (2026-11-13) — Availability rules** (supersedes D-04). price bar = bar close time (16:00 ET
stocks, candle close crypto); news = provider `datetime`; filing = EDGAR `acceptanceDateTime` (no
next-morning adjustment); XBRL fact = filed date of the 10-Q/10-K at 00:00 UTC; derived features =
their source's `available_at` (D-16 stands); signal = `as_of`; trade = fill time. Why: simpler, matches
the brief, and errs on the side of "known slightly earlier" only for filings accepted after 17:30 ET,
which the time-leak shift test tolerates because the cutoff is the next morning anyway.

**D-25 (2026-11-13) — `src/db.py` is the single connection factory and carries the access guard.**
`connect()` is read-only by default; `writer=True` takes `data/.duckdb.write.lock`. `guarded()` is a
context manager (contextvar) under which `connect()` raises `DirectDataAccessError`; `MarketView` uses
a private connection path. Strategies, features, impact and chat tools run inside the guard. Why: the
rule "all history through MarketView" becomes a runtime error instead of a code-review item. Rejected:
monkeypatching `duckdb.connect` globally (hides collector bugs).

**D-26 (2026-11-13) — Tests-first convention.** Tests written before the implementation are marked
`xfail(strict=True, raises=NotImplementedError)`; the suite stays green, the tests still execute and
fail for the stated reason, and an implementation must remove the marker (strict XPASS fails). Why:
AGENTS.md R1 requires green tests before a commit; this keeps both rules.

**D-27 (2026-11-13) — API surface for the two-day build** (supersedes the endpoint table that was in
docs/01): GET `/watchlist`, `/brief`, `/signals?ticker&day`, `/events/{ticker}`, `/evals`, `/models`,
`/jobs/{id}`; POST `/analyze`, `/collect`, `/arena`, `/chat`; `/health`. The intake service (phrase →
tickers) stays a Must service used by the UI; `POST /watchlist` and `/backtest` are folded into
`/arena` and dropped from the stub. Every endpoint answers 501 until its service exists, so the OpenAPI
contract is complete from day one.

**D-28 (2026-11-13) — Impact labels.** `gap_ar` = abnormal return (vs SPY) from close(D−1) to open(D),
realised at our entry, diagnostic only; `drift_ar` = abnormal return from open(D) to close(D+1), the
tradable target and the impact model's label. D = first trading day whose cutoff is ≥ the event's
`available_at`. Both in `impact_labels`, never served by MarketView. Supersedes the 1–3 day horizon of
D-11: one fixed horizon (D+1 close) keeps the label unambiguous and matches how the simulator trades.

**D-22 (2026-11-13) — Dependencies declared now** (all pinned by `uv.lock`): pydantic, pyyaml,
python-dotenv, tzdata (zoneinfo on Windows), duckdb, pandas, numpy, pyarrow, httpx, yfinance, fastapi,
uvicorn, streamlit; dev: pytest, ruff; optional extra `finbert`: torch, transformers. Anything else
needs its own D-nn (AGENTS.md R14). Not added yet: `smartmoneyconcepts` (Should), `scikit-learn`,
`lightgbm`, `shap` (stacking, Should), `apscheduler` (Should), `langfuse` (Should).

**D-29 (2026-11-13) — Collection sources and universe (data-preparation phase).**
- *News: Finnhub only* (the task prompt's fallback track). A FNSPID 2019–2022 dev track would need
  periods outside the frozen `config/splits.yaml` (R3) and FNSPID ends in 2023 (already rejected in
  D-05), so it cannot cover train (2025-12..) at all. No FNSPID download was attempted. Finnhub's
  free window starts 2025-11-13, so it covers train, validation and test; the warmup period gets
  news only for its last two weeks.
- *Finnhub returns at most ~250 items per call (newest first)*: windows start at 31 days and are
  halved while a response has ≥ 200 items; single days still at the cap are reported as gaps.
- *Universe* `config/universe.yaml`: 77 S&P 100 names (no dotted share classes), CIKs verified
  against SEC `company_tickers.json` by `python -m src.collectors.tickers`. *Watchlist* 10 names
  (supersedes D-18: CVX and COST dropped; every sector keeps at least one name).
- *XOM* moved to a holding company (new CIK 0002115436) in July 2026; filings up to Aug 2026 are
  under the old CIK 0000034088, so a company may list `former_ciks` and all are collected.
- *Prices*: yfinance `auto_adjust=False` Close (split-adjusted, not dividend-adjusted: dividend
  back-adjustment rewrites history with later dividends). Stooq is the fallback but currently
  answers with a JavaScript bot check from this machine, which the collector detects and logs.
- *EDGAR `acceptanceDateTime` is genuine UTC*: checked on 2026-11-13 against the filing index page
  (`21:30:40Z` in JSON = `16:30:40` Eastern on the index page). Stored without conversion.
- *Exchange calendar* for completeness = SPY's sessions (independent of the stock being checked).
- *Form 4* headers are stored (insider strategy is a Should item); no Form 4 text yet.
- `news` primary key is `(id, ticker)`: one Finnhub article can be tagged with several tickers.
- Not collected yet: Binance crypto candles (Should, not in this task).
Rejected: FNSPID-based dev track (breaks frozen splits), dividend-adjusted prices, treating
`acceptanceDateTime` as Eastern time (would have been wrong by 4–5 hours).

**D-30 (2026-11-13) — XBRL fact `available_at` = acceptance time of the 10-Q/10-K that carried it**
(supersedes the XBRL clause of D-24). The accession number (`accn`) of each fact is joined to the
filing's `acceptanceDateTime`; if the accession is unknown, filed date at 22:00 ET (the EDGAR
filing window closes then, so the real acceptance is never later). Why: D-24's "filed date 00:00
UTC" is 19:00–20:00 ET of the *previous* day; a 10-Q accepted at 16:30 ET on day D would have been
visible at the 09:00 ET cutoff of D, a look-ahead of up to ~20 hours. Only each filing's own
reporting period is stored (comparative prior-year values are skipped), quarters = 80–100-day
durations, fiscal years (10-K) = 350–380 days.

**D-31 (2026-11-13) — Test-period data is not collected in this session.** The task asked to
"collect the test period but not analyse it"; AGENTS.md section 7 and D-13 say `make collect-test`
runs only right before `make test-final`, and AGENTS.md says a rule wins over a prompt. So
`make collect-test` is implemented and covered by offline tests but was not run. Nothing is lost by
waiting: the test period (2026-09-01..11-06) stays inside Finnhub's one-year free window until
September 2027, and EDGAR keeps everything. Every dev collection fails its quality check if a
test-period row reaches the database.

**D-32 (2026-11-13) — LLM gateway details** (extends D-06). `LLM_MODE=mock` (forced in
`tests/conftest.py`) never opens a socket: answers come from a registered responder, otherwise the
caller's fallback runs. The daily budget counts every HTTP *attempt* (retries included) and is
checked before the request; cache hits are free. Client-side throttle 20 requests/min (live only).
A model is skipped for the rest of the run after 2 batches in which all its retries failed.
`config/llm.yaml` base URL fixed to `https://openrouter.ai/api/v1` (`api.openrouter.ai` does not
resolve). Rejected: counting one budget unit per batch (a retry storm could exceed the provider's
free quota while the counter says otherwise).

**D-33 (2026-11-14) — EDA tooling and its read path** (data-understanding phase).
- *Dependency*: `matplotlib` in the **dev** dependency group only (figures for `make eda`; the
  runtime image and the API never import it). Rejected: plotly/altair (heavier, need a browser
  renderer for static PNGs); hand-written SVG (no value over matplotlib).
- *Read path*: `src/eda/` is offline analysis, not a decision path. It opens the database with
  `src.db.connect()` (read-only) and every query is filtered to rows available before the first test
  cutoff, so the test period stays unread even if `make collect-test` had been run (R3). Forward
  returns (`gap_ar`, `drift_ar`) come from the pure functions in `src/impact/labels.py`, which take
  price series as arguments; the EDA, not the impact module, fetches those series.
- *Test-period lock*: `src/splits.py` raises `TestPeriodLocked` when `MarketView` is built for an
  `as_of` at or after the first test cutoff, unless the process runs inside `final_run()`, which only
  the `--final` entry points of `make test-final` open. `tests/test_test_period_lock.py` pins this
  and scans the repo for any other reference to the test period.

**D-34 (2026-11-14) — Data-quality rules, splits and the news labelling sheet.**
- *Splits*: `config/splits.yaml` was already frozen and committed (hash pinned in
  `tests/test_repo_rules.py`); it is not edited. The task's "train/val on the dev track" does not apply:
  there is no FNSPID dev track (D-29). The lock `tests/test_test_period_lock.py` is the new test for
  "no code path except `make test-final` reads test-period rows".
- *Dev cut is exclusive*: offline analysis reads `available_at < 2026-09-01 13:00 UTC` (first test
  cutoff), the same bound as the collector quality check; a row exactly at the cutoff is the first
  test day's input and is not analysed.
- *Duplicates*: the news key stays `(id, ticker)`; 32 % of articles are tagged with several tickers,
  which is information. Syndicated copies are found by `clean.dedup_key` (NFKC, case, punctuation)
  within 24 h per ticker and dropped by the future relevance component, not deleted from the database.
- *Language heuristic*: Latin-script share plus foreign function words; ambiguous words that occur in
  English names or contractions (de, van, la, le, di, per, et, ve) are excluded after "I've" and
  "Dirk Van De Put" were flagged. A heuristic, not a language identifier.
- *8-K header stripping*: removes the collector tag line, `Item x.xx` headings (also split over lines,
  wrapped titles), `Section n -` headings and inline item mentions restricted to real item numbers
  (reconciliation tables contain "special items 0.06").
- *Labelling sheet*: 150 headlines (the task's file name says 100) drawn by `evals/make_news_sheet.py`;
  strata = coincident 8-K item group (same ticker, accepted within ±24 h) × year × ticker, quotas in
  `config/data_quality.yaml`. The item stratum lives in `news_100.strata.csv`, not in the sheet, so the
  annotator does not see it. The sheet is saved as `news_100.draft.csv` (label-helper skill: a draft
  is renamed to `news_100.csv` and frozen only after the human fills it; R4 freezes committed gold
  sets). Rejected: a `llm_suggestion` column (the task forbids pre-filling), proportional sampling
  (2.02/5.02/1.01 would get ~0 rows).

**D-35 (2026-11-14) — Simulator and the first three strategies** (modeling phase).
- *Timing*: the decision for D uses `MarketView(cutoff_for(D))` (newest bar = close of D-1);
  fills at the open of D; a position closes at the close of D+h (`horizon_days` per strategy) or
  at the next open whose score is <= 0, whichever comes first; everything left is sold at the
  last close of the split, so every trade is a closed round trip. The open and close of D come
  from `src/sim/execution.py`, loaded once for the split's dates and never passed to a strategy.
- *Strategy contract*: `signal(ticker, view)`; no separate `as_of` argument (the view carries
  it, so the two cannot disagree). Runs inside `src.db.guarded()`.
- *Sizing*: equal weight `min(1/N, max_position_pct)` of equity marked at the open, N = names
  held after today's entries; existing positions are not resized (no turnover from drift);
  a buy spends at most the cash there is (no leverage); fractional shares.
- *Costs*: 5 bps per side on traded notional (task requirement), stored as
  `commission_pct_per_side: 0.0005` and `slippage_pct: 0` in `config/risk.yaml`; supersedes the
  10 + 5 bps of D-19. Stop-loss and drawdown halt in `risk.yaml` are not applied yet (risk manager
  is a separate component).
- *Momentum*: simplified 12-1 as *time-series* momentum, `close[t-21]/close[t-252] - 1 > 0`;
  per-ticker so it fits the `Signal` contract and a 10-name watchlist (a cross-sectional
  top-decile rank needs a larger universe). h = 20.
- *Mean reversion*: 5-day return scaled by 60-day daily volatility; long when z <= -1, h = 5.
  Replaces the RSI(14) placeholder in `config/strategies.yaml`.
- *Metrics*: Sharpe with rf = 0 and sqrt(252); max drawdown from the initial capital; one-way
  turnover = (buys + sells) / 2 / mean equity; hit rate per closed round trip after costs;
  95 % CIs by moving-block bootstrap (5-day blocks, fixed seed) because positions persist across
  days and i.i.d. resampling would understate the variance.
- *Universe*: the 10-name watchlist by default (`backtest.universe`). Target of a backtest eval =
  buy-and-hold Sharpe on the same names and split.
- *Splits*: `make backtest` accepts train and validation only; the test-period entry point
  (`--final`) is not written yet, so `make test-final` cannot run until it is.
Rejected: fills at the close of D (would use intraday data the signal did not see), daily
rebalancing to equal weight (turnover without information), i.i.d. bootstrap.

**D-36 (2026-11-14) — Walking skeleton: relevance, FinBERT sentiment, finbert_v0, API v0, UI v0**
(modeling + deployment phases).
- *Relevance*: a news item counts for its tagged ticker only if the symbol or a company alias
  (`config/nlp.yaml`) appears in the headline or summary; symbols that are words or very short
  (BA, T, LOW, ...) need a marker (`$BA`, `(BA)`, `NYSE: BA`). On train about 40 % of the
  watchlist's tagged news passes. Rejected: trusting Finnhub's ticker tag (most items under a
  ticker are about another company), an LLM relevance judge (budget, and not needed for v0).
- *Dedup*: MinHash (128 permutations, character 4-shingles of the normalised headline, Jaccard
  >= 0.6, same ticker within 24 h), implemented with numpy, no new dependency. The earliest copy
  is kept and an item is compared only with earlier kept items, so the decision for an item
  depends only on the past (same answer on a whole split and inside `MarketView(as_of)`).
  Rejected: `datasketch` (one more dependency for 40 lines), embedding cosine (needs a model at
  signal time), exact `dedup_key` only (misses "UPDATE 1-" / "- Reuters" variants).
- *Sentiment*: FinBERT `ProsusAI/finbert` pinned to HF commit 0bc2ac1, headline only (the gold
  labels are headline only), score = P(pos) - P(neg). D-09 stands: torch/transformers stay the
  optional `finbert` extra already declared in D-22; without it the service falls back to a
  lexicon and logs it. The lexicon is a compact list of finance cue words in the spirit of
  Loughran-McDonald (in `config/nlp.yaml`), not the full LM dictionary; it was written before
  the gold labels were read and is not tuned on them. Every input goes through
  `prepare_text` (HTML stripped, 8-K item headings removed as in D-34, truncated).
- *Cache*: `make features` writes the `sentiment` table (D-16), which is the cache: rows already
  present for a method are not rescored. Its key becomes `(source_ref, ticker, method)` because
  one article can be tagged with several tickers; the local table was empty and was recreated.
  Watchlist only by default (`features.universe`): ~78k articles, about an hour on CPU.
- *finbert_v0*: for day D, mean score of the relevant, deduplicated, scored items with
  `available_at` in (09:00 ET of the previous weekday, 09:00 ET of D]; at least 1 item, else 0
  (cash); hold to the close of D+1 (the `drift_ar` horizon, D-28); entry when score > 0
  (`confidence_threshold` is still null). Holidays are not skipped when finding the previous
  weekday (a window after a holiday is one day short). Nothing is fitted.
- *API*: `GET /signals?ticker=&as_of=` (the stub's `day` parameter is renamed to `as_of`, as
  the task asks): `as_of` is a trading day (cutoff 09:00 ET of D) or a tz-aware instant; naive
  instants are 422. The default is the last validation day, not today, because MarketView
  refuses test-period views (R3, D-33); test-period dates answer 403, unknown tickers 404, a
  missing database 503. `aggregated` stays null until the aggregator exists. `POST /analyze`
  returns sentiment only: `events` is null and `facts` empty until those components exist
  (rather than a made-up "none" event). `GET /evals?component=` serves `evals/results`, with
  the record's `details` (equity curve) added to `EvalResult`. `/health` is also served as
  `/api/v1/health`. `docs/openapi.json` is generated by `src.api.export_openapi` and a contract
  test fails when it differs from the app.
- *UI v0*: ticker + optional day -> signals table; equity curve of
  `backtest_finbert_v0_validation` from `/evals`. httpx only.
- *Gold set*: the sentiment eval reads the human-labelled `evals/labels/news_100.csv` as-is.
  Freezing it (header update, removing the draft) was not done in this session: edits under
  `evals/labels/` were refused by the tool permission layer, so the user freezes it.

**D-37 (2026-11-14) — `news_100` is frozen** (supersedes the gold-set bullet of D-36).
The user asked for label-helper steps 5-8 after the human finished labelling. Step 5: no LLM
suggestions, so kappa is n/a. Step 6: no planted errors, because the set has no LLM labeller or
judge to test (the eval plan's planted errors are for LLM components); this is stated in the
header. Step 7: only the provenance header changed (body byte-identical to the labelled file,
non-label columns identical to the draft, every label within the D-10 / pos-neu-neg spaces),
the draft was removed, and the set was committed alone. Step 8: README row, docs/evals.md.
`news_100.csv` and `news_100.strata.csv` are hash-pinned in `tests/test_repo_rules.py`, and
`evals/make_news_sheet.py` refuses to write once the frozen file exists (a rerun would have
redrawn the committed strata). New rows go to `news_100.v2.csv`.

**D-38 (2026-11-14) — Test-period sentiment is precomputed by `make features-test` into parquet**
(data preparation; amends D-13, D-16 and AGENTS.md R3 for this one step).
- *What*: `src/nlp/features_test.py` runs the dev scoring path (`features.pending` +
  `features.score_frame`, FinBERT pinned to the D-36 revision, no lexicon fallback) on the news of
  (first test cutoff - `lookback_days_before_split`, cutoff of the last test day] for the
  watchlist, skipping pairs the dev run already wrote, and writes `data/test_scores.parquet` plus a
  manifest (model, revision, window, counts). `make test-final` reruns it in place of
  `src.nlp.features --splits test` (a no-op when complete; it scores only news that a late
  `collect-test` added), and `MarketView.sentiment()` adds the parquet rows only for views at or
  after the first test cutoff, which `assert_readable` already confines to `final_run()`; a pair in
  both sources is taken from DuckDB. A missing parquet in the final run raises.
- *Why*: FinBERT on ~86k test articles takes over an hour on CPU; doing it inside `test-final`
  makes the one-shot run long and fragile, and writing DuckDB would hold the single writer lock
  (R7) for that hour. Inference with a frozen model has no tunable input and produces no metric,
  so running it early does not use the test period in the R3 sense, provided nobody looks at the
  scores: the CLI and manifest carry counts, throughput and ETA only.
- *Guards*: allowlisted in `tests/test_test_period_lock.py` as the only module that may read
  test-period news text; it has no SQL of its own and imports nothing from sim, strategies,
  impact, MarketView, API or LLM (tests); its test fixture drops every table except `news` and
  `sentiment`; it never takes the write lock (test); one short read-only connection is closed
  before inference. Resumable: one atomically written part per `chunk_articles` articles in
  `data/test_scores.parts/`, a resume with a different model/revision is refused.
- *Side fix*: `src.marketview` now imports `src.splits` as a module, which removes the latent
  circular import noted in the agent log.
- Rejected: scoring test news inside `test-final` (long, fragile one-shot run), writing it to
  the DuckDB `sentiment` table (blocks the single writer for an hour), a separate DuckDB file
  (another writer and lock to manage; parquet is read directly by DuckDB).

**D-39 (2026-11-14) — Event classifier, fact extraction, eval tooling, LLM bake-off protocol**
(modeling phase).
- *Events*: (a) rules = item codes of the tagged ticker's 8-Ks accepted within +/- 24 h of the
  headline (same window as the news_100 strata, D-34), 9.01 dropped, items outside D-10 ->
  other_item; the rule label is available at max(headline, filing acceptance) (D-16, D-24).
  (b) TF-IDF + one-vs-rest logistic regression on headlines, trained on the train split with (a)
  as weak labels (R3), settings fixed a priori in `config/extraction.yaml`. Why: (a) is free and
  exact about filings but blind to text; (b) reads text but inherits (a)'s noise. Rejected:
  keyword rules (hand-tuning against the only human gold set), training on validation.
- *Facts*: regex baseline on the first 4,000 chars of EX-99.1 (GAAP only: matches preceded by
  "adjusted / non-GAAP / organic / excluding / core" are skipped) and the LLM through
  `src/llm.py` with the same pydantic schema; the model returns the number as written plus a
  scale word and code multiplies. Every fact needs a grounding quote that occurs in the source
  and holds the value. Gold = XBRL 3-month Q1-Q3 values of 30 dev 8-K 2.02 filings. Rejected:
  letting the model output scaled values (arithmetic errors are ungroundable).
- *Dependency*: scikit-learn (TF-IDF, logistic regression), default install (~10 MB, no GPU).
  Rejected: a hand-written numpy TF-IDF/LR (more code to test, same result).
- *Second sentiment gold*: `zeroshot/twitter-financial-news-sentiment` validation split, pinned
  by revision + sha256 in `config/evals.yaml`, raw cache in `data/raw/hf/` (R15). Why: FinBERT
  was trained on PhraseBank. Bootstrap CIs (i.i.d., 2,000 resamples) for classification evals.
- *LLM evals and budget*: `make eval` sets `LLM_DAILY_BUDGET=0` unless `EVAL_LLM_LIVE=1`, so it
  serves cached responses and otherwise falls back; it never spends free-tier calls. The bake-off
  (`evals/eval_llm_bakeoff.py`, cohere/north-mini-code vs nvidia/nemotron-3-ultra, 3 calls each,
  10 items per call, <= 15 calls) is run on purpose (never parallel/decisions-fast, it is
  web-grounded); it ran on 2026-11-14 with 10 live calls, result in the model-choice bullet below.
- *Model choice (bake-off 2026-11-14, `evals/results/llm_bakeoff.json`, git 44fc8b2)*: for
  `events` and `facts` the primary is `nvidia/nemotron-3-ultra-550b-a55b:free`, the fallback
  `cohere/north-mini-code:free`, then the deterministic fallback of `config/llm.yaml`. Measured
  (10 live HTTP calls in total, 20 headlines + 10 releases / 30 XBRL pairs, so every number is
  indicative, not significant): schema adherence nemotron 0.75 (3/4 calls) vs cohere 0.33 (2/6);
  event exact-set match 0.90 vs 0.85 (18 vs 17 of 20); facts within 1 % 0.63 vs 0.40, coverage
  0.87, grounding 0.73, quarter match of found facts 0.12. Cohere's facts figures are the regex
  fallback, not the model: all 3 of its facts calls returned empty content (reasoning used the
  whole 8,192-token cap), so cohere is a fallback for `events` in practice and for `facts` only
  nominally. Latency per 10-item call: nemotron 32-36 s (events) and 139-142 s (facts), cohere
  17-20 s (events). inkling-2-small stays for chat/intake only; liquid/lfm-3-1.7b and
  liquid/lfm-2.5-2.6b are removed (not measured, their only tasks were sentiment/events);
  parallel/decisions-fast is never used.
- *Sentiment stays on FinBERT* (`config/nlp.yaml`, ProsusAI/finbert): no LLM is routed for
  `sentiment`. Bake-off macro-F1 on 20 headlines was 0.43 (nemotron) / 0.38 (cohere), not above
  FinBERT's 0.52 on the 150-item gold, and an LLM in the final run would cost budget per day.
- *LLM fact extraction policy*: (1) offline batch only (`make features` / evals), never called
  synchronously from an API request: a 10-release call took 139-142 s; the API serves stored
  facts. (2) The grounding check (`grounded_only`) is mandatory: 27 % of nemotron's facts were
  ungrounded and are dropped, never repaired. (3) The fiscal period is taken from XBRL (the
  filing's report period), never from the model: the model's period matched the gold quarter for
  only 12 % of found facts. Rejected: LLM sentiment (no gain on 20 items, budget), cohere as
  facts primary (0/3 facts calls returned content), synchronous LLM facts in `/api/v1/events`
  (latency far above a request timeout).
- *Budget counter fix*: `src/llm.py` consumes the daily counter only for real provider calls
  (mode `live`). Mock-mode attempts (tests, dry-runs) are free; earlier dry-runs had inflated
  `.llm_budget.json` with 30 phantom calls. Test: `test_mock_attempts_do_not_consume_the_daily_budget`.
- *Empty-content fix (2026-11-14)*: the first live bake-off scored 0.0 adherence because every
  reply had `content: null` (reasoning models spent the 1024/2048 `max_tokens` on reasoning) and
  the gateway logged it as `invalid_json` / "all models failed". Now `max_tokens: 8192` and
  `reasoning_effort: low` come from `config/llm.yaml`; an empty reply is status `empty_content`
  with finish_reason, completion tokens and reasoning size; the final error names the last cause.
  The 0.0 result in `evals/results/llm_bakeoff.json` is an artefact of this bug, not a model score.

**D-40 (2026-11-14) — Signal stack: impact model, calibration, event strategy, equal-weight
aggregator, risk manager** (modeling phase, dev splits only).
- *Features* (`src/impact/features.py`): per (ticker, D) from `MarketView(cutoff(D))` only:
  FinBERT mean/min/max/count of D's news window (`src/nlp/daily.py`, shared with finbert_v0),
  8-K item one-hots in the window, `growth_momentum` = YoY growth of the latest XBRL revenue /
  diluted EPS vs the same fiscal period a year earlier (available at the 10-Q/10-K acceptance,
  D-30; not "surprise": no consensus data), prior-day abnormal return, 20-day volatility. The
  same function serves training and the live strategies. Panels cached as parquet in
  `data/features/` keyed by a hash of the feature config (`make train`).
- *Labels*: `drift_ar` per (ticker, session) is built in `src/sim/labels.py`, because only
  `src/sim` may read the open of D (R2); `src/sim/impact_train.py` consumes the label frame as a
  training target and never as a feature. No label crosses a split boundary.
- *Models*: logistic regression (standardised) and HistGradientBoosting (no early stopping:
  it draws a shuffled holdout), both fit on train in time order.
- *Validation cut*: validation is split in time order into val_a (first half of sessions) and
  val_b. Everything "tuned on validation" (model choice by AUC, Platt calibrator, the hold /
  threshold of `impact` and `aggregate_equal`) uses val_a only; val_b is reported and never used
  for a choice. Why: fitting the calibrator and the threshold on all of validation and reporting
  on it would be in-sample. Platt fixed a priori over isotonic (about 300 rows in val_a);
  isotonic is still reported.
- *Event strategy*: per-item mean `drift_ar` over all 77 universe names' 8-Ks decided in train
  (`event_items.json`); score = mean of the weights of the window's items with >= 10 train
  events / 1 %. Hold period and entry threshold of `event` are chosen on train (strategy
  parameters, R3).
- *Holding period / threshold*: each stacked strategy holds exactly `hold_days` sessions
  (engine `min_hold_days` blocks score exits) and enters only above `entry_threshold`; grid
  `selection` in `config/strategies.yaml`, best Sharpe after costs with the risk manager, at
  least 10 round trips (`src/sim/select.py`, `selected.json`). The task asked for the threshold
  on train only; for `impact` and `aggregate_equal` that would pick it where the impact model is
  in-sample (train AUC 0.81), so they use val_a, which R3 also prescribes for the confidence
  threshold.
- *Aggregator*: fixed equal weights of the sentiment, event and impact scores. Hedge (D-17) not
  built in this session.
- *Risk manager* (`src/sim/risk.py`): position cap, stop-loss observed at a close and filled at
  the next open, drawdown halt from the running peak (seeded with the initial capital), cooldown
  of `cooldown_days` sessions after a losing exit. Applied to `backtest.risk_managed` strategies
  only, so earlier baselines (and the buy-and-hold target) keep their numbers.
Rejected: shuffled CV (time leakage), calibrating on train predictions (in-sample), LightGBM
(one more dependency for 1,220 rows), labelling events by news text (TF-IDF events F1 0.28).

**D-41 (2026-11-14) — Trained artifacts are not committed; targets that need them guard on them.**
`data/models/{impact.joblib,event_items.json,selected.json}` and `data/features/*.parquet` are
produced by `make train` on dev data and stay git-ignored (binary, reproducible, ~150 kB, and a
committed model would freeze a choice that `make train` is meant to re-derive). `make eval`,
`make arena` and `make backtest` for `all` or a stacked strategy depend on `need-train`, which
fails with a `make train` hint; `make backtest STRATEGY=momentum` and friends do not. Why: on a
fresh checkout `evals/run_all.py` caught the missing-model `FileNotFoundError` as "gold set not
on disk" and skipped the aggregator and six backtest records without failing (found in the Gate 2
review, `docs/gate2_review.md`). Rejected: committing the artifacts (see above), guarding inside
Python only (the Makefile is where a grader starts; the Python-side fix is `L-12`).

**D-42 (2026-11-14) — Strategy arena: controls, deflated Sharpe, trials log** (evaluation phase,
validation only). `make arena` runs `arena.strategies` (config/strategies.yaml) on validation
through the simulator and writes `evals/results/arena.json` and `docs/05_arena.md`.
- *Where*: the service is `src/sim/arena.py` (+ `src/sim/controls.py`), because the controls
  replay execution prices and only `src/sim` may read open(D) (R2, pinned by
  `test_impact_and_strategies_never_import_execution_or_labels`). `src/strategies/arena.py` is
  only the entry point the Makefile (and later `make test-final`) names.
- *Leaderboard*: Sharpe after costs with the existing moving-block bootstrap CI, plus the Sharpe
  on val_b alone: `impact` and `aggregate_equal` chose calibrator and hold/threshold on val_a
  (D-40), so their full-validation figure is partly in-sample.
- *Ablation*: `AggregateEqual.parts_log` records the component scores during the full run; the
  drop-one variants replay them (same hold, threshold, risk manager) instead of recomputing the
  features three times (~45 s each).
- *Random controls*: each control keeps the strategy's round trips (count, length in sessions,
  open or close exit) and places them at random; two modes: `names` (same entry sessions, random
  tickers: same number of positions every day, so exposure and trade count match almost exactly)
  and `timing_and_names` (entry sessions random too; a dense strategy may not fit and the
  achieved count is reported). Close exits use a new optional engine hook
  `horizon_for(ticker, day)`, without which a control exiting at the open frees cash the
  strategy did not have and is sized differently (finbert_v0: 53 % vs 71 % exposure before the
  fix). Placement is uniform over free (ticker, start) slots from an occupancy grid, not
  rejection sampling (which left trips unplaced). No risk manager on controls (its vetoes
  would break the trade-count match). Percentile = share of controls with a lower Sharpe.
  n = 200 per strategy and mode (final, set by the user; the first request said 1,000): a
  percentile then moves in steps of 0.5 points, enough to tell 50 from 95, and the run takes
  ~3 min instead of ~10.
- *Deflated Sharpe* (Bailey & Lopez de Prado 2014): N = every row of `evals/trials.csv`
  (selection grids of `make train`, impact model and calibrator alternatives, arena strategies,
  ablation variants); the variance of trial Sharpes uses the rows that have one (annualised in
  the log, converted to per day). The log is upserted by trial id, never truncated, so a future
  Hedge or LLM-researcher iteration adds to N (D-17). Random controls are not trials (not
  candidates). Target 0.95. Normal CDF from `statistics.NormalDist` (no new dependency).
Rejected: N = arena strategies only (ignores the 60-point selection grids, the real search);
permutation of daily returns as the control (destroys the trade structure and exposure);
rejection-sampled placement (biased against long trips, under-filled dense strategies). Not
done: the masking experiment (out of scope by request), `--final` entry points (L-01).

**D-43 (2026-11-14) — The final run is one function and takes the lock first** (fixes L-01).
`src/sim/final.py::run_final()` is the only code that opens `splits.final_run()`; both `--final`
entry points (`src.strategies.arena`, `src.sim.run`) call it, and `make test-final` calls one of
them once (after `need-train`, `collect-test`, `features-test`). Order: (1) `preflight()` checks
file existence only (models in `data/models`, `data/test_scores.parquet` with a complete
manifest, no lock): a missing prerequisite leaves the period unused; (2) `final_run()` creates
`TEST_FINAL.lock` with exclusive create (started_at, git sha) **before** anything is read and
appends `finished_at` + `status=ok|failed` on exit, so a crash consumes the period instead of
allowing a second read; (3) the arena on the test split (same code as validation; no val_b
column; `evals/trials.csv` is read for N, not extended) and one `backtest_<strategy>_test`
record per strategy from the same simulations. Outputs: `evals/results/arena_test.json`,
`backtest_*_test.json`, `docs/06_final_run.md`. `resolve_split` accepts the test split only
while `final_run()` is open. Rehearsed on validation with the split name, lock and outputs
redirected to a scratch directory (the rehearsal found a `relative_to` crash in the report
path that would have failed a real final run after the lock; fixed). Rejected: two separate
`--final` processes (the second either refuses or needs a weaker lock); writing the lock after
the runs (the L-01 hole); a lock per stage (allows partial re-reads).

**D-44 (2026-11-14) — Hedge aggregation and the confidence threshold** (completes D-17's Must;
`src/strategies/hedge.py`, `src/sim/select_hedge.py`). `aggregate_hedge` weights the
`aggregate_equal` components (sentiment, event, impact) by multiplicative weights over the last
`aggregator.lookback_days` = 20 *realised* sessions: loss of a component on (ticker, d') =
(1 − s·sign(drift_ar)) / 2, averaged over tickers; drift_ar of d' is computed from the view's own
bars with `src.impact.labels.abnormal_returns`, so the newest usable label at the cutoff of D is
d' = D−2 (no label table, no execution prices: R2; a test pins "never before its close"). The
confidence threshold `p_min` turns a suggestion into cash when the impact model's calibrated p is
below it; then the usual entry threshold applies. eta ∈ {0, 1, 3, 10} × p_min ∈ {off, 0.50,
0.52, 0.55} are chosen on **val_a only** with the same rule as D-40 (best Sharpe after costs with
the risk manager, ≥ 10 round trips); hold and entry threshold are `aggregate_equal`'s. eta = 0 /
p_min off is fixed equal weights and is in the grid. **Result: equal weights without a threshold
won on val_a (Sharpe 1.14, 69 round trips); every eta > 0 and every p_min was lower** (best
alternative eta 3 / off 1.03; any p_min ≤ 0.66), so `aggregate_hedge` currently trades exactly
like `aggregate_equal`. val_b of the rejected points was not computed (L-05). The 16 grid points
enter `evals/trials.csv` (N of the deflated Sharpe) at the next `make arena`. The walk-forward
strategy remembers the scores it computed earlier in the same run; a fresh object (e.g. one API
call) starts with equal weights. Rejected: weights from a static val_a fit (not online, D-17);
the label table inside the strategy (R2); a p_min grid above 0.55 (fewer than 10 round trips).
