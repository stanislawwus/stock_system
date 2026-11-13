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
