# 01 — Business Understanding

CRISP-DM phase 1 gate artifact (AGENTS.md section 2). Sources: the assignment and the project plan in
`docs/inputs/`. Design choices referenced as `D-nn` are in `docs/decisions.md`.

## Problem

A retail investor with a watchlist of US stocks (plus BTC and ETH) cannot read every news item and SEC
filing. They want to know, each morning before the open, what happened to their tickers, what kind of
event it was, what it tended to coincide with in price, and what a set of transparent strategies would
do about it. They also want to ask "why did X move?" and get an answer that cites the news and filings
it relies on. Finally, they want to know whether such suggestions would have made money, measured
honestly in a simulator rather than claimed.

The assignment requires a working NLP prototype built with an AI coding assistant, a quality evaluation
for every component, CRISP-DM, agent instruction files, and a demo video.

Everything the system outputs is a **simulation, not investment advice** (the sentence appears in the
UI and in every signal-bearing API response).

## Users

| User | Needs | Interface |
|---|---|---|
| Retail investor (primary) | morning brief per ticker; event and sentiment labels with sources; daily suggestion with confidence and the "who is for / who is against" breakdown; "why did X move?" chat; simulator results | Streamlit UI |
| Another script or app (secondary) | the same signals, events and text analysis as JSON | REST API `/api/v1`, Swagger at `/docs` |
| Grader / reviewer | run from a clean clone with one command; trace one input through the system; see an eval per component; see how the agent was managed | `make up`, Swagger, `docs/`, `AGENTS.md`, `agent_log.md` |

## What the system does (scenarios)

1. **Monitor**: collect daily prices (yfinance), hourly BTC/ETH candles (Binance), company news
   (Finnhub) and SEC filings (8-K, Form 4, XBRL facts) for the watchlist into DuckDB, every row stamped
   with `available_at`.
2. **Understand text**: deduplicate and filter relevant news; sentiment (−1/0/+1); event type aligned
   with 8-K item numbers; revenue and EPS extracted from earnings releases.
3. **Estimate impact**: abnormal return (stock minus SPY) 1–3 days after an event; direction model.
4. **Suggest**: an arena of strategies (baseline, technical, NLP, event, crypto) each returns a signal
   in [−1, 1] with confidence and reasons; a Hedge aggregator combines them; a risk manager can veto.
5. **Simulate**: trades at the open of day D using only information available before 09:00 ET of D;
   commissions, position limits, stop-loss, drawdown halt; leaderboard and equity curves.
6. **Explain**: a chat agent whose tools are the same read-only service functions; answers cite news
   ids and filing accession numbers; wording reports co-occurrence, never causation.

## Success criteria

Targets are guidance, not promises: a missed target with a 2–3 sentence explanation is a valid report
result (plan, stage 5). "Why this metric" is required by AGENTS.md R4.

| Component | Metric | Gold set | Target | Why this metric |
|---|---|---|---|---|
| Intake agent | accuracy of company name → ticker; profile field accuracy | known name→ticker pairs; 25 synthetic dialogues | ≥ 0.95 / ≥ 0.90 | a wrong ticker poisons every downstream step |
| Collectors | completeness, duplicate share, missing trading days | exchange calendar, dedup keys | ≥ 99 % days, 0 dups | gaps silently degrade signals |
| Relevance + dedup | precision / recall on pairs | auto pairs from the same 8-K event + ~30 hand labels | P ≥ 0.90, R ≥ 0.80 | precision protects the sentiment index from noise; recall shows what is lost |
| Sentiment | macro-F1, lexicon vs LLM | Financial PhraseBank sample (n=500) | ≥ 0.80 | classes are imbalanced; macro-F1 weights negative and neutral equally |
| Event classifier | macro-F1 | 8-K item numbers from EDGAR (automatic) + 60 hand-labelled news | ≥ 0.75 | item classes are rare and unequal; accuracy would reward "8.01 other" |
| Fact extraction | share of values within ±1 % of XBRL | EDGAR companyfacts for the same quarter (automatic) | ≥ 0.90 | the number is either right or useless; tolerance covers rounding |
| Impact estimate | directional accuracy, AUC | abnormal 1–3 day returns after events | AUC > 0.55 | AUC is threshold-free and chance is 0.5, so the lift is readable |
| Strategies | return, Sharpe, max drawdown, hit rate, trade count | buy-and-hold; 1000 random strategies | each strategy explained vs baselines | one number hides risk; the set shows return per unit of risk and luck |
| Aggregated strategy | Sharpe on validation and (once) test; deflated Sharpe; ablation; Brier of confidence | buy-and-hold, best single strategy chosen on validation, equal weights, random strategies | Sharpe > buy-and-hold and > 95 % of random strategies | deflated Sharpe accounts for every attempt; Brier checks that confidence means something |
| SMC detector / strategy (Should) | precision/recall on synthetic candles; expectancy, profit factor, parameter sensitivity | synthetic patterns; random entries with the same risk/reward; random walk | any result with controls is valid | SMC has no peer-reviewed evidence, so the controls are the result |
| Chat agent | faithfulness; numbers in the answer match DuckDB; citation validity; LLM-judge kappa vs human | ~30 questions with expected values | ≥ 0.90 faithful; kappa ≥ 0.6 | a fluent wrong answer is the main failure mode of RAG-like agents |
| LLM layer | share of responses valid against schema before / after retries; calls per day; cache hit rate | all traced calls | ≥ 98 % valid after retries | measures model reliability separately from component quality |
| API layer | contract tests pass; p95 latency of `/analyze` and `/chat` | pytest + TestClient; traces | 100 % tests; p95 ≤ 10 s | the grader's experience is the API |
| End-to-end | smoke scenarios passing | 10 scenarios through the API | 10 / 10 | the demo must not break |
| All LLM evals | planted errors caught | `evals/labels/planted_errors.jsonl` | ≥ 90 % | proves the metric can tell good from bad |

## Scope

| Priority | Item | Note |
|---|---|---|
| **Must** | pydantic schemas shared by API and LLM structured output | R13 |
| Must | collectors (yfinance, Finnhub, SEC EDGAR 8-K + XBRL), DuckDB, quality checks, `available_at` | R2, R7, D-04 |
| Must | `MarketView(as_of)` and frozen splits | R2, R3, D-05 |
| Must | intake: phrase → tickers + risk profile; company name → ticker | LLM with table fallback |
| Must | news processing: dedup, relevance, sentiment (lexicon baseline vs LLM) | D-09 |
| Must | event classifier for news and 8-K, taxonomy = 8-K items | D-10 |
| Must | fact extraction (revenue, EPS) from earnings releases, XBRL gold | |
| Must | impact estimate: abnormal return 1–3 days, direction model | D-11 |
| Must | strategies: buy-and-hold, momentum, mean reversion, news sentiment, event | long-only, R8 |
| Must | simulator: open-of-D fills, commissions, risk manager (limit, stop-loss, drawdown halt) | |
| Must | aggregation with online weights (Hedge) + confidence threshold chosen on validation | D-17 |
| Must | FastAPI layer per the endpoint table below, jobs with `job_id`, contract tests, local tracing | D-06 |
| Must | chat agent (`POST /chat`) with read-only service tools and citations | R6, R11 |
| Must | Streamlit client over the API: dashboard, arena, chat | |
| Must | evals for every component, `make eval`, `docs/evals.md`, report, video | R4 |
| **Should** | SMC strategy on BTC/ETH hourly candles with controls | smartmoneyconcepts lib |
| Should | contrarian and insider (Form 4) strategies | |
| Should | stacking meta-model with SHAP; LLM researcher (≤ 5 iterations on validation, rules as YAML) | D-17 |
| Should | market-regime detector (ADX, realised volatility), strategy × regime table | rules only |
| Should | scheduled collection (APScheduler) inside the API | |
| Should | masked-prompt experiment (names and dates replaced) for LLM components | memory vs analysis |
| Should | Langfuse tracing in addition to local JSONL | D-06 |
| **Could** | bull-vs-bear debate strategy with a judge | |
| Could | multimodal LLM marks liquidity zones on charts, IoU vs algorithm | |
| Could | streaming chat answers (SSE) | |
| Could | crypto news sentiment | |

**Out of scope**: real trading, intraday/high-frequency trading, non-US exchanges, derivatives,
authentication and multi-user API, portfolio optimisation beyond the position limits.

### API contract (prefix `/api/v1`, Swagger at `/docs`)

| Endpoint | Does |
|---|---|
| `POST /watchlist` | intake: phrase → tickers and risk profile |
| `POST /collect` | background collection → `job_id` |
| `GET /prices/{ticker}` | prices from DuckDB |
| `GET /news/{ticker}` | news with sentiment, relevance, event type |
| `GET /events/{ticker}` | 8-K filings with classification and extracted facts |
| `POST /analyze` | arbitrary text → sentiment, event, facts |
| `GET /signals/{ticker}?date=` | all strategy signals and the aggregated signal as of that date |
| `POST /backtest` | start a backtest → `job_id` |
| `GET /jobs/{id}` | job status and result |
| `GET /arena` | strategy leaderboard |
| `POST /chat` | question → answer with citations |
| `GET /health` | liveness (no prefix) |

### One input through the system (the demo trace)

1. A watchlist company files an 8-K with item 2.02 (results). `POST /collect` stores it with
   `available_at` = EDGAR acceptance time (next 06:00 ET if accepted after 17:30 ET).
2. `GET /events/{ticker}`: event classifier → "quarterly results"; fact extraction → revenue, EPS;
   XBRL later confirms them (eval).
3. `GET /news/{ticker}`: sentiment of the release and the related Finnhub news; duplicates collapsed.
4. Impact estimate gives a direction and strength for 1–3 days.
5. `GET /signals/{ticker}?date=D`: every strategy's signal as of 09:00 ET on D, the Hedge aggregate with
   confidence (e.g. "buy, 0.72"), the risk manager's verdict.
6. `POST /backtest` → `GET /jobs/{id}`: the simulator fills at the open of D and shows the trade on the
   equity curve.
7. Streamlit shows the trade and the "for / against" breakdown; `POST /chat` answers "why?" citing the
   filing and the news, in co-occurrence wording.

## Data

| Data | Source | Used for | Limits |
|---|---|---|---|
| daily OHLCV, 12 stocks + SPY | yfinance | strategies, simulator, impact | unofficial library; cache raw on disk |
| hourly BTC/ETH candles | Binance public API | SMC strategy | public rate limits |
| company news | Finnhub (free) | sentiment, events, chat | ~60 req/min, ~1 year history |
| 8-K, Form 4, XBRL companyfacts | SEC EDGAR | events, facts gold, insider strategy | User-Agent with contact, ≤ 10 req/s |
| labelled sentiment | Financial PhraseBank | sentiment eval | — |
| historical news + prices | FNSPID | training/eval of NLP components | ends 2023, not for the LLM backtest |

Splits (frozen, `config/splits.yaml`, D-05): warmup 2024-01-02..2025-11-28 (prices only);
train 2025-12-01..2026-05-29; validation 2026-06-01..2026-08-31; test 2026-09-01..2026-11-06, run once.

Watchlist (D-18): AAPL, MSFT, NVDA, JPM, GS, XOM, CVX, WMT, COST, MCD, JNJ, BA; SPY benchmark; BTC, ETH.

## Risks

| Risk | Mitigation |
|---|---|
| Look-ahead bias (future data in a decision) | `available_at` on every row, `MarketView(as_of)`, 09:00 ET cutoff, open(D) only in `src/sim`, `time-leak-check` skill, shift test |
| LLM remembers the period instead of analysing (knowledge cutoffs of free models are not stated) | test period is the most recent window; masked-prompt experiment; cheap non-LLM strategies as controls |
| Free-model rate limits and outages | hard daily budget, cache, batching, fallback model list, deterministic fallbacks, precomputed features |
| Component without an eval | definition of done in AGENTS.md section 4; `make eval` must cover every component |
| Evals that only an LLM judges | ~30 human labels and Cohen's kappa; planted errors |
| App does not start for the grader | `make setup && make up` from a clean clone; demo slice in `data/demo/` without keys |
| Logic leaks into routers or the UI | R13; `new-endpoint` skill; contract tests |
| API work eats eval time | services and evals first; API is a thin wrapper after |
| Long requests time out | collect and backtest as jobs with `job_id`; LLM cache |
| Prompt injection via news or filings | R6: delimiters, read-only tools, injection tests |
| Two writers corrupt DuckDB | R7: file lock, read-only connections elsewhere |
| Test period used twice | `make test-final` lock file committed to git |
| Similar implementation to a classmate | arena + aggregation, SEC layer with automatic gold, SMC controls, masking experiment, REST API |
| Scope creep | Must / Should / Could; Should only after Must is green; cut-line below |
| Video longer than 10 minutes | minute-by-minute script, cached demo cases |

## 2-day cut-line

If two working days remain before submission and the Must list is not green, ship exactly this and
nothing else; write the rest up as limitations.

Keep:

1. collectors for prices and Finnhub news with the demo slice; `MarketView`; frozen splits
2. sentiment with the lexicon fallback and the LLM path, Financial PhraseBank eval
3. event classifier on 8-K items (rules + LLM), automatic EDGAR gold, eval
4. strategies: buy-and-hold, momentum, news sentiment; simulator with risk manager; Hedge aggregation;
   validation run with the random-strategy control
5. API: `/health`, `/news/{ticker}`, `/signals/{ticker}`, `/analyze`, `/chat` (citations from DuckDB
   records, LLM prose optional), contract tests
6. Streamlit: one dashboard page and the chat
7. `make eval` with results for every shipped component, `docs/evals.md`, report, video

Drop (document as not done): fact extraction and the event strategy, impact model (replace by a
rule: event type → sign), mean reversion, all Should/Could items, background jobs (run collect and
backtest synchronously), openapi export beyond what FastAPI serves at `/docs`.
