# Stock Awareness Assistant

A system for a university NLP assignment: it tracks stock news and SEC filings for a watchlist,
classifies sentiment and events, estimates price impact, runs trading strategies in an arena,
aggregates them, trades them in a simulator, and answers questions through a chat agent with citations.

Everything it outputs is a **simulation, not investment advice**.

## Run

```sh
make setup      # uv sync, enable git hooks, create .env from .env.example
# fill in .env: OPENROUTER_API_KEY, FINNHUB_API_KEY, SEC_USER_AGENT
make up         # API http://localhost:8000/docs  ·  UI http://localhost:8501
make test       # offline tests
make lint
```

Other targets: `make collect-dev`, `make features`, `make eval`, `make arena`, `make backtest`,
`make collect-test`, `make test-final` (runs the test period exactly once). `make help` lists them.

## Data

```sh
make collect-dev                                  # warmup + train + validation into data/stock.duckdb
uv run python -m src.collectors.run --periods train --sources prices,news --tickers AAPL,MSFT
uv run python -m src.collectors.report            # rewrite docs/02_data_sources.md
uv run python -m src.collectors.tickers           # verify config/universe.yaml CIKs against SEC
```

The collector is the only DuckDB writer (file lock `data/.duckdb.write.lock`). Raw responses are
cached in `data/raw/<source>/<period>/`, so a re-run spends no API calls. `SEC_USER_AGENT` must
hold a contact e-mail (`AppName name@domain`). A full dev collection takes about an hour, mostly
Finnhub's 60 calls/min. `make collect-test` runs only right before `make test-final` (D-13, D-31).

## Walking skeleton (news sentiment -> signal -> simulator -> API -> UI)

```sh
uv sync --extra finbert   # optional: FinBERT (torch, CPU is fine); without it the lexicon is used
make features             # FinBERT over train + validation news of the watchlist -> DuckDB `sentiment`
make backtest STRATEGY=finbert_v0 SPLIT=val
make up                   # then open http://localhost:8501, type a watchlist ticker
```

## API examples

```sh
curl http://localhost:8000/api/v1/health
curl "http://localhost:8000/api/v1/signals?ticker=AAPL&as_of=2026-08-14"     # cutoff 09:00 ET of that day
curl "http://localhost:8000/api/v1/signals?ticker=AAPL"                      # default: last validation day
curl -X POST http://localhost:8000/api/v1/analyze -H "Content-Type: application/json"      -d '{"text": "Boeing shares plunge after FAA grounds jets", "ticker": "BA"}'
curl "http://localhost:8000/api/v1/evals?component=backtest_finbert_v0_validation"
```

`/signals` refuses dates in the test period (403, AGENTS.md R3) and unknown tickers (404).
Every signal-bearing response carries the disclaimer. Contract: `docs/openapi.json`
(regenerate with `uv run python -m src.api.export_openapi`).

## Where to look

- `AGENTS.md` — rules for the coding agent and for humans (time rules, splits, evals, LLM budget)
- `docs/01_business_understanding.md` — problem, users, success criteria, scope, risks, cut-line
- `docs/decisions.md` — design decisions (`D-nn`)
- `docs/inputs/` — assignment, project plan, allowed free models
- `agent_log.md` — one line per agent session
- `.claude/skills/` — agent skills (time-leak-check, add-collector, new-endpoint, run-evals, label-helper, update-report)

Status: data layer, simulator, three price strategies and the walking skeleton (relevance +
dedup, FinBERT sentiment, `finbert_v0`, `/signals`, `/analyze`, `/evals`, Streamlit v0) are
implemented; events, facts, impact, aggregation and chat are next.
