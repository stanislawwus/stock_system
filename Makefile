# Stock Awareness Assistant — task runner. GNU make >= 3.82 (uses .RECIPEPREFIX, no tabs).
.RECIPEPREFIX = >
SHELL := /bin/sh
.DEFAULT_GOAL := help

export PYTHONPATH := .
RUN := uv run
API_PORT ?= 8000
UI_PORT ?= 8501
FINAL_LOCK := evals/results/TEST_FINAL.lock

.PHONY: eda help setup up api ui test lint fmt eval features collect-dev collect-test backtest arena test-final

help: ## list targets
> @grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /  -  /'

setup: ## install deps, enable git hooks, create .env from the example
> uv sync
> git config core.hooksPath .githooks
> @test -f .env || cp .env.example .env
> @mkdir -p data/raw data/cache/llm data/traces evals/results
> @echo "setup done. Fill in .env (OPENROUTER_API_KEY, FINNHUB_API_KEY, SEC_USER_AGENT)."

up: ## run API (:$(API_PORT), Swagger at /docs) and Streamlit UI (:$(UI_PORT)) together
> @sh -c '$(RUN) uvicorn src.api.main:app --host 0.0.0.0 --port $(API_PORT) & api=$$!; \
>   trap "kill $$api 2>/dev/null" EXIT INT TERM; \
>   $(RUN) streamlit run src/ui/app.py --server.port $(UI_PORT) --server.headless true'

api: ## run only the API with reload
> $(RUN) uvicorn src.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

ui: ## run only the Streamlit client
> $(RUN) streamlit run src/ui/app.py --server.port $(UI_PORT)

test: ## offline unit + contract tests
> $(RUN) pytest

lint: ## ruff check + format check
> $(RUN) ruff check .
> $(RUN) ruff format --check .

fmt: ## apply ruff formatting and safe fixes
> $(RUN) ruff check --fix .
> $(RUN) ruff format .

collect-dev: ## collect warmup + train + validation periods into DuckDB (single writer)
> $(RUN) python -m src.collectors.run --periods warmup,train,validation

collect-test: ## collect the test period only; run right before test-final (decisions D-13)
> $(RUN) python -m src.collectors.run --periods test

eda: ## figures into docs/figs and docs/02_data_understanding.md, data-quality audit docs/04
> $(RUN) python -m src.eda.quality
> $(RUN) python -m src.eda.run

features: ## precompute NLP features (sentiment, events, facts, impact) for dev splits
> $(RUN) python -m src.nlp.features --splits train,validation

eval: ## run every component eval, write evals/results/*.json and the table in docs/evals.md
> $(RUN) python -m evals.run_all

arena: ## run all strategies on the validation split and write the leaderboard
> $(RUN) python -m src.strategies.arena --split validation

STRATEGY ?= all
SPLIT ?= validation
backtest: ## simulate STRATEGY=<all|buy_and_hold|momentum|mean_reversion> on SPLIT=<train|val>
> $(RUN) python -m src.sim.run --strategy $(STRATEGY) --split $(SPLIT)

test-final: ## the ONE run on the test period; refuses to run twice (AGENTS.md R3)
> @if [ -f "$(FINAL_LOCK)" ]; then \
>   echo "ERROR: $(FINAL_LOCK) exists - the test period has already been used once."; exit 1; fi
> $(RUN) python -m src.collectors.run --periods test
> $(RUN) python -m src.nlp.features --splits test
> $(RUN) python -m src.strategies.arena --split test --final
> $(RUN) python -m src.sim.run --split test --final
> @printf 'run_at=%s\ngit_sha=%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$(git rev-parse HEAD)" > "$(FINAL_LOCK)"
> @echo "Test period used. Commit $(FINAL_LOCK) now: git add $(FINAL_LOCK) evals/results && git commit -m 'eval: final test run'"
