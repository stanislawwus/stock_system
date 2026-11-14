# Stock Awareness Assistant — task runner. GNU make >= 3.82 (uses .RECIPEPREFIX, no tabs).
.RECIPEPREFIX = >
SHELL := /bin/sh
.DEFAULT_GOAL := help

export PYTHONPATH := .
RUN := uv run
# Servers bind to loopback only: nothing here is meant to be reachable from a network.
API_PORT ?= 8000
UI_PORT ?= 8501
HOST ?= 127.0.0.1
FINAL_LOCK := evals/results/TEST_FINAL.lock

.PHONY: eda help setup up down up-local demo demo-build api ui test lint fmt eval features train need-train features-test collect-dev collect-test backtest arena test-final eval-chat eval-api
TRAIN_ARTIFACTS := data/models/impact.joblib data/models/event_items.json data/models/selected.json
STACKED := event impact aggregate_equal aggregate_hedge

help: ## list targets
> @grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /  -  /'

setup: ## install deps, enable git hooks, create .env from the example
> uv sync
> git config core.hooksPath .githooks
> @test -f .env || cp .env.example .env
> @mkdir -p data/raw data/cache/llm data/traces evals/results
> @echo "setup done. Fill in .env (OPENROUTER_API_KEY, FINNHUB_API_KEY, SEC_USER_AGENT)."

up: ## build and start API (127.0.0.1:8000) and UI (127.0.0.1:8501) in Docker on the demo data (D-48)
> docker compose up --build -d --wait
> @echo "API  http://127.0.0.1:8000/docs   UI  http://127.0.0.1:8501   (make down to stop)"

down: ## stop and remove the Docker stack
> docker compose down

up-local: ## run API and Streamlit without Docker (needs `make setup`, `make demo` or a full database)
> @sh -c '$(RUN) uvicorn src.api.main:app --host $(HOST) --port $(API_PORT) & api=$$!; \
>   trap "kill $$api 2>/dev/null" EXIT INT TERM; \
>   $(RUN) streamlit run src/ui/app.py --server.address $(HOST) --server.port $(UI_PORT) --server.headless true'

demo: ## create data/stock.duckdb, data/models and the LLM cache from the committed data/demo slice
> $(RUN) python -m src.demo load

demo-build: ## rewrite data/demo from the full dev database (maintainers; see README)
> $(RUN) python -m src.demo build

api: ## run only the API with reload
> $(RUN) uvicorn src.api.main:app --host $(HOST) --port $(API_PORT) --reload

ui: ## run only the Streamlit client
> $(RUN) streamlit run src/ui/app.py --server.address $(HOST) --server.port $(UI_PORT)

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

train: ## impact features (cached parquet), models on train, calibrator on val_a, 8-K item table, hold/threshold + Hedge eta/p_min selection on val_a (D-40, D-44)
> $(RUN) python -m src.sim.impact_train
> $(RUN) python -m src.sim.select
> $(RUN) python -m src.sim.select_hedge

need-train: ## fail early when the `make train` artifacts (git-ignored, D-41) are missing
> @for f in $(TRAIN_ARTIFACTS); do test -f "$$f" || { \
>   echo "ERROR: $$f is missing - run 'make train' first (data/models is not committed, D-41)"; exit 1; }; done

features-test: ## FinBERT inference on test news -> data/test_scores.parquet, resumable (D-38)
> $(RUN) python -m src.nlp.features_test

eval: need-train ## run every component eval, write evals/results/*.json and the table in docs/evals.md
> $(RUN) python -m evals.run_all

eval-chat: ## chat agent eval: code checks only, 0 LLM calls -> evals/results/chat.json (D-46)
> $(RUN) python -m evals.eval_chat

eval-api: ## API eval: contract tests, LLM schema validity from saved traces, latency; no live LLM call -> evals/results/api.json (D-48)
> $(RUN) python -m evals.eval_api

arena: need-train ## run all strategies on the validation split and write the leaderboard
> $(RUN) python -m src.strategies.arena --split validation

STRATEGY ?= all
SPLIT ?= validation
backtest: ## simulate STRATEGY=<all|buy_and_hold|momentum|mean_reversion|finbert_v0|event|impact|aggregate_equal> on SPLIT=<train|val>
> @case " all $(STACKED) " in *" $(STRATEGY) "*) $(MAKE) --no-print-directory need-train ;; esac
> $(RUN) python -m src.sim.run --strategy $(STRATEGY) --split $(SPLIT)

test-final: need-train ## the ONE run on the test period; lock written at its start (R3, L-01, D-43)
> @if [ -f "$(FINAL_LOCK)" ]; then echo "ERROR: $(FINAL_LOCK) exists - the test period has already been used once."; exit 1; fi
> $(RUN) python -m src.collectors.run --periods test
> $(RUN) python -m src.nlp.features_test
> $(RUN) python -m src.strategies.arena --final
> @echo "Test period used. Commit it now: git add $(FINAL_LOCK) evals/results docs && git commit -m 'eval: final test run'"
