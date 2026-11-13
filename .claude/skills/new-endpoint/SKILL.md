---
name: new-endpoint
description: Expose a finished service function through the FastAPI layer (/api/v1) with a pydantic schema, a thin router, a TestClient contract test, a curl example and a refreshed docs/openapi.json. Use only after the component has its unit test and eval.
---

# new-endpoint

Architecture rules: AGENTS.md R12, R13. Routers contain no logic; the service function is the unit of
truth and the eval target. The API contract (prefix `/api/v1`, endpoint list) is in
`docs/01_business_understanding.md`.

## When to use

- a component has a service function, a passing unit test and an `evals/results/<component>.json`
- the plan's endpoint table lists it (or a D-nn justifies a new one)

## Steps

1. **Check the gate.** The service function exists in `src/<package>/service.py`, its test passes,
   and its eval result exists. If not, stop and do that first (skill `run-evals`).
2. **Schema.** Add/reuse the request and response pydantic models in `src/api/schemas.py` (re-export
   the component's models; do not duplicate fields). Any response that carries a signal, suggestion or
   analysis has `disclaimer: str` with the fixed sentence "This is a simulation, not investment advice."
3. **Router.** Create `src/api/routers/<name>.py`: parse input → call the service → return the schema.
   No DuckDB access, no business logic, no LLM calls in the router. Long work (collect, backtest):
   enqueue in `src/api/jobs.py`, return `{"job_id": ...}`; status via `GET /jobs/{id}`.
4. **Register** the router in `src/api/main.py` with prefix `/api/v1` and a tag.
5. **Contract test** in `tests/api/test_<name>.py` with `fastapi.testclient.TestClient`, offline
   (monkeypatch the service or use the demo DuckDB slice): status code, response validates against the
   schema, disclaimer present, error path (404 unknown ticker, 422 bad body). Add the endpoint to the
   p95 latency table in `evals/results/api.json` if `make eval` measures it.
6. **README.** Add a `curl` example under "API examples".
7. **OpenAPI.** Export `docs/openapi.json` (`uv run python -m src.api.export_openapi`) and commit it.
8. **UI (optional, last).** Only if the component needs a screen: add a call in `src/ui/client.py`
   (httpx) and a widget; the UI never touches the database.
9. `make lint && make test`, conventional commit `feat(api): add GET /api/v1/<name>`, log line in
   `agent_log.md`.

## Output

Router, schema, contract test, README example, refreshed `docs/openapi.json`.
