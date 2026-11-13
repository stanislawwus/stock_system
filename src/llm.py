"""Single entry point for every LLM call (AGENTS.md rule 6).

Contract (to be implemented; no product code yet):
- models come from config/llm.yaml, which may only list IDs from docs/inputs/free_models.md
- call(prompt, schema: type[BaseModel], items: list) -> list[BaseModel]: batches several
  items per request, validates JSON with pydantic, retries with backoff, falls back down the
  model list on failure
- a hard daily call budget (config/llm.yaml -> daily_budget, counter in .llm_budget.json);
  raise BudgetExceeded once reached, callers must switch to their deterministic fallback
- on-disk cache keyed by sha256(model + prompt + schema + items)
- JSONL trace of every call in data/traces/
- untrusted text (news, filings) is wrapped in delimiters and never treated as instructions
- model output is data, never executed
"""
