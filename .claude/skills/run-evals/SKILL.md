---
name: run-evals
description: Run every component eval offline against its frozen gold set, write evals/results/<component>.json, refresh the metrics table in docs/evals.md and explain every score below target. Use after changing any component and before update-report.
---

# run-evals

Rules: AGENTS.md R3 (never on the test split), R4 (metric · gold set · target · why), section 5 (record
format). Targets per component are in `docs/01_business_understanding.md`.

## When to use

- after any change to a component, its prompt, its config, or its fallback
- before `update-report`
- when a reviewer asks "how good is X?"

## Steps

1. **Pre-flight.** `make lint && make test` green. `git status` clean enough to record a `git_sha`.
   Confirm the LLM budget left (`.llm_budget.json`) and that the LLM cache dir exists; evals should
   mostly hit the cache. If the budget is exhausted, evals must still complete through the deterministic
   fallbacks and record `model: <fallback>`.
2. **Run** `make eval` (→ `python -m evals.run_all`). Each eval script:
   - loads its gold set from `evals/labels/` (or derives it automatically and states how),
   - calls the service function directly (no HTTP),
   - restricts itself to `train`/`validation` data (never `test`),
   - includes the planted-error subset (`evals/labels/planted_errors.jsonl`) and reports how many
     were caught,
   - writes `evals/results/<component>.json` in the section-5 format, including `why_this_metric`.
3. **LLM-judge calibration.** Where an LLM scores outputs (faithfulness, relevance), the eval also
   reports Cohen's kappa against the ~30 human labels; kappa < 0.4 means the judge is not trusted and the
   score is marked as such.
4. **Schema-validity rate.** For LLM components, record the share of responses that validated before
   and after retries (the FastAPI-layer metric, target ≥ 98 %).
5. **Refresh the table.** Regenerate the metrics table in `docs/evals.md` from `evals/results/*.json`
   (component · metric · value · target · pass/fail · gold set · n · model · git_sha).
6. **Explain low scores.** For every value below target write 2–3 sentences under the table: where the
   errors are (confusion pairs, examples), why, what to change next. A missed target with an explanation
   is an acceptable result; a missed target without one is not.
7. **Commit** results and the doc together: `eval: refresh results (<components>)`.

## Output

Updated `evals/results/*.json`, updated `docs/evals.md`, a short summary in the reply with pass/fail per
component and the number of planted errors caught.
