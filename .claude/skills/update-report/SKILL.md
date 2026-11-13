---
name: update-report
description: Refresh the submission report (docs/report.md) with the latest eval results, arena/backtest tables, diagrams, API contract and agent-development reflection, checking that every number matches evals/results. Use before submission and after any large change.
---

# update-report

The report structure follows the assignment and the plan (stage 7): one section per requirement, about a
page each. Numbers in the report are copied from `evals/results/*.json`, never typed by hand.

## When to use

- before submission
- after a large change (new component, re-run evals, final test run)
- after `run-evals` changed any value

## Steps

1. **Freshness.** Confirm `evals/results/*.json` are newer than the last component change
   (`git log -1 -- src/` vs `run_at`). If not, run the `run-evals` skill first.
2. **Sections** of `docs/report.md` (create from this list if missing):
   1. Task and scope: business logic kept in scope and why (from `docs/01_business_understanding.md`).
   2. CRISP-DM by phase with links to `docs/01`, `docs/02`, data checks, evals, deployment.
   3. Diagram and components: Mermaid data-flow from `docs/architecture.md`; per component purpose,
      input, output, model/tool calls; the trace of one input (8-K item 2.02 → events → news → impact →
      signals → backtest → chat).
   4. API contract: endpoint table and `docs/openapi.json` export.
   5. Data and preparation: sources, limits, `available_at` rules (D-04), splits from
      `config/splits.yaml` and why those dates (D-05).
   6. Evals: the table from `docs/evals.md`, why each metric, explanation of low scores, LLM-judge
      kappa, planted-error catch rate.
   7. Arena and aggregated strategy: per-strategy table, correlation matrix, strategy × regime,
      ablation, aggregator comparison, 1000 random strategies and deflated Sharpe, masked-prompt
      experiment, final test run (from `TEST_FINAL.lock` and `evals/results/*_test.json`).
   8. Limitations: model memory, short test period, free-API limits, no auth, single writer.
   9. Agentic development: AGENTS.md rules that mattered, skills, 2–3 entries from `agent_log.md`
      where the agent went wrong and what rule fixed it.
3. **Consistency check.** Every number in the report appears in some `evals/results/*.json` or in
   `config/`; grep each value. Every wording follows R11 (co-occurrence, not causation); the disclaimer
   appears once in the introduction.
4. **Diagrams.** Re-render Mermaid blocks if the components changed; keep the same six layers as the
   plan (sources → collectors/DuckDB → NLP → arena → aggregator/risk/sim → API → UI).
5. **Export** to PDF if required (`docs/report.pdf`) and commit: `docs: update report`.

## Output

Updated `docs/report.md` (and PDF), a bullet list in the reply of what changed and which numbers moved.
