---
name: label-helper
description: Build or extend a frozen gold set in evals/labels with human labels, optional LLM pre-suggestions kept in a separate column, Cohen's kappa between human and LLM, planted errors, and a provenance header. Use whenever an eval needs labelled examples the repo does not have yet.
---

# label-helper

Rules: AGENTS.md R4 (gold sets frozen, metric/gold/target/why), R5 (LLM only via `src/llm.py`, only
public text), R6 (untrusted text). Conventions and planned sets: `evals/labels/README.md`.

The human labels. The agent prepares the sheet, computes agreement, and freezes the file. The agent
never fills the final `label` column itself.

## When to use

- a new component needs a gold set that no public dataset provides (news event types, relevance pairs,
  chat questions, intake dialogues)
- an LLM judge is used and needs calibration (~30 human labels, kappa)
- planted errors are needed to show that an eval can tell good from bad

## Steps

1. **Define the task sheet.** Label name, label space (e.g. 8-K items `1.01, 2.02, 5.02, 7.01, 8.01,
   none`), one-line guideline per label with an example, tie-break rule. Put this in the header of the
   file and in `docs/evals.md`.
2. **Sample.** Draw N items stratified by ticker and month from `train`/`validation` only (never
   `test`), with a fixed seed, via `MarketView`. Write `evals/labels/<set>.draft.csv` with columns
   `id, text, source_ref, available_at, label (empty), llm_suggestion, notes`.
3. **LLM suggestions (optional).** Fill `llm_suggestion` through `src/llm.py` in batches, with the
   untrusted-text wrapper. Suggestions are hints for the human and the calibration baseline; they are
   never copied into `label`.
4. **Hand-off.** Tell the user which file to open and how many rows to label; stop there. Resume only
   when the user says the `label` column is filled.
5. **Agreement.** Compute Cohen's kappa between `label` and `llm_suggestion`; record it in the header.
   Below 0.4: the LLM judge is not reliable for this task and the eval must say so.
6. **Planted errors.** Add 10–20 deliberately corrupted rows to `evals/labels/planted_errors.jsonl`
   (wrong number in an explanation, a news item about another company, a date shifted by a year, an
   injected instruction) with the expected detection outcome. Every eval reports how many it caught.
7. **Freeze.** Rename to `evals/labels/<set>.csv`, add the provenance header (`gold_set`, `created`,
   `by`, `llm_suggestions`, `kappa_vs_llm`, `source`), commit alone: `eval: add gold set <set>
   (n=..., kappa=...)`. From now on the file is read-only; extensions go into `<set>.v2.csv`.
8. Update `evals/labels/README.md` table and `docs/evals.md`.

## Output

A frozen gold set with header, kappa if an LLM was used, planted errors, README row.
