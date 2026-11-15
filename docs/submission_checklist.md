# Submission checklist (2026-11-15)

Every requirement of `docs/inputs/NLP. Assignment 1. Build an NLP System (1) (4).pdf` checked
against the repository at commit `5a361d4` (on top of the frozen tag `freeze-1` = `47d8c2a` and the
single test run). Status: **PASS** / **PARTIAL** / **FAIL**, with the file or command that is the
evidence. Nothing in the test-period results was rerun or changed for this audit.

## 1. Assignment procedure

| # | Requirement (assignment text) | Status | Evidence |
|---|---|---|---|
| 1.1 | Define the task; extend the business requirement as long as it can be evaluated | PASS | `docs/01_business_understanding.md`: problem, users, success criteria per component with "why this metric", Must / Should / Could, the 2-day cut-line (section "2-day cut-line"), risks |
| 1.2 | Prepare the data; check existing datasets before generating synthetic ones | PASS | `docs/02_data_understanding.md`, `docs/02_data_sources.md` (52,104 bars, 317,416 news, 2,670 8-Ks, 2,423 XBRL facts, all real; D-29 rejects FNSPID, Financial PhraseBank and the Twitter set are used as public gold); `docs/04_data_quality.md`; no synthetic data anywhere |
| 1.3 | Describe the components: data-flow diagram, purpose / input / output per component, model and tool calls, one input through the whole system | PASS | `docs/02_architecture.md` (mermaid diagram; section "One input end to end"), `REPORT.md` section 3 (diagram, component table with input -> output and model / tool, the NVDA 8-K trace of 2026-08-26), `README.md` architecture diagram |
| 1.4 | Build the prototype with an agents / convention md file that steers the coding agent, CRISP-DM instructed through it | PASS | `AGENTS.md` (18 rules, CRISP-DM gates in section 2, definition of done, eval record format), `CLAUDE.md` points to it, `.githooks/commit-msg`, `tests/test_repo_rules.py` enforces R1 / R3 / R5 |
| 1.5 | Evaluate every component; distinguish high and low scores; explain why scores are not better; metric choice is judged | PASS | 15 component records in `evals/results/*.json` (plus 24 backtests and the test run), each with `metric`, `target`, `why_this_metric`, `gold_set`, `n`; `docs/03_eval_plan.md` ("Why this metric" column); `docs/evals.md` "Notes per component" with 2-3 sentences for every score below target; `REPORT.md` section 6 "Honest analysis of the low scores" |
| 1.6 | CRISP-DM reflected in the report, with how it was applied | PASS | `REPORT.md` section 2 (gate artifact per phase, the two loops back: relevance rules after the data audit, arena controls after the Gate 2 review); `docs/gate2_review.md` |
| 1.7 | Create SKILLS for the agentic development flow | PASS | `.claude/skills/{add-collector,label-helper,new-endpoint,run-evals,time-leak-check,update-report}/SKILL.md` (6 skills), listed in AGENTS.md section 8 |
| 1.8 | Not a vanilla RAG | PASS | collectors, dedup / relevance, FinBERT sentiment, event classifier, fact extraction, impact model with calibration, eight strategies, aggregator, risk manager, simulator, chat agent with read-only tools (`REPORT.md` section 3) |

## 2. Submission deliverables

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 2.1 | Code with setup and run instructions | PASS | `README.md` Quickstart (`make up`), "What a reviewer can run" (demo-only vs full data), "Development without Docker"; `Makefile` (`make help`) |
| 2.2 | High-level system diagrams in the report | PASS | `REPORT.md` section 3 mermaid; `docs/02_architecture.md`; `README.md` |
| 2.3 | Description of the business logic kept in scope | PASS | `REPORT.md` section 1 ("What was built", "Why this scope", "Out of scope") and section 8 (table of cut Should-items) |
| 2.4 | Eval results in the report | PASS | `REPORT.md` sections 6 and 7; `docs/evals.md`; `docs/05_arena.md`; `docs/06_final_run.md`; committed `evals/results/` |
| 2.5 | Report as PDF or notebook | PARTIAL | The report is `REPORT.md` (Markdown in the repo, readable on GitHub). No PDF export was produced; export with any Markdown-to-PDF tool before upload if the submission form requires a PDF |
| 2.6 | `make up` from a clean clone | PASS | `sh scripts/clean_clone_check.sh` on `5a361d4` (2026-11-15): "21 passed, 0 failed, GATE 3: PASS" (boot, FinBERT analyze, no-LLM chat with citations, unknown ticker, test-period 403, budget-0 degraded banner); warm image build, containers healthy, ports on 127.0.0.1 only |
| 2.7 | Video of at most 10 minutes: requirements, components and diagram, running demo with cases, evaluations, agentic environment | PARTIAL | Script ready: `docs/video_script.md` (9:30 plan, every input works on the demo slice, requirement checklist at the end). **The recording itself is still to be made.** |

## 3. Assessment criteria

| Criterion | Status | Evidence |
|---|---|---|
| System understanding: explain each component and trace one input (4 pts) | PASS | `REPORT.md` section 3 trace (collect -> NLP -> impact -> signals -> simulate -> explain) and `docs/02_architecture.md` "One input end to end"; video minutes 1:30-2:30 |
| Working prototype: complete the selected task (4 pts) | PASS | Demo stack on committed data (2.6); 472 offline tests green (`make test`, 2026-11-15); 14 endpoints with contract tests (`evals/results/api.json`: 32 / 32); the strategies run in the simulator and the chat cites sources |
| Evaluation for each component (4 pts) | PASS | 1.5 above. Passing: collectors 1.0, simulator 1.0, risk 1.0, relevance 0.974, intake 0.974, chat 1.0, api 1.0. Below target with written analysis: sentiment 0.516, events 0.282, facts 0.333, llm_bakeoff 0.75, impact 0.498, calibration -0.031, aggregator -3.145, arena DSR 0.0015; test run DSR 0.016 (`REPORT.md` sections 6-7) |
| Use of the AI coding assistant: instructional md files, reflection (3 pts) | PASS | `AGENTS.md`, `CLAUDE.md`, 6 skills, `agent_log.md` (one row per session with "where the agent went wrong"), `docs/decisions.md` (D-01..D-49); `REPORT.md` section 9 reflection with five logged failures and what caught them |

## 4. Repository rules (AGENTS.md) spot checks

| Rule | Status | Evidence |
|---|---|---|
| CRISP-DM gates (section 2): artifact per phase | PASS | 01, 02 (+ figs), 04 + `make collect-dev`, evals per component, `docs/evals.md`, `make up` (see 2.6) |
| R3 frozen splits, test period used once | PASS | `evals/results/TEST_FINAL.lock` (started 13:32:09 UTC, `git_sha=47d8c2a`, `status=ok`); `make test-final` refuses when the lock exists; `tests/test_test_period_lock.py`, `tests/sim/test_final.py` |
| R4 every component ships with an eval; `make eval` reproduces `docs/evals.md` | PARTIAL | Records exist for every component (1.5). `make eval` was **not rerun** in this audit (needs the full database and `make train`; the user asked for no reruns). The committed records carry the git sha of their run; the only code change after them that touches an eval is D-49 (eval_collectors, planted slices), pinned by `tests/collectors/test_report_eval.py` (6 / 6 with and without the lock). Note: the committed `collectors.json` lists 4 "test-period rows in a dev collection" quality failures: the test period had already been collected for `make features-test` (D-38) when that record was written; its metric (completeness 1.0) is unaffected |
| R5 only `src/llm.py` calls a provider; free models; budget; fallbacks | PASS | `tests/test_repo_rules.py`, `tests/test_llm_gateway.py`, `config/llm.yaml`, `docs/inputs/free_models.md` |
| R6 untrusted text, injection test per LLM component | PASS | `tests/test_injection.py` (8 texts), chat poisoned headlines 5 / 5 (`evals/results/chat.json`) |
| R11 co-occurrence wording and disclaimer everywhere | PASS | `src/schemas.py` DISCLAIMER; chat eval wording check 1.0; clean-clone check asserts "not investment advice" in `/analyze`, `/chat`, `/signals` |
| R12 offline tests | PASS | `uv run pytest`: 472 passed in 125 s (2026-11-15), no network |
| Lint | PASS | `uv run ruff check .` all checks passed; `ruff format --check .` 187 files formatted |

## 5. Hygiene

| Check | Status | Evidence |
|---|---|---|
| No AI attribution in commit messages | PARTIAL | `git log --format=%B \| grep -i -E "co-authored\|claude\|generated with"` prints **one** line: the subject `docs: add AGENTS.md rules and CLAUDE.md pointer` (commit `6d92347`), which names the file `CLAUDE.md`, not an attribution. No `Co-Authored-By`, `Claude-Session` or "Generated with" line exists in any commit body. Not rewritten: a rebase would move `freeze-1`, the lock's `git_sha` and every `git_sha` in `evals/results/` |
| No secrets in history or the working tree | PASS | Pattern scan (OpenRouter `sk-or-v1-`, AWS, GitHub tokens, `*_API_KEY=<value>`, private keys) over `git log -p --all`: 0 hits. Working tree: 1 file, `.env`, which is git-ignored and untracked (`git ls-files` lists only `.env.example`). No value was printed |
| `.env`, `.llm_budget.json`, `data/` (except `data/demo/`), `*.duckdb` untracked | PASS | `.gitignore` covers all four; `git ls-files` shows no `.env`, no `.llm_budget.json`, no `*.duckdb`, and under `data/` only the 17 files of `data/demo/` |
| Repository size | PASS | 303 tracked files, 4.1 MB tracked; `.git` 7.0 MB; `data/demo/` about 0.9 MB (parquet, 3 model files, 7 LLM cache files) |
| Test-period data not in the repository | PASS | `data/demo/manifest.json` cutoff `2026-08-31T20:00:00+00:00`; `python -m src.demo check` and `tests/test_demo.py` verify no newer row; `data/test_scores.parquet` is git-ignored |

## 6. Fixed in this audit

- `quality.run_checks` takes the test-period lock state explicitly; the collectors eval passes
  `False` for its planted slices, so `make eval` on a clone with the lock reports 6 / 6 again
  (D-49; tests parametrised over both lock states).
- `docs/06_final_run.md`: three strings inherited from the validation template corrected by hand
  (val_b line, "autumn 2026", "on the test split"); numbers unchanged, note added under the header.
- `README.md`: "What a reviewer can run" (demo-only clone vs full data; `evals/results/` holds the
  committed records); the chat curl example now uses a date inside the demo news window.
- `REPORT.md`: 40+ numbers spot-checked against `evals/results/*.json`, `data/models/selected.json`,
  `data/demo/` and `docs/02_data_sources.md` (test and validation leaderboards, CIs, costs, DSRs,
  percentiles, correlations, ablations, sentiment / events / facts / impact / calibration details,
  API latencies, Hedge grid, config hash, lock times, data totals). No mismatch found; no edit needed.

## 7. Open before upload

1. Record the video from `docs/video_script.md` (2.7).
2. Export `REPORT.md` to PDF if the submission form requires a PDF (2.5).
3. Push the repository and check that the GitHub rendering of `REPORT.md` shows the mermaid diagram.
