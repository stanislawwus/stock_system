# Agent log

One line per session (AGENTS.md R10): `date · what was asked · where the agent went wrong · what changed in AGENTS.md / skills`.
This is the raw material for the "use of the AI coding assistant" reflection in the report and the video.

| Date | Asked | Where the agent went wrong | Changed in AGENTS.md / skills |
|---|---|---|---|
| 2026-11-13 | Design phase: docs/02 (mermaid + trace), docs/03 eval plan with gap_ar/drift_ar, src/schemas.py, MarketView + db guard interfaces, llm.py interface, OpenAPI stub with 11 thin routers, UI httpx client, tests first (3 MarketView tests fail by design). | First test run: forgot the `/api/v1` prefix in one contract test and left "9.01" out of the event taxonomy although D-10 listed it; a bash heredoc batch with many files failed to parse, so files were written one by one. Brief's availability rules differed from D-04 (no next-morning rule for filings): superseded by D-24 rather than mixed. | AGENTS.md R2/R8/R12 updated (D-24 rules, Signal fields, xfail convention); time-leak-check step 5 updated; decisions D-23..D-28. |
| 2026-11-13 | Read assignment + plan; set up repo: AGENTS.md (18 rules), CLAUDE.md pointer, uv/ruff/pytest skeleton, Makefile, 6 skills, docs/01, docs/decisions.md, frozen splits, configs, repo-rules tests. No product code. | Plan PDF and brief disagreed on layout (src/services, api/ at root) and on the cutoff (close of D-1 vs 09:00 ET); agent followed the brief and recorded both in decisions D-01, D-03 instead of asking. Reading the plan PDF by page range failed (no poppler); whole-file read worked. | Created AGENTS.md and all six skills; added `.RECIPEPREFIX` Makefile so Windows editors cannot break tabs; pinned splits hash in tests. |
