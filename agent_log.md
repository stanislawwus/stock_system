# Agent log

One line per session (AGENTS.md R10): `date · what was asked · where the agent went wrong · what changed in AGENTS.md / skills`.
This is the raw material for the "use of the AI coding assistant" reflection in the report and the video.

| Date | Asked | Where the agent went wrong | Changed in AGENTS.md / skills |
|---|---|---|---|
| 2026-11-13 | Read assignment + plan; set up repo: AGENTS.md (18 rules), CLAUDE.md pointer, uv/ruff/pytest skeleton, Makefile, 6 skills, docs/01, docs/decisions.md, frozen splits, configs, repo-rules tests. No product code. | Plan PDF and brief disagreed on layout (src/services, api/ at root) and on the cutoff (close of D-1 vs 09:00 ET); agent followed the brief and recorded both in decisions D-01, D-03 instead of asking. Reading the plan PDF by page range failed (no poppler); whole-file read worked. | Created AGENTS.md and all six skills; added `.RECIPEPREFIX` Makefile so Windows editors cannot break tabs; pinned splits hash in tests. |
