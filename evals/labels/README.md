# evals/labels — frozen gold sets

Each gold set is a CSV (or JSONL) with a provenance header comment and is **frozen once committed**
(AGENTS.md R4). Add new sets or new rows only through the `label-helper` skill, in their own commit.

Planned sets (file names are the contract used by eval scripts):

| File | Component | Source | Size target |
|---|---|---|---|
| `financial_phrasebank_sample.csv` | sentiment | Financial PhraseBank (agree ≥ 75%) sample | 500 |
| `relevance_pairs.csv` | relevance + dedup | auto pairs from same 8-K event + ~30 hand labels | ~150 |
| `event_items_8k.csv` | event classifier | 8-K item numbers from EDGAR index (automatic) | all filings in dev splits |
| `news_100.csv` (draft: `news_100.draft.csv`) | sentiment + event classifier (news) | 150 headlines sampled by `evals/make_news_sheet.py`, stratified by coincident 8-K item group, year, ticker; human labels, no LLM pre-fill | 150 |
| `xbrl_facts.csv` | fact extraction | EDGAR companyfacts (automatic) | all 8-K 2.02 in dev splits |
| `intake_dialogues.jsonl` | intake agent | synthetic dialogues, name → ticker pairs | 25 |
| `chat_questions.jsonl` | chat agent | ~30 questions with expected numbers/citations from DuckDB | 30 |
| `planted_errors.jsonl` | all LLM components | deliberately corrupted inputs every eval must catch | 20 |

Header convention (first lines of every file):

```
# gold_set: <name>
# created: <YYYY-MM-DD>  by: <human|auto:<script>>  llm_suggestions: <yes/no>  kappa_vs_llm: <value or n/a>
# source: <dataset or derivation>
```
