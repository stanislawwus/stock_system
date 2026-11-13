# OpenRouter free models (snapshot from https://openrouter.ai/collections/free-models, 2026-11-13)

Source: the collection page, ranked by 7-day token usage. "Not stated" means the page gives no detail. Knowledge cutoffs are NOT stated for any model. Free models may log prompts: send only public data.

## Allowed for src/llm.py (priority order)

| Priority | Model ID | Context | Params | Stated capabilities | Role |
|---|---|---|---|---|---|
| 1 | `cohere/north-mini-code:free` | 256K (64K out) | 3B active / 30B MoE | tool use, JSON schema output, interleaved reasoning | primary for fact extraction (JSON) |
| 2 | `nvidia/nemotron-3-ultra-550b-a55b:free` | 1M | 55B active / 550B MoE | multi-step reasoning, planning, agentic | quality candidate: facts, chat rephrasing |
| 3 | `thinkingmachines/inkling-2-small:free` | ~1M | 17B active / 594B MoE | tool calling, variable reasoning effort | fallback 1 |
| 4 | `liquid/lfm-3-1.7b:free` | 262K | 1.7B | tool calling, structured outputs | fast cheap fallback (weak accuracy) |
| 5 | `liquid/lfm-2.5-2.6b:free` | 66K | 2.6B | data extraction, RAG | fallback 2 for short snippets |
| 6 | `apodex/apodex-1.1-mini:free` | 262K | not stated | reasoning-first, research, tools | optional |

## Do NOT use

| Model ID | Why |
|---|---|
| `parallel/decisions-fast:free` | Web-grounded answers: breaks the as_of time rules (leaks the future); not a text-generation model |
| `respan/span-01-lite` | Conversation-span scoring; no `:free` suffix shown |
| `nvidia/llama-nemotron-rerank-vl-1b-v2:free` | Reranker, not a generator |
| `nvidia/nemotron-3.5-super-vl:free`, `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | Multimodal, not needed |
| `recursive/flint-v1.0:free`, `particle-ai/particle-1:free`, `baidu/ernie-5.5-flash-preview:free`, `dots-studio/dots-3-note-preview:free`, `nvidia/nemotron-3.5-lightning:free` | Coding/agentic focus, no stated JSON or extraction support; keep as last-resort fallbacks only |

## Rules

- All models are rate-limited; a call budget is enforced in src/llm.py.
- Always validate output with pydantic; never trust a model to follow the JSON format.
- Preview models can disappear: the fallback list lives in config/llm.yaml, not in code.
