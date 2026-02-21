# MemoryBench

Every AI memory product tells you the same story: "We store your conversations and retrieve them when needed." They're all lying by omission. They store summaries. Extracted facts. Compressed representations. The original context, the thing that actually matters, gets thrown away.

This is the fundamental mistake. When you told your AI assistant "I tried both dark roast and light roast last week, and honestly the dark roast was so much better," mem0 stores `User prefers dark roast coffee`. The comparison is gone. The timing is gone. The conviction is gone. You've lost the signal and kept the label.

We built MemoryBench to prove this matters, and to build something better.

<img width="3584" height="2154" alt="original" src="https://github.com/user-attachments/assets/7fe49b7e-ed0b-4861-92a5-fa5d199cfc72" />

## The Numbers

500 questions. 115k+ tokens of conversation history. Six question types designed to break memory systems: factual recall, preference understanding, temporal reasoning, knowledge updates, multi-session synthesis, and assistant recall. An LLM judge with ground truth answers. No vibes. Just a score.

| Provider | Accuracy | Notes |
|----------|----------|-------|
| Supermemory | 85.9% | Proprietary API |
| **RAG v1.7 (ours)** | **82.8%** | Open source, runs locally |
| mem0 | 72.1% | Cloud API |
| OpenClaw QMD | 58.3% | |
| Filesystem | 54.2% | Naive baseline |

### Per-Category Breakdown (v1.7)

| Question Type | Accuracy | What It Tests |
|---|---|---|
| Single-session user facts | 97.1% | "How many bass did I catch?" |
| Single-session assistant recall | 94.6% | "What recipe did you suggest?" |
| Knowledge updates | 83.3% | "Where do I work now?" (changed jobs) |
| Multi-session synthesis | 78.2% | Connecting info across conversations |
| Temporal reasoning | 69.2% | "When did I start running?" |
| Preferences | 63.3% | "What's my favorite coffee?" |

The retrieval quality is 97.2% Hit@K with 0.912 MRR. The right information is almost always found. The remaining errors are reasoning failures, not retrieval failures. That's a fundamentally different problem to solve.

## Why This Exists

The memory market is full of products competing on the same dimension: better embeddings, better chunking, better summarization. They're all playing the same game. The game is wrong.

The real question isn't "how do we store memories more efficiently?" It's "how do we preserve the context that makes memories useful?"

When I extract atomic facts from a conversation, I get perfect keyword matches. When I keep the full conversation chunk alongside those facts, I get understanding. The difference shows up in preference questions (63% vs 90% in early versions) because preferences require nuance that single-line facts can't capture.

v1.7 does both. Facts for precision matching, parent chunks for context. The fact finds the needle; the chunk gives you the haystack around it.

## The Architecture

```
Query → Rewrite (gpt-5-nano) → Embed (text-embedding-3-large)
                                        ↓
                    ┌───────────────────────────────────────┐
                    │         Parallel Search               │
                    │  BM25 keywords + Vector similarity    │
                    │  Atomic fact matching                 │
                    │  Entity graph (2-hop BFS)             │
                    └───────────────────┬───────────────────┘
                                        ↓
                    Fact → Parent Chunk Injection
                                        ↓
                    Session Boost (+0.1 for fact-matched sessions)
                                        ↓
                    LLM Rerank (gpt-5-nano, query-type aware)
                                        ↓
                    Entity + Relationship context
                                        ↓
                    User Profile injection
                                        ↓
                    Answer (GPT-5, reasoning)
```

Six retrieval signals fused together:
1. **BM25** (0.3 weight): exact keyword matching for names, dates, specific terms
2. **Vector similarity** (0.7 weight): semantic matching for paraphrased queries
3. **Atomic facts**: line-level precision matching with parent chunk retrieval
4. **Entity graph**: 2-hop BFS traversal for multi-hop relationship questions
5. **User profile**: persistent biographical facts always injected
6. **LLM reranker**: query-type-aware scoring (temporal, preference, factual, multi-hop)

Everything runs locally. The only external calls are to OpenAI for embeddings and LLM inference. No memory API subscriptions. No data leaving your machine except to the model provider.

## Quick Start

```bash
bun install
cp .env.example .env.local  # Add OPENAI_API_KEY
bun run src/index.ts run -p rag -b longmemeval -j gpt-4o -r my-run
```

## MCP Server (Claude Code Integration)

The RAG system ships as an MCP server. Claude Code gets long-term memory across sessions with per-repo isolation.

```bash
# Start the RAG server
bun run src/rag-server.ts

# MCP server registers as rag-memory in Claude Code settings
# Tools: memory_search, memory_store, memory_use_repo, memory_profile, memory_list_repos
```

Two memory layers:
- **Personal** (`jinstronda`): user bio, preferences, opinions, relationships. Always searched.
- **Per-repo** (`repo-<name>`): project-specific decisions, architecture, debugging findings. Isolated per project.

Search queries both. Every session starts by loading context. Breakthroughs get stored automatically. Context survives compaction.

## Configuration

```bash
OPENAI_API_KEY=           # Required (embeddings + LLM)
DATABASE_URL=             # Optional (PostgreSQL + pgvector for production)
RAG_PORT=3847             # Optional (server port)
RAG_CACHE_DIR=            # Optional (defaults to ./data/cache/rag)
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Full pipeline: ingest, index, search, answer, evaluate, report |
| `compare` | Run the same benchmark across multiple providers side by side |
| `test` | Test a single question |
| `show-failures` | Debug failed questions |
| `serve` | Start the web UI |

## What's Next

Preferences at 63.3% is the gap. The atomic facts decomposition strips conversational nuance that preference questions need. The parent chunk injection in v1.7 partially addresses this, but the real fix is teaching the reranker to weight preference-bearing content higher and preserving comparative language in fact extraction.

Temporal reasoning at 69.2% improved after wiring question dates into the answer prompt (they were silently missing in v1.6). Full evaluation of the date fix is pending.

The ceiling for this architecture is probably 88-90%. Getting past Supermemory's 85.9% requires solving the preference and temporal problems. The retrieval is already there. The reasoning isn't.

## License

MIT
