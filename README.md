# MemoryBench

Here's a question nobody in AI memory is asking: what if every product in this space is solving the wrong problem?

They all compete on the same axis. Better embeddings. Smarter chunking. Fancier summarization. That's not building something new, that's incrementalism. Going from 1 to 1.1 while calling it a breakthrough.

The real problem is simpler and more damning than anyone admits. When you tell your AI "I tried both dark roast and light roast last week, and honestly the dark roast was so much better," every memory product on the market stores `User prefers dark roast coffee`. The comparison is gone. The timing. The conviction. They kept the label and threw away the signal. All of them. And they all think they're doing a good job because the label is technically correct.

I built MemoryBench because I got tired of guessing. 500 questions, 115k+ tokens of conversation, an LLM judge with ground truth answers. No vibes. A number.

<img width="3584" height="2154" alt="original" src="https://github.com/user-attachments/assets/7fe49b7e-ed0b-4861-92a5-fa5d199cfc72" />

## The Numbers

| Provider | Accuracy | Notes |
|----------|----------|-------|
| Supermemory | 85.9% | Proprietary API |
| **RAG v1.7 (ours)** | **82.8%** | Open source, runs locally |
| mem0 | 72.1% | Cloud API |
| OpenClaw QMD | 58.3% | |
| Filesystem | 54.2% | Naive baseline |

Six question types built to break things: factual recall, preference understanding, temporal reasoning, knowledge updates, multi-session synthesis, and assistant recall.

### Per-Category Breakdown (v1.7)

| Question Type | Accuracy | What It Tests |
|---|---|---|
| Single-session user facts | 97.1% | "How many bass did I catch?" |
| Single-session assistant recall | 94.6% | "What recipe did you suggest?" |
| Knowledge updates | 83.3% | "Where do I work now?" (changed jobs) |
| Multi-session synthesis | 78.2% | Connecting info across conversations |
| Temporal reasoning | 69.2% | "When did I start running?" |
| Preferences | 63.3% | "What's my favorite coffee?" |

Retrieval quality: 97.2% Hit@K, 0.912 MRR. The system finds the right information almost every time. When it gets answers wrong, it's a reasoning problem, not a retrieval problem. Most teams in this space haven't figured out that the bottleneck moved.

## The Secret

Everyone competes on the same dimension. Better embeddings. Better chunking. Better summarization. When everyone plays the same game, nobody wins big.

The question that matters isn't "how do we store memories more efficiently?" It's "how do we preserve the context that makes memories useful?" Those sound similar. They're not. The first question leads you to optimize compression. The second forces you to rethink the architecture.

Extract atomic facts from a conversation and you get perfect keyword matches. Keep the full conversation chunk alongside those facts and you get understanding. The gap shows clearly in preference questions: 63% with facts alone, 90% with chunks. Preferences carry nuance. Single-line facts can't. This isn't a tuning problem. It's a category error that every fact-extraction system makes, and none of them seem aware of it.

v1.7 does both. Facts for precision. Chunks for context. The fact finds the needle; the chunk gives you the haystack around it.

## Architecture

```
Query -> Rewrite (gpt-5-nano) -> Embed (text-embedding-3-large)
                                        |
                    +---------------------------------------+
                    |         Parallel Search               |
                    |  BM25 keywords + Vector similarity    |
                    |  Atomic fact matching                 |
                    |  Entity graph (2-hop BFS)             |
                    +-------------------+-------------------+
                                        |
                    Fact -> Parent Chunk Injection
                                        |
                    Session Boost (+0.1 for fact-matched sessions)
                                        |
                    LLM Rerank (gpt-5-nano, query-type aware)
                                        |
                    Entity + Relationship context
                                        |
                    User Profile injection
                                        |
                    Answer (GPT-5, reasoning)
```

Six retrieval signals in parallel:
1. **BM25** (0.3 weight): exact keyword matching for names, dates, specific terms
2. **Vector similarity** (0.7 weight): semantic matching for paraphrased queries
3. **Atomic facts**: line-level precision matching with parent chunk retrieval
4. **Entity graph**: 2-hop BFS traversal for multi-hop relationship questions
5. **User profile**: persistent biographical facts always injected
6. **LLM reranker**: query-type-aware scoring (temporal, preference, factual, multi-hop)

Everything runs on your machine. The only external calls go to OpenAI for embeddings and inference. No memory API subscriptions. Your data stays yours.

## Dual-Layer Memory

Instead of building one system that's mediocre at everything, build two that are each excellent at one thing. This is the part most people miss because they're too focused on making their single pipeline 5% better.

**mem0 sidecar** (Python, port 3848): fact extraction, knowledge graph, profile building. You say "I moved from Madrid to Berlin." It extracts the fact, updates the graph, replaces "lives in Madrid" with "lives in Berlin." Structured knowledge, always current.

**RAG server** (Bun, port 3847): chunk storage and hybrid search. Full conversation chunks with BM25 + vector similarity. When you need the exact context of a discussion, the original words are there. No lossy compression. No summaries. The actual conversation.

Four tools the LLM can call:
- `memory_recall`: facts and relationships from mem0 + 2-hop graph traversal. Quick lookups: preferences, personal info, who knows who.
- `memory_deep_search`: full conversation chunks from RAG. Exact quotes, detailed context, complete discussions.
- `memory_store`: explicitly save something to both layers. The LLM decides when information is worth persisting.
- `memory_forget`: wipe both layers clean.

The model picks the right tool based on the question. "What's my favorite coffee?" hits mem0. "What did we discuss about the migration last week?" hits RAG. This works better than any routing heuristic I could hardcode because the model understands intent.

```bash
# Start both servers
cd mem0-server && source .venv/bin/activate && python server.py &
bun run src/rag-server.ts &
```

```bash
# mem0 sidecar endpoints
POST /add              # extract facts from messages
GET  /search           # search facts by query
GET  /memories         # list all memories for a user
GET  /graph            # search graph relationships
GET  /graph/deep       # 2-hop BFS graph traversal
DELETE /memories/:id   # delete a memory
DELETE /memories       # delete all for a user

# RAG server endpoints
POST /ingest           # store conversation chunks
POST /search           # hybrid BM25 + vector search
DELETE /clear/:tag     # clear container data
```

## Auto-Capture

The system learns from every conversation without the user doing anything. Messages buffer up, then flush to both layers in the background when either 15 messages accumulate or 5 minutes of idle pass, whichever comes first. No blocking. No manual "save this" commands needed.

Facts get extracted and deduplicated. Conversations get chunked and indexed. The knowledge graph grows with every interaction. Over time the system knows you better than any single conversation could teach it, because it has all of them.

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
RAG_PORT=3847             # Optional (RAG server port)
RAG_CACHE_DIR=            # Optional (defaults to ./data/cache/rag)
MEM0_PORT=3848            # Optional (mem0 sidecar port)
MEM0_DATA_DIR=            # Optional (defaults to ./data)
MEM0_LLM_MODEL=           # Optional (defaults to gpt-5-nano)
MEM0_EMBEDDING_MODEL=     # Optional (defaults to text-embedding-3-small)
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Full pipeline: ingest, index, search, answer, evaluate, report |
| `compare` | Run the same benchmark across multiple providers side by side |
| `test` | Test a single question |
| `show-failures` | Debug failed questions |
| `serve` | Start the web UI |

## Where This Goes

Preferences at 63.3% is the gap worth closing. Atomic fact decomposition strips conversational nuance that preference questions need. Parent chunk injection in v1.7 partially addresses this, but the real fix is teaching the reranker to weight preference-bearing content higher and preserving comparative language during fact extraction.

Temporal reasoning hit 69.2% after wiring question dates into the answer prompt (they were silently missing in v1.6). More to squeeze out there.

The ceiling for this architecture is around 88-90%. Getting past Supermemory's 85.9% means solving preferences and temporal reasoning. The retrieval already works at 97.2%. You can't meaningfully improve 97.2%. The teams still optimizing retrieval haven't noticed the bottleneck moved to reasoning. That's the next problem, and it's a different kind of problem entirely.

## License

MIT
