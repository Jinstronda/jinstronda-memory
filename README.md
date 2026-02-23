# jinstronda-memory

I kept losing context. Every time Claude started a new session, everything from the last one was gone. My preferences, project decisions, debugging breakthroughs, all of it. I'd re-explain the same architecture three times in a day.

So I built two memory servers that run locally and learn from every conversation. Now my LLM actually remembers things.

## How it works

Everyone in this space picks a lane. Store extracted facts, or store raw chunks. I tried both independently and they each fail in different, annoying ways.

Facts are precise but lossy. "User prefers dark roast" throws away the comparison, the timing, the conviction behind the original statement. Meanwhile raw chunks preserve all that context but you can't point-query them. Ask "where does João live?" and you're searching through thousands of conversation fragments hoping the right one surfaces.

I run both. That's the whole insight.

**mem0** (Python, port 3848) does structured knowledge. Extracts facts, builds a knowledge graph, keeps a user profile current. You move cities, it updates the graph. You mention a friend, it maps the relationship. gpt-5-nano for extraction, Kuzu for the graph, Qdrant for vectors. All local, no cloud APIs beyond OpenAI.

**RAG** (Bun, port 3847) stores the raw conversations. Full chunks with hybrid BM25 + vector search. When you need the exact words from a discussion last week, they're there. Not a summary. The actual conversation.

The LLM chooses which one to hit based on the question. This works better than anything I could hardcode.

## Quick start

```bash
git clone https://github.com/Jinstronda/jinstronda-memory.git
cd jinstronda-memory

./setup.sh       # installs everything
./start.sh       # boots both servers
./stop.sh        # kills them
```

You need an `OPENAI_API_KEY` in `.env`. That's it.

```bash
cp .env.example .env
# add your key
```

## API

### mem0 (port 3848)

```bash
POST /add              # extract facts from text
GET  /search           # search facts
GET  /memories         # list all memories
GET  /graph            # search graph relationships
GET  /graph/deep       # 2-hop BFS traversal
POST /graph/dedupe     # run dedup on knowledge graph
DELETE /memories/:id   # delete one
DELETE /memories       # delete all
```

### RAG (port 3847)

```bash
POST /ingest           # store conversation chunks
POST /search           # hybrid BM25 + vector search
DELETE /clear/:tag     # clear container data
```

## The four tools

Wire this into an LLM via MCP or function calling and it gets four tools:

- `memory_recall` - quick lookups from mem0. Preferences, personal info, relationships.
- `memory_deep_search` - full conversation chunks from RAG. Exact quotes, detailed context.
- `memory_store` - save something to both layers explicitly.
- `memory_forget` - nuke everything.

## Auto-capture

You don't have to manually save anything. Messages buffer during conversations and flush to both layers in the background (15 messages or 5 minutes idle, whichever comes first). The knowledge graph grows with every interaction.

## Graph dedup

Left alone, the knowledge graph gets messy. The LLM generates `fought_draw_with`, `fights_with_draw`, `tied_with` for what is obviously the same relationship. A background task runs every 5 minutes and cleans this up: deletes garbage edges (message content stuffed into relationship names, self-references, timestamp entities), then clusters the rest by cosine similarity at 0.95 and merges duplicates. Bulk embeds 2048 names per API call so the whole thing finishes in about 45 seconds.

There's also a custom prompt injected at graph creation time that tells the LLM to stick to canonical 1-3 word names. Prevention + cleanup.

## Migration

Coming from an existing OpenClaw setup:

```bash
bun run scripts/migrate-all.ts        # workspace + github + chroma
python3 scripts/migrate-history.py     # old history.db
python3 scripts/dedupe-graph.py        # one-time graph cleanup
```

All scripts retry failures (3 rounds, exponential backoff).

## Config

| Variable | Default | What |
|----------|---------|------|
| `OPENAI_API_KEY` | required | embeddings + LLM |
| `MEM0_LLM_MODEL` | gpt-5-nano | fact extraction model |
| `MEM0_EMBEDDING_MODEL` | text-embedding-3-small | vector embeddings |
| `RAG_PORT` | 3847 | RAG server port |
| `MEM0_PORT` | 3848 | mem0 server port |
| `DATABASE_URL` | (sqlite) | PostgreSQL + pgvector for production |
| `GRAPH_DEDUP_THRESHOLD` | 0.95 | cosine similarity for edge merging |
| `GRAPH_DEDUP_INTERVAL` | 300 | seconds between scheduled dedup runs |

## Benchmarks

Tested against MemoryBench, 500 questions over 115k+ tokens of conversation with an LLM judge and ground truth answers:

| Provider | Accuracy |
|----------|----------|
| Supermemory | 85.9% |
| **This system (RAG v1.7)** | **82.8%** |
| mem0 cloud | 72.1% |
| Filesystem baseline | 54.2% |

Retrieval hits 97.2% Hit@K. The system finds the right information almost every time. When it gets answers wrong, it's a reasoning problem, not a retrieval problem. Most teams in this space haven't noticed the bottleneck moved.

## License

MIT
