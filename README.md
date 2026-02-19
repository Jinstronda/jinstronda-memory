# MemoryBench

Most AI memory systems are black boxes. You send conversations in, you get answers out, and you have no idea if the system actually remembers what matters. We built MemoryBench because we needed to know.

<img width="3584" height="2154" alt="original" src="https://github.com/user-attachments/assets/7fe49b7e-ed0b-4861-92a5-fa5d199cfc72" />

The idea is simple: take any memory provider (Supermemory, Mem0, Zep, or your own RAG pipeline), throw 500 questions at it across 115k+ tokens of conversation history, and see what sticks. Every question has a ground truth answer. An LLM judge scores each response. You get a number.

That number matters more than any demo.

## Results

We built an open-source RAG provider alongside the framework to prove you don't need a proprietary memory API to get good recall. Here's where things stand on LongMemEval:

| Provider | Accuracy |
|----------|----------|
| Supermemory | 85.9% |
| **RAG Provider (ours)** | **82.8%** |
| OpenClaw QMD | 58.3% |
| Filesystem | 54.2% |

That's 89% of the gap closed between a naive filesystem approach and Supermemory's proprietary system, using nothing but open-source components: GPT-5 for answering, GPT-5-mini for extraction and reranking, and text-embedding-3-small for embeddings.

Full per-category breakdown in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## How it works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Benchmarks │    │  Providers  │    │   Judges    │
│  (LoCoMo,   │    │ (Supermem,  │    │  (GPT-4o,   │
│  LongMem..) │    │  Mem0, Zep) │    │  Claude..)  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       └──────────────────┼──────────────────┘
                          ▼
              ┌───────────────────────┐
              │      MemoryBench      │
              └───────────┬───────────┘
                          ▼
   ┌────────┬─────────┬────────┬──────────┬────────┐
   │ Ingest │ Indexing│ Search │  Answer  │Evaluate│
   └────────┴─────────┴────────┴──────────┴────────┘
```

You pick a benchmark, a provider, and a judge. The framework handles everything else: ingesting conversation sessions, waiting for indexing, searching for relevant context, generating answers, and evaluating them against ground truth.

Every phase checkpoints independently, so when something fails at question 347 you don't start over from scratch.

## Quick Start

```bash
bun install
cp .env.example .env.local  # Add your API keys
bun run src/index.ts run -p supermemory -b locomo
```

## Configuration

```bash
# Providers (at least one)
SUPERMEMORY_API_KEY=
MEM0_API_KEY=
ZEP_API_KEY=

# Judges (at least one)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# PostgreSQL + pgvector (optional, for production)
DATABASE_URL=postgresql://user:pass@localhost:5432/memorybench
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Full pipeline: ingest, index, search, answer, evaluate, report |
| `compare` | Run the same benchmark across multiple providers side by side |
| `ingest` | Ingest benchmark data into a provider |
| `search` | Run search phase only |
| `test` | Test a single question |
| `status` | Check run progress |
| `list-questions` | Browse benchmark questions |
| `show-failures` | Debug failed questions |
| `serve` | Start the web UI |
| `help` | Show help (`help providers`, `help models`, `help benchmarks`) |

## Options

```
-p, --provider         Memory provider (supermemory, mem0, zep, rag)
-b, --benchmark        Benchmark (locomo, longmemeval, convomem)
-j, --judge            Judge model (gpt-4o, sonnet-4, gemini-2.5-flash, etc.)
-r, --run-id           Run identifier (auto-generated if omitted)
-m, --answering-model  Model for answer generation (default: gpt-4o)
-l, --limit            Limit number of questions
-q, --question-id      Specific question (for test command)
--force                Clear checkpoint and restart
```

## Examples

```bash
# Full run
bun run src/index.ts run -p mem0 -b locomo

# Resume an existing run
bun run src/index.ts run -r my-test

# Compare providers head to head
bun run src/index.ts compare -p supermemory,mem0,zep -b locomo -s 5

# Different models for answering and judging
bun run src/index.ts run -p zep -b longmemeval -j sonnet-4 -m gemini-2.5-flash

# Debug failures
bun run src/index.ts show-failures -r my-test
```

## The RAG Provider

The interesting part. We wrote a fully open-source RAG pipeline that gets within 3 points of Supermemory:

1. **Extract**: GPT-5-mini pulls structured memories, entities, and relationships from each conversation
2. **Chunk + Embed**: Memories get chunked (1600 chars, 320 overlap) and embedded with text-embedding-3-small
3. **Hybrid Search**: BM25 keyword matching + vector cosine similarity, fused at 0.7/0.3 weighting
4. **Rerank**: GPT-5-mini reranks the top 40 candidates down to 20
5. **Knowledge Graph**: Entity graph provides multi-hop relationship context
6. **Answer**: GPT-5 with chain-of-thought reasoning and explicit date arithmetic

The search index persists to disk, so you can re-search and re-answer without re-ingesting. That saves hours on iterative prompt tuning.

For production, set `DATABASE_URL` to use PostgreSQL + pgvector instead of the in-memory engine. This gives you HNSW-indexed vector search, native full-text search, and concurrent-safe storage that scales beyond what fits in memory.

## Extending

| Component | Guide |
|-----------|-------|
| Add Provider | [src/providers/README.md](src/providers/README.md) |
| Add Benchmark | [src/benchmarks/README.md](src/benchmarks/README.md) |
| Add Judge | [src/judges/README.md](src/judges/README.md) |
| Project Structure | [src/README.md](src/README.md) |

## License

MIT
