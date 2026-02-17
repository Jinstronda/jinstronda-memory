# openclaw-mem0-memory

I was paying $20/month for Supermemory's OpenClaw memory plugin. It worked fine. Semantic search, auto-recall, auto-capture, user profiles. But $20/month for something that runs a vector DB and an LLM extraction step felt off.

So I forked their plugin and replaced the entire backend with [Mem0](https://github.com/mem0ai/mem0) (Apache 2.0, self-hosted). Same hooks, same tools, same OpenClaw integration. Zero cost beyond the API calls you're already paying for.

## How it works

The plugin registers hooks into OpenClaw's lifecycle:

- **before_agent_start**: queries your memory store, injects relevant context into the prompt
- **agent_end**: captures the conversation, sends it to Mem0 for fact extraction and storage

Mem0 handles the hard parts. It extracts facts from raw conversation text, deduplicates against existing memories, and maintains a vector store for semantic search.

The architecture is simple:

```
OpenClaw plugin (TypeScript)
    |
    v
Mem0 REST server (FastAPI, server/)
    |
    +-- ChromaDB (local vector store)
    +-- GPT-5-Mini or Claude (fact extraction)
    +-- text-embedding-3-small (embeddings)
```

I migrated 1141 memories from Supermemory using the included migration script. Semantic search works perfectly. The recall quality is the same or better since Mem0's extraction is solid.

## Setup

### 1. Install the plugin

```bash
# copy this repo to your openclaw plugins directory
cp -r . ~/.openclaw/plugins/openclaw-mem0-memory
cd ~/.openclaw/plugins/openclaw-mem0-memory
bun install
```

### 2. Start the Mem0 server

```bash
pip install mem0ai fastapi uvicorn chromadb aiohttp
cd server
python server.py
```

The server runs on port 8080 by default. Set `MEM0_PORT` env var to change it.

You need either `OPENAI_API_KEY` (for GPT-5-Mini extraction + embeddings) or `ANTHROPIC_API_KEY` (for Claude extraction) plus `OPENAI_API_KEY` (embeddings always use OpenAI). The server loads env vars from `~/.openclaw/workspace/.env` automatically.

### 3. Configure OpenClaw

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "openclaw-mem0-memory": {
        "enabled": true,
        "config": {
          "mem0Url": "http://localhost:8080",
          "autoRecall": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

Restart OpenClaw. Done.

## Tools and Commands

The AI gets four tools automatically:

| Tool | What it does |
|------|-------------|
| `mem0_store` | Save information to long-term memory |
| `mem0_search` | Semantic search across memories |
| `mem0_forget` | Delete a memory by query or ID |
| `mem0_profile` | View user profile (persistent facts + recent context) |

Slash commands: `/remember <text>` and `/recall <query>`.

CLI: `openclaw mem0 status`, `openclaw mem0 search <query>`, `openclaw mem0 profile`, `openclaw mem0 wipe`.

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mem0Url` | string | `http://localhost:8080` | Mem0 server URL |
| `userId` | string | `openclaw_{hostname}` | Memory namespace |
| `autoRecall` | boolean | `true` | Inject memories before every AI turn |
| `autoCapture` | boolean | `true` | Store conversations after every turn |
| `maxRecallResults` | number | `10` | Max memories injected per turn |
| `profileFrequency` | number | `50` | Full profile injection every N turns |
| `captureMode` | string | `"all"` | `"all"` filters noise, `"everything"` captures all |
| `debug` | boolean | `false` | Verbose logging |

## Migration from Supermemory

If you have existing memory files in `~/.openclaw/workspace/memory/`, run:

```bash
cd server
python migrate.py
```

This reads all `.md` files, chunks them by section, and ingests everything into Mem0 concurrently. Edit `USER_ID` in `migrate.py` to match your config.

## License

MIT
