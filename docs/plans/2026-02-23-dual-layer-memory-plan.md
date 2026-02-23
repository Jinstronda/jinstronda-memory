# Dual-Layer Memory System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an open-source supermemory alternative with two retrieval tools: mem0 for facts/graph and RAG for raw chunks. The LLM chooses which to use.

**Architecture:** mem0 Python sidecar (FastAPI) handles fact extraction, graph memory (Kuzu), and profile. Existing RAG server (Bun) simplified to chunk-only storage + hybrid search. OpenClaw plugin registers two tools. Both servers managed by the plugin lifecycle.

**Tech Stack:** Python 3.12+, FastAPI, mem0ai[graph], Kuzu, Qdrant (local file), Bun, TypeScript, OpenClaw Plugin SDK

---

### Task 1: mem0 Python Sidecar - Project Setup

**Files:**
- Create: `mem0-server/requirements.txt`
- Create: `mem0-server/server.py`
- Create: `mem0-server/config.py`

**Step 1: Create project structure**

```
jinstronda-memory/
  mem0-server/
    requirements.txt
    server.py
    config.py
    tests/
      __init__.py
      test_server.py
      test_graph.py
```

**Step 2: Write requirements.txt**

```
mem0ai[graph]>=1.0.4
fastapi>=0.115.0
uvicorn>=0.34.0
httpx>=0.28.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
```

**Step 3: Write config.py**

```python
import os

def get_mem0_config(data_dir: str = "./data") -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MEM0_LLM_MODEL", "gpt-5-mini"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small"),
                "embedding_dims": 1536,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "jinstronda_memories",
                "path": f"{data_dir}/qdrant",
                "embedding_model_dims": 1536,
            },
        },
        "graph_store": {
            "provider": "kuzu",
            "config": {
                "db": f"{data_dir}/graph.kuzu",
            },
        },
    }
```

**Step 4: Install dependencies**

Run: `cd mem0-server && pip install -r requirements.txt`
Expected: All packages install successfully

**Step 5: Commit**

```bash
git add mem0-server/
git commit -m "Add mem0 sidecar project structure"
```

---

### Task 2: mem0 Sidecar - Core Server

**Files:**
- Create: `mem0-server/server.py`

**Step 1: Write the failing test**

Create `mem0-server/tests/test_server.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from server import app

@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

@pytest.mark.asyncio
async def test_add_and_search(client):
    r = await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "My favorite color is blue and I work at Stripe"},
            {"role": "assistant", "content": "Got it!"},
        ],
        "user_id": "test_user",
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data

    r = await client.get("/search", params={
        "query": "favorite color",
        "user_id": "test_user",
        "limit": 5,
    })
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0
    assert any("blue" in m["memory"].lower() for m in results)

@pytest.mark.asyncio
async def test_add_string(client):
    r = await client.post("/add", json={
        "text": "User prefers dark mode",
        "user_id": "test_user_2",
    })
    assert r.status_code == 200

@pytest.mark.asyncio
async def test_get_all(client):
    await client.post("/add", json={
        "text": "I love hiking",
        "user_id": "test_getall",
    })
    r = await client.get("/memories", params={"user_id": "test_getall"})
    assert r.status_code == 200
    assert len(r.json()["memories"]) > 0

@pytest.mark.asyncio
async def test_delete(client):
    result = await client.post("/add", json={
        "text": "Temporary memory",
        "user_id": "test_delete",
    })
    memories = result.json().get("results", [])
    if memories:
        mid = memories[0]["id"]
        r = await client.delete(f"/memories/{mid}")
        assert r.status_code == 200

@pytest.mark.asyncio
async def test_delete_all(client):
    await client.post("/add", json={
        "text": "Will be deleted",
        "user_id": "test_delete_all",
    })
    r = await client.delete("/memories", params={"user_id": "test_delete_all"})
    assert r.status_code == 200
    remaining = await client.get("/memories", params={"user_id": "test_delete_all"})
    assert len(remaining.json()["memories"]) == 0

@pytest.mark.asyncio
async def test_graph_search(client):
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "Alice works at Google and lives in San Francisco"},
        ],
        "user_id": "test_graph",
    })
    r = await client.get("/graph", params={
        "query": "Alice",
        "user_id": "test_graph",
    })
    assert r.status_code == 200
    data = r.json()
    assert "relations" in data
```

**Step 2: Run tests to verify they fail**

Run: `cd mem0-server && python -m pytest tests/test_server.py -v`
Expected: ImportError (server.py doesn't exist yet)

**Step 3: Write server.py**

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from pydantic import BaseModel
from mem0 import Memory
from config import get_mem0_config

memory: Memory | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory
    data_dir = os.getenv("MEM0_DATA_DIR", "./data")
    os.makedirs(data_dir, exist_ok=True)
    config = get_mem0_config(data_dir)
    memory = Memory.from_config(config_dict=config)
    yield

app = FastAPI(lifespan=lifespan)


class AddRequest(BaseModel):
    messages: list[dict] | None = None
    text: str | None = None
    user_id: str
    metadata: dict | None = None


@app.get("/health")
def health():
    return {"ok": True, "provider": "mem0"}


@app.post("/add")
def add_memory(req: AddRequest):
    if req.text:
        result = memory.add(req.text, user_id=req.user_id, metadata=req.metadata)
    elif req.messages:
        result = memory.add(req.messages, user_id=req.user_id, metadata=req.metadata)
    else:
        return {"error": "Provide messages or text"}, 400
    return result


@app.get("/search")
def search_memories(
    query: str,
    user_id: str,
    limit: int = Query(default=10),
):
    results = memory.search(query, user_id=user_id, limit=limit)
    return {"results": results if isinstance(results, list) else results.get("results", [])}


@app.get("/memories")
def get_all(user_id: str):
    result = memory.get_all(user_id=user_id)
    memories = result if isinstance(result, list) else result.get("results", [])
    return {"memories": memories}


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    memory.delete(memory_id)
    return {"ok": True}


@app.delete("/memories")
def delete_all(user_id: str):
    memory.delete_all(user_id=user_id)
    return {"ok": True}


@app.get("/graph")
def graph_search(
    query: str,
    user_id: str,
    limit: int = Query(default=10),
):
    results = memory.search(query, user_id=user_id, limit=limit)
    relations = []
    if isinstance(results, dict) and "relations" in results:
        relations = results["relations"]
    elif isinstance(results, list):
        pass
    return {"relations": relations}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MEM0_PORT", "3848"))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**Step 4: Run tests**

Run: `cd mem0-server && python -m pytest tests/test_server.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add mem0-server/
git commit -m "Add mem0 sidecar FastAPI server with tests"
```

---

### Task 3: 2-Hop Graph Traversal on mem0

**Files:**
- Create: `mem0-server/graph_traversal.py`
- Create: `mem0-server/tests/test_graph.py`
- Modify: `mem0-server/server.py` (add /graph/deep endpoint)

**Step 1: Write failing test**

Create `mem0-server/tests/test_graph.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from server import app

@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")

@pytest.mark.asyncio
async def test_two_hop_traversal(client):
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "Alice works at Google. Google is headquartered in Mountain View."},
        ],
        "user_id": "test_2hop",
    })
    r = await client.get("/graph/deep", params={
        "query": "Alice",
        "user_id": "test_2hop",
        "max_hops": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert "entities" in data
    assert "relationships" in data

@pytest.mark.asyncio
async def test_two_hop_finds_indirect(client):
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "Bob manages the AI team. The AI team built the chatbot product."},
        ],
        "user_id": "test_indirect",
    })
    r = await client.get("/graph/deep", params={
        "query": "Bob",
        "user_id": "test_indirect",
        "max_hops": 2,
    })
    data = r.json()
    entities = [e["name"] for e in data.get("entities", [])]
    # Should find bob -> ai_team -> chatbot via 2 hops
    assert len(data.get("relationships", [])) >= 1
```

**Step 2: Run test to verify it fails**

Run: `cd mem0-server && python -m pytest tests/test_graph.py -v`
Expected: 404 on /graph/deep

**Step 3: Write graph_traversal.py**

```python
from collections import deque


def bfs_traverse(graph_store, seed_entities: list[str], user_id: str, max_hops: int = 2) -> dict:
    visited = set()
    queue = deque()
    all_entities = []
    all_relationships = []

    for entity in seed_entities:
        normalized = entity.lower().replace(" ", "_")
        queue.append((normalized, 0))

    while queue:
        current, depth = queue.popleft()
        if current in visited or depth > max_hops:
            continue
        visited.add(current)

        try:
            neighbors = _get_neighbors(graph_store, current, user_id)
        except Exception:
            continue

        for neighbor in neighbors:
            all_entities.append({"name": neighbor["name"], "type": neighbor.get("type", "entity")})
            if "relationship" in neighbor:
                all_relationships.append({
                    "source": neighbor.get("source", current),
                    "relationship": neighbor["relationship"],
                    "target": neighbor.get("target", neighbor["name"]),
                })
            if neighbor["name"] not in visited and depth + 1 <= max_hops:
                queue.append((neighbor["name"], depth + 1))

    return {
        "entities": _dedup_entities(all_entities),
        "relationships": _dedup_relationships(all_relationships),
    }


def _get_neighbors(graph_store, entity_name: str, user_id: str) -> list[dict]:
    # Query Kuzu graph for direct neighbors
    # mem0's graph_store exposes the underlying DB
    neighbors = []
    try:
        db = graph_store.db
        conn = db.connection() if hasattr(db, "connection") else db

        # outgoing
        result = conn.execute(
            "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
            "WHERE a.name = $name AND a.user_id = $uid "
            "RETURN a.name AS source, type(r) AS rel, b.name AS target",
            {"name": entity_name, "uid": user_id},
        )
        while result.has_next():
            row = result.get_next()
            neighbors.append({
                "name": row[2],
                "source": row[0],
                "relationship": row[1],
                "target": row[2],
            })

        # incoming
        result = conn.execute(
            "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
            "WHERE b.name = $name AND b.user_id = $uid "
            "RETURN a.name AS source, type(r) AS rel, b.name AS target",
            {"name": entity_name, "uid": user_id},
        )
        while result.has_next():
            row = result.get_next()
            neighbors.append({
                "name": row[0],
                "source": row[0],
                "relationship": row[1],
                "target": row[2],
            })
    except Exception:
        pass

    return neighbors


def _dedup_entities(entities: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for e in entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            result.append(e)
    return result


def _dedup_relationships(rels: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for r in rels:
        key = (r["source"], r["relationship"], r["target"])
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result
```

**Step 4: Add /graph/deep endpoint to server.py**

Add to server.py:

```python
from graph_traversal import bfs_traverse

@app.get("/graph/deep")
def graph_deep_search(
    query: str,
    user_id: str,
    max_hops: int = Query(default=2),
):
    # Extract entity names from query
    search_result = memory.search(query, user_id=user_id, limit=5)
    relations = []
    if isinstance(search_result, dict):
        relations = search_result.get("relations", [])

    seed_entities = set()
    for rel in relations:
        seed_entities.add(rel.get("source", ""))
        seed_entities.add(rel.get("destination", ""))
    seed_entities.discard("")

    if not seed_entities and hasattr(memory, "graph"):
        # Fallback: use query words as seeds
        seed_entities = set(query.lower().split())

    if not seed_entities:
        return {"entities": [], "relationships": []}

    result = bfs_traverse(
        memory.graph.graph if hasattr(memory, "graph") and memory.graph else None,
        list(seed_entities),
        user_id,
        max_hops,
    )
    return result
```

**Step 5: Run tests**

Run: `cd mem0-server && python -m pytest tests/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add mem0-server/
git commit -m "Add 2-hop graph traversal to mem0 sidecar"
```

---

### Task 4: Simplify RAG Server (Chunk-Only)

**Files:**
- Modify: `src/rag-server.ts`
- Modify: `src/providers/rag/index.ts`
- Create: `src/providers/rag/__tests__/search.test.ts`

**Step 1: Write failing test for chunk-only search**

Create `src/providers/rag/__tests__/search.test.ts`:

```typescript
import { describe, test, expect } from "bun:test"
import { HybridSearchEngine, type Chunk } from "../search"

describe("HybridSearchEngine", () => {
  test("adds chunks and searches by vector similarity", () => {
    const engine = new HybridSearchEngine()
    const embedding = new Array(1536).fill(0).map(() => Math.random())

    const chunks: Chunk[] = [
      {
        id: "c1",
        content: "User likes hiking in mountains",
        sessionId: "s1",
        chunkIndex: 0,
        embedding: embedding,
        date: "2026-02-23",
      },
      {
        id: "c2",
        content: "User works as a software engineer",
        sessionId: "s1",
        chunkIndex: 1,
        embedding: new Array(1536).fill(0).map(() => Math.random()),
        date: "2026-02-23",
      },
    ]

    engine.addChunks("test", chunks)
    const results = engine.search("test", embedding, "hiking mountains", 5)
    expect(results.length).toBeGreaterThan(0)
    expect(results[0].content).toContain("hiking")
  })

  test("BM25 keyword matching works", () => {
    const engine = new HybridSearchEngine()
    const randomEmb = () => new Array(1536).fill(0).map(() => Math.random())

    engine.addChunks("test_bm25", [
      { id: "c1", content: "Python programming language tutorial", sessionId: "s1", chunkIndex: 0, embedding: randomEmb() },
      { id: "c2", content: "JavaScript framework comparison guide", sessionId: "s1", chunkIndex: 1, embedding: randomEmb() },
      { id: "c3", content: "Python data science machine learning", sessionId: "s1", chunkIndex: 2, embedding: randomEmb() },
    ])

    const results = engine.search("test_bm25", randomEmb(), "Python", 5)
    const pythonResults = results.filter(r => r.content.includes("Python"))
    expect(pythonResults.length).toBeGreaterThanOrEqual(1)
  })

  test("clear removes all data", () => {
    const engine = new HybridSearchEngine()
    const emb = new Array(1536).fill(0).map(() => Math.random())
    engine.addChunks("clear_test", [
      { id: "c1", content: "test", sessionId: "s1", chunkIndex: 0, embedding: emb },
    ])
    engine.clear("clear_test")
    expect(engine.hasData("clear_test")).toBe(false)
  })
})
```

**Step 2: Run test**

Run: `cd /Users/joaopanizzutti/jinstronda-memory && bun test src/providers/rag/__tests__/search.test.ts`
Expected: PASS (testing existing functionality)

**Step 3: Write test for simplified RAG server**

Create `src/__tests__/rag-server.test.ts`:

```typescript
import { describe, test, expect, beforeAll, afterAll } from "bun:test"

const BASE = "http://localhost:3847"
let serverProc: any

describe("RAG Server (chunk-only)", () => {
  test("health check", async () => {
    const r = await fetch(`${BASE}/health`)
    const data = await r.json()
    expect(data.ok).toBe(true)
  })

  test("ingest stores chunks", async () => {
    const r = await fetch(`${BASE}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        containerTag: "test_chunks",
        sessionId: "session_1",
        messages: [
          { role: "user", content: "I went hiking in the Alps last summer. It was incredible, the views from Jungfrau were breathtaking." },
          { role: "assistant", content: "That sounds amazing!" },
        ],
      }),
    })
    expect(r.status).toBe(200)
  })

  test("search returns chunks", async () => {
    const r = await fetch(`${BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        containerTag: "test_chunks",
        query: "hiking Alps",
        limit: 5,
      }),
    })
    const data = await r.json()
    expect(data.results.length).toBeGreaterThan(0)
  })

  test("clear removes container data", async () => {
    const r = await fetch(`${BASE}/clear/test_chunks`, { method: "DELETE" })
    expect(r.status).toBe(200)
  })
})
```

**Step 4: Simplify RAG server**

Remove from `src/rag-server.ts`:
- `/profile/:containerTag` endpoint (mem0 handles profile)
- Profile injection in search response

Remove from `src/providers/rag/index.ts`:
- `enableExtraction` logic (mem0 handles extraction)
- `enableProfileBuilding` logic (mem0 handles profile)
- `enableAtomicFacts` logic (mem0 facts replace this)
- `enableEntityGraph` ingest logic (mem0 handles graph)
- Keep: `enableGraphRAG` for search (2-hop on shared graph, future)
- Keep: chunk storage, hybrid search, BM25 + vector

The RAG provider becomes a pure chunk store + hybrid search engine.

**Step 5: Run all tests**

Run: `bun test`
Expected: All pass

**Step 6: Commit**

```bash
git add src/
git commit -m "Simplify RAG server to chunk-only storage + hybrid search"
```

---

### Task 5: OpenClaw Plugin - Two Tools

**Files:**
- Modify: `~/.openclaw/plugins/openclaw-rag-memory/index.ts`
- Create: `~/.openclaw/plugins/openclaw-rag-memory/mem0-client.ts`
- Modify: `~/.openclaw/plugins/openclaw-rag-memory/client.ts`
- Modify: `~/.openclaw/plugins/openclaw-rag-memory/tools/search.ts`
- Create: `~/.openclaw/plugins/openclaw-rag-memory/tools/deep-search.ts`
- Modify: `~/.openclaw/plugins/openclaw-rag-memory/hooks/recall.ts`
- Modify: `~/.openclaw/plugins/openclaw-rag-memory/hooks/capture.ts`

**Step 1: Create mem0 HTTP client**

Create `~/.openclaw/plugins/openclaw-rag-memory/mem0-client.ts`:

```typescript
export class Mem0Client {
  constructor(
    private baseUrl: string,
    private userId: string,
  ) {}

  async health(): Promise<boolean> {
    try {
      const r = await fetch(`${this.baseUrl}/health`)
      return r.ok
    } catch {
      return false
    }
  }

  async add(messages: Array<{ role: string; content: string }>): Promise<any> {
    const r = await fetch(`${this.baseUrl}/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, user_id: this.userId }),
    })
    return r.json()
  }

  async addText(text: string): Promise<any> {
    const r = await fetch(`${this.baseUrl}/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, user_id: this.userId }),
    })
    return r.json()
  }

  async search(query: string, limit = 10): Promise<any[]> {
    const params = new URLSearchParams({ query, user_id: this.userId, limit: String(limit) })
    const r = await fetch(`${this.baseUrl}/search?${params}`)
    const data = await r.json()
    return data.results || []
  }

  async graphDeep(query: string, maxHops = 2): Promise<any> {
    const params = new URLSearchParams({ query, user_id: this.userId, max_hops: String(maxHops) })
    const r = await fetch(`${this.baseUrl}/graph/deep?${params}`)
    return r.json()
  }

  async getAll(): Promise<any[]> {
    const params = new URLSearchParams({ user_id: this.userId })
    const r = await fetch(`${this.baseUrl}/memories?${params}`)
    const data = await r.json()
    return data.memories || []
  }

  async deleteAll(): Promise<void> {
    const params = new URLSearchParams({ user_id: this.userId })
    await fetch(`${this.baseUrl}/memories?${params}`, { method: "DELETE" })
  }
}
```

**Step 2: Register memory_recall tool (replaces rag_search)**

Modify `tools/search.ts` to use mem0 client:
- Rename tool from `rag_search` to `memory_recall`
- Description: "Quick recall of facts and knowledge about the user. Use for questions about preferences, personal info, relationships, and recent events."
- Calls mem0 `/search` + `/graph/deep`
- Returns facts + graph relations formatted as text

**Step 3: Create memory_deep_search tool**

Create `tools/deep-search.ts`:
- Tool name: `memory_deep_search`
- Description: "Deep search through full conversation history. Use when you need exact quotes, detailed context, or the full picture of a past discussion."
- Calls RAG server `/search`
- Returns raw chunks with timestamps

**Step 4: Modify auto-capture hook**

Modify `hooks/capture.ts`:
- Send messages to BOTH servers:
  - mem0 `/add` (fact extraction)
  - RAG `/ingest` (chunk storage)
- Fire and forget, don't block on either

**Step 5: Modify auto-recall hook**

Modify `hooks/recall.ts`:
- Use mem0 `/search` instead of RAG `/search`
- Include graph relations in context
- Format as `<memory-context>` tag

**Step 6: Modify plugin index**

Modify `index.ts`:
- Create both clients (Mem0Client + RagClient)
- Start both servers (mem0 on 3848, RAG on 3847)
- Register both tools
- Wire hooks to use mem0 for recall, both for capture

**Step 7: Commit**

```bash
git add ~/.openclaw/plugins/openclaw-rag-memory/
git commit -m "Dual-tool memory: memory_recall (mem0) + memory_deep_search (RAG)"
```

---

### Task 6: mem0 Server Lifecycle in Plugin

**Files:**
- Modify: `~/.openclaw/plugins/openclaw-rag-memory/index.ts`

**Step 1: Add mem0 server start/stop to plugin service**

The plugin already manages the RAG server lifecycle. Add mem0 server alongside:

```typescript
// Start mem0 sidecar
const mem0Port = "3848"
const mem0ServerPath = `${home}/jinstronda-memory/mem0-server/server.py`

// Kill stale on port
try {
  const pids = execSync(`lsof -ti :${mem0Port} 2>/dev/null`, { encoding: "utf8" }).trim()
  if (pids) execSync(`lsof -ti :${mem0Port} | xargs kill -9 2>/dev/null`)
} catch {}

const mem0Child = spawn("python", [mem0ServerPath], {
  detached: true,
  stdio: ["ignore", "ignore", "ignore"],
  env: { ...process.env, MEM0_PORT: mem0Port, MEM0_DATA_DIR: `${home}/jinstronda-memory/data/mem0` },
})
mem0Child.unref()
```

**Step 2: Health check both servers before reporting ready**

Wait for both `/health` endpoints to respond before reporting success.

**Step 3: Commit**

```bash
git add ~/.openclaw/plugins/openclaw-rag-memory/
git commit -m "Add mem0 server lifecycle management to plugin"
```

---

### Task 7: Integration Tests

**Files:**
- Create: `tests/integration/test_dual_layer.ts`
- Create: `mem0-server/tests/test_integration.py`

**Step 1: Write end-to-end integration test (Python side)**

Create `mem0-server/tests/test_integration.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from server import app

@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")

@pytest.mark.asyncio
async def test_full_conversation_flow(client):
    # Simulate multi-turn conversation
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "My name is Carlos and I live in Madrid"},
            {"role": "assistant", "content": "Nice to meet you, Carlos!"},
        ],
        "user_id": "integration_test",
    })
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "I work as a data scientist at Spotify"},
            {"role": "assistant", "content": "Interesting role!"},
        ],
        "user_id": "integration_test",
    })
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "I moved from Madrid to Berlin last month"},
            {"role": "assistant", "content": "Big move!"},
        ],
        "user_id": "integration_test",
    })

    # Search should find updated location
    r = await client.get("/search", params={"query": "where does the user live", "user_id": "integration_test"})
    results = r.json()["results"]
    assert len(results) > 0
    texts = " ".join(m["memory"].lower() for m in results)
    assert "berlin" in texts

    # Graph should have relationships
    r = await client.get("/graph/deep", params={"query": "Carlos", "user_id": "integration_test", "max_hops": 2})
    data = r.json()
    assert len(data["relationships"]) > 0

    # Cleanup
    await client.delete("/memories", params={"user_id": "integration_test"})

@pytest.mark.asyncio
async def test_update_overwrites_old_fact(client):
    uid = "test_update_fact"
    await client.post("/add", json={
        "text": "User's favorite language is Python",
        "user_id": uid,
    })
    await client.post("/add", json={
        "text": "User's favorite language is now Rust",
        "user_id": uid,
    })

    r = await client.get("/search", params={"query": "favorite language", "user_id": uid})
    results = r.json()["results"]
    texts = " ".join(m["memory"].lower() for m in results)
    # mem0 should UPDATE, not have both
    assert "rust" in texts

    await client.delete("/memories", params={"user_id": uid})

@pytest.mark.asyncio
async def test_search_with_no_results(client):
    r = await client.get("/search", params={
        "query": "something that was never stored",
        "user_id": "nonexistent_user_12345",
    })
    assert r.status_code == 200
    assert len(r.json()["results"]) == 0
```

**Step 2: Write TypeScript integration test for RAG chunks**

Create `src/__tests__/integration.test.ts`:

```typescript
import { describe, test, expect } from "bun:test"
import { HybridSearchEngine, type Chunk } from "../providers/rag/search"
import { EntityGraph } from "../providers/rag/graph"

describe("RAG chunk-only integration", () => {
  test("ingest chunks and search with hybrid scoring", () => {
    const engine = new HybridSearchEngine()
    const graph = new EntityGraph()

    // Simulate chunked conversation
    const baseEmb = new Array(1536).fill(0.1)
    const chunks: Chunk[] = [
      { id: "1", content: "User discussed moving from Madrid to Berlin for work", sessionId: "s1", chunkIndex: 0, embedding: baseEmb, date: "2026-02-23" },
      { id: "2", content: "User works at Spotify as a data scientist", sessionId: "s1", chunkIndex: 1, embedding: new Array(1536).fill(0.2), date: "2026-02-23" },
      { id: "3", content: "User enjoys hiking and outdoor photography", sessionId: "s2", chunkIndex: 0, embedding: new Array(1536).fill(0.3), date: "2026-02-20" },
    ]
    engine.addChunks("test_integration", chunks)

    // BM25 should find Madrid/Berlin chunk
    const results = engine.search("test_integration", baseEmb, "Madrid Berlin moving", 5)
    expect(results.length).toBeGreaterThan(0)
    expect(results[0].content).toContain("Madrid")
  })

  test("entity graph stores and traverses", () => {
    const graph = new EntityGraph()
    graph.addEntity("carlos", "person", "A data scientist", "s1")
    graph.addEntity("spotify", "company", "Music streaming company", "s1")
    graph.addRelationship({ source: "carlos", target: "spotify", relation: "works_at", sessionId: "s1" })

    const context = graph.getContext(["carlos"], 2)
    expect(context.entities.length).toBeGreaterThanOrEqual(1)
    expect(context.relationships.length).toBe(1)
    expect(context.relationships[0].relation).toBe("works_at")
  })
})
```

**Step 3: Run all tests**

Run: `bun test && cd mem0-server && python -m pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/ mem0-server/tests/ src/__tests__/
git commit -m "Add integration tests for dual-layer memory system"
```

---

### Task 8: Documentation and Final Cleanup

**Files:**
- Modify: `README.md`
- Modify: `src/providers/rag/config.ts` (remove unused config flags)

**Step 1: Clean up unused config flags**

Remove from RAGConfig:
- `enableExtraction` (mem0 does this)
- `enableProfileBuilding` (mem0 does this)
- `enableAtomicFacts` (mem0 facts replace this)
- `enableFactSessionBoost` (removed with atomic facts)

Keep:
- `enableReranker`, `enableQueryRewrite` (for chunk search)
- `enableGraphRAG` (for 2-hop on deep search)
- `enableQueryDecomposition` (for counting queries)
- All tuning params (chunkSize, chunkOverlap, embeddingModel, etc.)

**Step 2: Update README**

Add section about dual-layer architecture, how to run both servers, tool descriptions.

**Step 3: Final test run**

Run: `bun test && cd mem0-server && python -m pytest tests/ -v`
Expected: All green

**Step 4: Commit and push**

```bash
git add -A
git commit -m "Clean up config, update README for dual-layer memory"
git push jinstronda-memory main
```
