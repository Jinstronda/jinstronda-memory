# Dual-Layer Memory System: mem0 + RAG

## Goal
Open-source supermemory alternative. Two retrieval tools (facts + chunks) the LLM chooses between. mem0 handles fact extraction/graph, RAG handles raw chunk storage. Both share a graph DB for 2-hop traversal.

## Architecture

### Ingest (auto-capture)
```
Conversation messages
  |-- mem0 (Python sidecar) --> extract facts --> Qdrant + Neo4j
  |                             ADD/UPDATE/DELETE logic
  |                             entity graph
  |                             profile
  |
  |-- RAG (Bun server) ------> chunk text --> embed --> HybridSearchEngine
                                link to session
```

### Retrieval (two tools)
```
memory_recall ------> mem0 server --> facts + relations + profile
                      fast, <200ms

memory_deep_search -> RAG server --> raw chunks + 2-hop graph
                      ~300ms, full context
```

### Infrastructure
```
OpenClaw Plugin (TS)
  |-- mem0 Server (Python, port 3848) -- facts + graph
  |-- RAG Server (Bun, port 3847) ----- chunks + hybrid search
  |-- Neo4j (shared) ------------------ entity graph for both
  |-- Qdrant (mem0 managed) ----------- fact embeddings
```

## Components

### 1. mem0 Python sidecar (new)
- FastAPI wrapper around mem0 library
- Endpoints: /add, /search, /graph, /profile, /delete, /health
- Config: gpt-5-mini extraction, text-embedding-3-small embeddings
- Qdrant for vector storage, Neo4j for graph
- containerTag maps to mem0 user_id for isolation

### 2. RAG server (existing, simplified)
- Keeps: chunk storage, hybrid search (BM25 + vector), embedding
- Keeps: 2-hop entity graph traversal (queries shared Neo4j)
- Removes: fact extraction, profile building (mem0 does this)
- Removes: atomic facts layer (mem0 facts replace this)

### 3. OpenClaw plugin (modified)
- Two tools: memory_recall (mem0), memory_deep_search (RAG)
- Auto-capture sends to both servers
- Auto-recall uses mem0 by default (facts + profile)
- Profile exclusively from mem0

### 4. Shared Neo4j graph
- mem0 writes entities + relationships
- RAG reads for 2-hop traversal on deep search
- Same DB, same schema, read/write separation

## Data Flow

### Ingest example
Input: "I moved to NYC from SF, starting at Stripe next week"

mem0:
- UPDATE fact: "User lives in NYC" (was "User lives in SF")
- ADD fact: "User works at Stripe"
- Graph: user--lives_in-->nyc, user--works_at-->stripe

RAG:
- Chunk stored with timestamp and session ID

### Query: "where does the user live?"
LLM calls memory_recall -> mem0 -> "User lives in NYC" (0.95)

### Query: "what exactly did we discuss about moving?"
LLM calls memory_deep_search -> RAG -> full conversation chunk + 2-hop graph context

## Test Strategy
- Unit tests for mem0 sidecar endpoints
- Unit tests for RAG server simplified endpoints
- Integration tests for ingest (both servers receive data)
- Integration tests for retrieval (both tools return correct results)
- Tool selection tests (LLM picks right tool for query type)
- Graph consistency tests (mem0 writes, RAG reads same data)
- UPDATE/DELETE tests (fact deduplication, contradiction handling)
- Profile accuracy tests
- Concurrent access tests
