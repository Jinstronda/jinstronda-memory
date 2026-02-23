import { describe, test, expect } from "bun:test"
import { HybridSearchEngine, type Chunk } from "../providers/rag/search"
import { EntityGraph } from "../providers/rag/graph"

describe("RAG chunk-only integration", () => {
  test("ingest chunks and search with hybrid scoring", () => {
    const engine = new HybridSearchEngine()

    const baseEmb = new Array(1536).fill(0.1)
    const chunks: Chunk[] = [
      { id: "1", content: "User discussed moving from Madrid to Berlin for work", sessionId: "s1", chunkIndex: 0, embedding: baseEmb, date: "2026-02-23" },
      { id: "2", content: "User works at Spotify as a data scientist", sessionId: "s1", chunkIndex: 1, embedding: new Array(1536).fill(0.2), date: "2026-02-23" },
      { id: "3", content: "User enjoys hiking and outdoor photography", sessionId: "s2", chunkIndex: 0, embedding: new Array(1536).fill(0.3), date: "2026-02-20" },
    ]
    engine.addChunks("test_integration", chunks)

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

  test("clear removes all container data", () => {
    const engine = new HybridSearchEngine()
    const emb = new Array(1536).fill(0.1)

    engine.addChunks("clear_test", [
      { id: "1", content: "test chunk", sessionId: "s1", chunkIndex: 0, embedding: emb },
    ])
    expect(engine.hasData("clear_test")).toBe(true)

    engine.clear("clear_test")
    expect(engine.hasData("clear_test")).toBe(false)

    const results = engine.search("clear_test", emb, "test", 5)
    expect(results.length).toBe(0)
  })

  test("no profile in search results", () => {
    const engine = new HybridSearchEngine()
    const emb = new Array(1536).fill(0.1)

    engine.addChunks("no_profile", [
      { id: "1", content: "some conversation chunk", sessionId: "s1", chunkIndex: 0, embedding: emb },
    ])

    const results = engine.search("no_profile", emb, "conversation", 5)
    for (const r of results) {
      expect((r as any)._type).toBeUndefined()
    }
  })

  test("multiple sessions stored independently", () => {
    const engine = new HybridSearchEngine()
    const emb1 = new Array(1536).fill(0.1)
    const emb2 = new Array(1536).fill(0.9)

    engine.addChunks("multi_session", [
      { id: "1", content: "First session about Python programming", sessionId: "session_a", chunkIndex: 0, embedding: emb1 },
      { id: "2", content: "Second session about cooking pasta", sessionId: "session_b", chunkIndex: 0, embedding: emb2 },
    ])

    const s1 = engine.getChunksBySession("multi_session", "session_a")
    const s2 = engine.getChunksBySession("multi_session", "session_b")
    expect(s1.length).toBe(1)
    expect(s2.length).toBe(1)
    expect(s1[0].content).toContain("Python")
    expect(s2[0].content).toContain("cooking")
  })
})
