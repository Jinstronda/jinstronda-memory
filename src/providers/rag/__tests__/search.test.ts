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

  test("save and load round-trips correctly", () => {
    const engine = new HybridSearchEngine()
    const emb = new Array(1536).fill(0).map(() => Math.random())
    engine.addChunks("save_test", [
      { id: "c1", content: "Save and load test content", sessionId: "s1", chunkIndex: 0, embedding: emb, date: "2026-02-23" },
    ])

    const saved = engine.save("save_test")
    expect(saved).not.toBeNull()

    const engine2 = new HybridSearchEngine()
    engine2.load("save_test", saved!)

    expect(engine2.hasData("save_test")).toBe(true)
    expect(engine2.getChunkCount("save_test")).toBe(1)

    const results = engine2.search("save_test", emb, "save load", 5)
    expect(results.length).toBe(1)
    expect(results[0].content).toContain("Save and load")
  })

  test("search returns empty for nonexistent container", () => {
    const engine = new HybridSearchEngine()
    const emb = new Array(1536).fill(0).map(() => Math.random())
    const results = engine.search("nonexistent", emb, "anything", 5)
    expect(results.length).toBe(0)
  })

  test("hybrid scoring combines vector and BM25", () => {
    const engine = new HybridSearchEngine()

    const targetEmb = new Array(1536).fill(0.5)
    const otherEmb = new Array(1536).fill(-0.5)

    engine.addChunks("hybrid", [
      { id: "c1", content: "hiking mountains trails outdoors", sessionId: "s1", chunkIndex: 0, embedding: targetEmb },
      { id: "c2", content: "cooking recipes kitchen food", sessionId: "s1", chunkIndex: 1, embedding: otherEmb },
    ])

    const results = engine.search("hybrid", targetEmb, "hiking mountains", 5)
    expect(results[0].content).toContain("hiking")
    expect(results[0].score).toBeGreaterThan(results[1].score)
  })

  test("getChunksBySession returns correct chunks", () => {
    const engine = new HybridSearchEngine()
    const emb = () => new Array(1536).fill(0).map(() => Math.random())

    engine.addChunks("session_test", [
      { id: "c1", content: "session 1 chunk", sessionId: "s1", chunkIndex: 0, embedding: emb() },
      { id: "c2", content: "session 2 chunk", sessionId: "s2", chunkIndex: 0, embedding: emb() },
      { id: "c3", content: "session 1 another chunk", sessionId: "s1", chunkIndex: 1, embedding: emb() },
    ])

    const s1Chunks = engine.getChunksBySession("session_test", "s1")
    expect(s1Chunks.length).toBe(2)
    expect(s1Chunks.every(c => c.sessionId === "s1")).toBe(true)
  })
})
