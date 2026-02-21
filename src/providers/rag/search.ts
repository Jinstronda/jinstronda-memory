/**
 * Hybrid Search Engine (BM25 + Vector)
 *
 * Implements the search approach used by OpenClaw's memory system and QMD:
 * - BM25 keyword search for exact term matching
 * - Vector similarity search using cosine distance
 * - Hybrid score fusion: (vector_score * 0.7) + (bm25_score * 0.3)
 *
 * All data is stored in-memory per container for fast access during benchmarking.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface Chunk {
  id: string
  content: string
  sessionId: string
  chunkIndex: number
  embedding: number[]
  date?: string
  eventDate?: string
  metadata?: Record<string, unknown>
}

export interface SearchResult {
  content: string
  score: number
  vectorScore: number
  bm25Score: number
  rerankScore?: number
  sessionId: string
  chunkIndex: number
  date?: string
  eventDate?: string
  metadata?: Record<string, unknown>
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
  "a",
  "an",
  "the",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "may",
  "might",
  "shall",
  "can",
  "to",
  "of",
  "in",
  "for",
  "on",
  "with",
  "at",
  "by",
  "from",
  "as",
  "into",
  "through",
  "during",
  "before",
  "after",
  "above",
  "below",
  "between",
  "out",
  "off",
  "over",
  "under",
  "again",
  "then",
  "once",
  "here",
  "there",
  "when",
  "where",
  "why",
  "how",
  "all",
  "each",
  "every",
  "both",
  "few",
  "more",
  "most",
  "other",
  "some",
  "such",
  "no",
  "nor",
  "not",
  "only",
  "own",
  "same",
  "so",
  "than",
  "too",
  "very",
  "just",
  "because",
  "but",
  "and",
  "or",
  "if",
  "while",
  "about",
  "up",
  "i",
  "me",
  "my",
  "we",
  "our",
  "you",
  "your",
  "he",
  "him",
  "his",
  "she",
  "her",
  "it",
  "its",
  "they",
  "them",
  "their",
  "what",
  "which",
  "who",
  "whom",
  "this",
  "that",
  "these",
  "those",
  "am",
])

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t))
}

// ─── BM25 Index ──────────────────────────────────────────────────────────────

const BM25_K1 = 1.2
const BM25_B = 0.75

interface BM25Index {
  invertedIndex: Map<string, Map<string, number>>
  docLengths: Map<string, number>
  avgDocLength: number
  totalLength: number
  docCount: number
}

function createBM25Index(): BM25Index {
  return {
    invertedIndex: new Map(),
    docLengths: new Map(),
    avgDocLength: 0,
    totalLength: 0,
    docCount: 0,
  }
}

function addToBM25Index(index: BM25Index, chunkId: string, text: string): void {
  // Guard against duplicate inserts (crash recovery)
  const oldLength = index.docLengths.get(chunkId)
  if (oldLength !== undefined) {
    index.totalLength -= oldLength
    index.docCount--
  }

  const tokens = tokenize(text)
  index.docLengths.set(chunkId, tokens.length)
  index.docCount++
  index.totalLength += tokens.length
  index.avgDocLength = index.totalLength / index.docCount

  // Build term frequency map
  const termFreqs = new Map<string, number>()
  for (const token of tokens) {
    termFreqs.set(token, (termFreqs.get(token) || 0) + 1)
  }

  // Update inverted index
  for (const [term, freq] of termFreqs) {
    if (!index.invertedIndex.has(term)) {
      index.invertedIndex.set(term, new Map())
    }
    index.invertedIndex.get(term)!.set(chunkId, freq)
  }
}

function searchBM25(index: BM25Index, query: string): Map<string, number> {
  const queryTerms = tokenize(query)
  const scores = new Map<string, number>()

  for (const term of queryTerms) {
    const postings = index.invertedIndex.get(term)
    if (!postings) continue

    const df = postings.size
    const idf = Math.log((index.docCount - df + 0.5) / (df + 0.5) + 1)

    for (const [chunkId, tf] of postings) {
      const docLength = index.docLengths.get(chunkId) || 0
      const numerator = tf * (BM25_K1 + 1)
      const denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (docLength / index.avgDocLength))
      const termScore = idf * (numerator / denominator)

      scores.set(chunkId, (scores.get(chunkId) || 0) + termScore)
    }
  }

  return scores
}

// ─── Vector Search ───────────────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0

  let dot = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB)
  if (denominator === 0) return 0

  return dot / denominator
}

// ─── Hybrid Search Engine ────────────────────────────────────────────────────

/** Weight for vector similarity in hybrid score */
const VECTOR_WEIGHT = 0.7
/** Weight for BM25 keyword score in hybrid score */
const BM25_WEIGHT = 0.3

export class HybridSearchEngine {
  private containers: Map<
    string,
    {
      chunks: Map<string, Chunk>
      bm25Index: BM25Index
    }
  > = new Map()

  private getContainer(containerTag: string) {
    if (!this.containers.has(containerTag)) {
      this.containers.set(containerTag, {
        chunks: new Map(),
        bm25Index: createBM25Index(),
      })
    }
    return this.containers.get(containerTag)!
  }

  addChunks(containerTag: string, chunks: Chunk[]): void {
    const container = this.getContainer(containerTag)

    for (const chunk of chunks) {
      container.chunks.set(chunk.id, chunk)
      addToBM25Index(container.bm25Index, chunk.id, chunk.content)
    }
  }

  search(containerTag: string, queryEmbedding: number[], query: string, limit: number): SearchResult[] {
    const container = this.containers.get(containerTag)
    if (!container || container.chunks.size === 0) return []

    // BM25 keyword scores
    const bm25Scores = searchBM25(container.bm25Index, query)

    // Vector similarity scores
    const vectorScores = new Map<string, number>()
    for (const [chunkId, chunk] of container.chunks) {
      const sim = cosineSimilarity(queryEmbedding, chunk.embedding)
      vectorScores.set(chunkId, sim)
    }

    // Normalize BM25 scores to 0-1 range
    let maxBM25 = 0
    for (const score of bm25Scores.values()) {
      if (score > maxBM25) maxBM25 = score
    }

    const normalizedBM25 = new Map<string, number>()
    for (const [chunkId, score] of bm25Scores) {
      normalizedBM25.set(chunkId, maxBM25 > 0 ? score / maxBM25 : 0)
    }

    // Compute hybrid scores
    const hybridScores: Array<{
      chunkId: string
      score: number
      vectorScore: number
      bm25Score: number
    }> = []

    for (const [chunkId] of container.chunks) {
      const vs = vectorScores.get(chunkId) || 0
      const bs = normalizedBM25.get(chunkId) || 0
      const hybrid = VECTOR_WEIGHT * vs + BM25_WEIGHT * bs

      hybridScores.push({
        chunkId,
        score: hybrid,
        vectorScore: vs,
        bm25Score: bs,
      })
    }

    // Sort by hybrid score descending
    hybridScores.sort((a, b) => b.score - a.score)

    // Return top results
    return hybridScores.slice(0, limit).map((result) => {
      const chunk = container.chunks.get(result.chunkId)!
      return {
        content: chunk.content,
        score: result.score,
        vectorScore: result.vectorScore,
        bm25Score: result.bm25Score,
        sessionId: chunk.sessionId,
        chunkIndex: chunk.chunkIndex,
        date: chunk.date,
        eventDate: chunk.eventDate,
        metadata: chunk.metadata,
      }
    })
  }

  save(containerTag: string): { chunks: Chunk[]; bm25: SerializedBM25 } | null {
    const container = this.containers.get(containerTag)
    if (!container) return null

    return {
      chunks: [...container.chunks.values()],
      bm25: serializeBM25(container.bm25Index),
    }
  }

  load(containerTag: string, data: { chunks: Chunk[]; bm25: SerializedBM25 }): void {
    const bm25Index = deserializeBM25(data.bm25)
    const chunks = new Map<string, Chunk>()
    for (const chunk of data.chunks) {
      chunks.set(chunk.id, chunk)
    }
    this.containers.set(containerTag, { chunks, bm25Index })
  }

  hasData(containerTag: string): boolean {
    return (this.containers.get(containerTag)?.chunks.size || 0) > 0
  }

  clear(containerTag: string): void {
    this.containers.delete(containerTag)
  }

  getChunksBySession(containerTag: string, sessionId: string): Chunk[] {
    const container = this.containers.get(containerTag)
    if (!container) return []
    const results: Chunk[] = []
    for (const chunk of container.chunks.values()) {
      if (chunk.sessionId === sessionId) results.push(chunk)
    }
    return results
  }

  getChunkCount(containerTag: string): number {
    return this.containers.get(containerTag)?.chunks.size || 0
  }
}

// BM25 serialization (Maps -> plain objects for JSON)

interface SerializedBM25 {
  invertedIndex: Record<string, Record<string, number>>
  docLengths: Record<string, number>
  avgDocLength: number
  totalLength: number
  docCount: number
}

function serializeBM25(index: BM25Index): SerializedBM25 {
  const invertedIndex: Record<string, Record<string, number>> = {}
  for (const [term, postings] of index.invertedIndex) {
    invertedIndex[term] = Object.fromEntries(postings)
  }
  return {
    invertedIndex,
    docLengths: Object.fromEntries(index.docLengths),
    avgDocLength: index.avgDocLength,
    totalLength: index.totalLength,
    docCount: index.docCount,
  }
}

function deserializeBM25(data: SerializedBM25): BM25Index {
  const invertedIndex = new Map<string, Map<string, number>>()
  for (const [term, postings] of Object.entries(data.invertedIndex)) {
    invertedIndex.set(term, new Map(Object.entries(postings).map(([k, v]) => [k, v as number])))
  }
  return {
    invertedIndex,
    docLengths: new Map(Object.entries(data.docLengths).map(([k, v]) => [k, v as number])),
    avgDocLength: data.avgDocLength,
    totalLength: data.totalLength,
    docCount: data.docCount,
  }
}
