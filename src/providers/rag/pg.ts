import pg from "pg"
import pgvector from "pgvector/pg"
import type { Chunk, SearchResult } from "./search"
import type { RelationshipEdge, GraphSearchResult } from "./graph"

const { Pool } = pg

const VECTOR_WEIGHT = 0.7
const BM25_WEIGHT = 0.3
const MAX_GRAPH_ENTITIES = 10
const MAX_GRAPH_RELATIONSHIPS = 20

export class PgStore {
  private pool: InstanceType<typeof Pool>

  constructor(connectionString: string) {
    this.pool = new Pool({
      connectionString,
      max: 20,
      idleTimeoutMillis: 30_000,
      connectionTimeoutMillis: 10_000,
    })
    this.pool.on("connect", async (client) => {
      await pgvector.registerTypes(client)
    })
  }

  async initialize(): Promise<void> {
    const client = await this.pool.connect()
    try {
      await client.query("CREATE EXTENSION IF NOT EXISTS vector")

      await client.query(`
        CREATE TABLE IF NOT EXISTS rag_chunks (
          id TEXT PRIMARY KEY,
          container_tag TEXT NOT NULL,
          content TEXT NOT NULL,
          session_id TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          embedding vector(1536) NOT NULL,
          date TEXT,
          metadata JSONB
        )
      `)

      await client.query(`CREATE INDEX IF NOT EXISTS idx_chunks_container ON rag_chunks(container_tag)`)
      await client.query(`CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops)`)

      await client.query(`
        ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
          GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
      `)
      await client.query(`CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON rag_chunks USING gin (content_tsv)`)

      await client.query(`
        CREATE TABLE IF NOT EXISTS rag_entities (
          container_tag TEXT NOT NULL,
          name TEXT NOT NULL,
          type TEXT NOT NULL,
          summary TEXT NOT NULL,
          session_ids TEXT[] NOT NULL DEFAULT '{}',
          PRIMARY KEY (container_tag, name)
        )
      `)

      await client.query(`
        CREATE TABLE IF NOT EXISTS rag_relationships (
          container_tag TEXT NOT NULL,
          source TEXT NOT NULL,
          target TEXT NOT NULL,
          relation TEXT NOT NULL,
          date TEXT,
          session_id TEXT NOT NULL,
          PRIMARY KEY (container_tag, source, relation, target)
        )
      `)

      await client.query(`CREATE INDEX IF NOT EXISTS idx_rel_source ON rag_relationships(container_tag, source)`)
      await client.query(`CREATE INDEX IF NOT EXISTS idx_rel_target ON rag_relationships(container_tag, target)`)
    } finally {
      client.release()
    }
  }

  async close(): Promise<void> {
    await this.pool.end()
  }

  async addChunks(containerTag: string, chunks: Chunk[]): Promise<void> {
    if (chunks.length === 0) return

    const client = await this.pool.connect()
    try {
      await client.query("BEGIN")

      for (const chunk of chunks) {
        await client.query(
          `INSERT INTO rag_chunks (id, container_tag, content, session_id, chunk_index, embedding, date, metadata)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
           ON CONFLICT (id) DO UPDATE SET
             content = EXCLUDED.content,
             session_id = EXCLUDED.session_id,
             chunk_index = EXCLUDED.chunk_index,
             embedding = EXCLUDED.embedding,
             date = EXCLUDED.date,
             metadata = EXCLUDED.metadata`,
          [
            chunk.id,
            containerTag,
            chunk.content,
            chunk.sessionId,
            chunk.chunkIndex,
            pgvector.toSql(chunk.embedding),
            chunk.date || null,
            chunk.metadata ? JSON.stringify(chunk.metadata) : null,
          ]
        )
      }

      await client.query("COMMIT")
    } catch (e) {
      await client.query("ROLLBACK")
      throw e
    } finally {
      client.release()
    }
  }

  async search(
    containerTag: string,
    queryEmbedding: number[],
    query: string,
    limit: number
  ): Promise<SearchResult[]> {
    const embeddingSql = pgvector.toSql(queryEmbedding)

    const { rows } = await this.pool.query(
      `WITH vector_results AS (
        SELECT id, content, session_id, chunk_index, date, metadata,
          1 - (embedding <=> $1::vector) AS vector_score
        FROM rag_chunks
        WHERE container_tag = $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
      ),
      bm25_results AS (
        SELECT id, ts_rank(content_tsv, plainto_tsquery('english', $4)) AS bm25_score
        FROM rag_chunks
        WHERE container_tag = $2 AND content_tsv @@ plainto_tsquery('english', $4)
      )
      SELECT v.*, COALESCE(b.bm25_score, 0) AS bm25_score
      FROM vector_results v
      LEFT JOIN bm25_results b ON v.id = b.id`,
      [embeddingSql, containerTag, limit, query]
    )

    if (rows.length === 0) return []

    let maxBm25 = 0
    for (const row of rows) {
      const s = parseFloat(row.bm25_score)
      if (s > maxBm25) maxBm25 = s
    }

    const results: SearchResult[] = rows.map((row) => {
      const vectorScore = parseFloat(row.vector_score)
      const rawBm25 = parseFloat(row.bm25_score)
      const normalizedBm25 = maxBm25 > 0 ? rawBm25 / maxBm25 : 0
      const score = VECTOR_WEIGHT * vectorScore + BM25_WEIGHT * normalizedBm25

      return {
        content: row.content,
        score,
        vectorScore,
        bm25Score: normalizedBm25,
        sessionId: row.session_id,
        chunkIndex: row.chunk_index,
        date: row.date || undefined,
        metadata: row.metadata || undefined,
      }
    })

    results.sort((a, b) => b.score - a.score)
    return results
  }

  async hasData(containerTag: string): Promise<boolean> {
    const { rows } = await this.pool.query(
      "SELECT 1 FROM rag_chunks WHERE container_tag = $1 LIMIT 1",
      [containerTag]
    )
    return rows.length > 0
  }

  async getChunkCount(containerTag: string): Promise<number> {
    const { rows } = await this.pool.query(
      "SELECT COUNT(*)::int AS count FROM rag_chunks WHERE container_tag = $1",
      [containerTag]
    )
    return rows[0].count
  }

  async addEntity(
    containerTag: string,
    name: string,
    type: string,
    summary: string,
    sessionId: string
  ): Promise<void> {
    const canonical = name.trim()
    if (!canonical) return

    await this.pool.query(
      `INSERT INTO rag_entities (container_tag, name, type, summary, session_ids)
       VALUES ($1, $2, $3, $4, ARRAY[$5])
       ON CONFLICT (container_tag, name) DO UPDATE SET
         summary = SUBSTRING(rag_entities.summary || ' ' || EXCLUDED.summary FROM 1 FOR 500),
         session_ids = array_cat(rag_entities.session_ids, EXCLUDED.session_ids)`,
      [containerTag, canonical, type.toLowerCase(), summary, sessionId]
    )
  }

  async addRelationship(containerTag: string, edge: RelationshipEdge): Promise<void> {
    await this.pool.query(
      `INSERT INTO rag_relationships (container_tag, source, target, relation, date, session_id)
       VALUES ($1, $2, $3, $4, $5, $6)
       ON CONFLICT DO NOTHING`,
      [containerTag, edge.source, edge.target, edge.relation, edge.date || null, edge.sessionId]
    )
  }

  async findEntitiesInQuery(containerTag: string, query: string): Promise<string[]> {
    const words = query
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2)

    if (words.length === 0) return []

    const conditions = words.map((_, i) => `lower(name) LIKE $${i + 2}`).join(" OR ")
    const params: (string | string[])[] = [containerTag, ...words.map((w) => `%${w}%`)]

    const { rows } = await this.pool.query(
      `SELECT DISTINCT name FROM rag_entities WHERE container_tag = $1 AND (${conditions})`,
      params
    )

    return rows.map((r) => r.name)
  }

  async getContext(
    containerTag: string,
    entityNames: string[],
    maxHops = 2
  ): Promise<GraphSearchResult> {
    const visitedNodes = new Set<string>()
    const resultEntities: GraphSearchResult["entities"] = []
    const resultRels: GraphSearchResult["relationships"] = []
    const seenEdges = new Set<string>()

    // Fetch seed entities
    if (entityNames.length > 0) {
      const { rows } = await this.pool.query(
        `SELECT name, type, summary FROM rag_entities
         WHERE container_tag = $1 AND name = ANY($2)`,
        [containerTag, entityNames]
      )
      for (const row of rows) {
        visitedNodes.add(row.name)
        if (resultEntities.length < MAX_GRAPH_ENTITIES) {
          resultEntities.push({ name: row.name, type: row.type, summary: row.summary })
        }
      }
    }

    let frontier = entityNames.filter((n) => visitedNodes.has(n))

    for (let hop = 0; hop < maxHops && frontier.length > 0; hop++) {
      const nextFrontier: string[] = []

      const { rows: edges } = await this.pool.query(
        `SELECT source, target, relation, date FROM rag_relationships
         WHERE container_tag = $1 AND (source = ANY($2) OR target = ANY($2))`,
        [containerTag, frontier]
      )

      for (const edge of edges) {
        const edgeKey = `${edge.source}|${edge.relation}|${edge.target}`
        if (seenEdges.has(edgeKey)) continue
        seenEdges.add(edgeKey)

        if (resultRels.length < MAX_GRAPH_RELATIONSHIPS) {
          resultRels.push({
            source: edge.source,
            target: edge.target,
            relation: edge.relation,
            date: edge.date || undefined,
          })
        }

        for (const other of [edge.source, edge.target]) {
          if (!visitedNodes.has(other)) {
            visitedNodes.add(other)
            nextFrontier.push(other)
          }
        }
      }

      // Fetch newly discovered nodes
      if (nextFrontier.length > 0) {
        const { rows: newNodes } = await this.pool.query(
          `SELECT name, type, summary FROM rag_entities
           WHERE container_tag = $1 AND name = ANY($2)`,
          [containerTag, nextFrontier]
        )
        for (const row of newNodes) {
          if (resultEntities.length < MAX_GRAPH_ENTITIES) {
            resultEntities.push({ name: row.name, type: row.type, summary: row.summary })
          }
        }
      }

      frontier = nextFrontier
    }

    return { entities: resultEntities, relationships: resultRels }
  }

  async getNodeCount(containerTag: string): Promise<number> {
    const { rows } = await this.pool.query(
      "SELECT COUNT(*)::int AS count FROM rag_entities WHERE container_tag = $1",
      [containerTag]
    )
    return rows[0].count
  }

  async getEdgeCount(containerTag: string): Promise<number> {
    const { rows } = await this.pool.query(
      "SELECT COUNT(*)::int AS count FROM rag_relationships WHERE container_tag = $1",
      [containerTag]
    )
    return rows[0].count
  }

  async clear(containerTag: string): Promise<void> {
    const client = await this.pool.connect()
    try {
      await client.query("BEGIN")
      await client.query("DELETE FROM rag_chunks WHERE container_tag = $1", [containerTag])
      await client.query("DELETE FROM rag_entities WHERE container_tag = $1", [containerTag])
      await client.query("DELETE FROM rag_relationships WHERE container_tag = $1", [containerTag])
      await client.query("COMMIT")
    } catch (e) {
      await client.query("ROLLBACK")
      throw e
    } finally {
      client.release()
    }
  }
}
