import { readFile, writeFile, mkdir } from "fs/promises"
import { embedMany, embed } from "ai"
import { createOpenAI } from "@ai-sdk/openai"
import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import { HybridSearchEngine } from "./search"
import type { Chunk, SearchResult } from "./search"
import { RAG_PROMPTS } from "./prompts"
import { isCountingQuery, decomposeQuery } from "./decompose"
import { EntityGraph } from "./graph"
import type { SerializedGraph, GraphSearchResult } from "./graph"
import { ContainerLock } from "./lock"
import { loadConfig, type RAGConfig } from "./config"

const EMBEDDING_BATCH_SIZE = 100

function chunkText(text: string, chunkSize: number, overlap: number): string[] {
  if (text.length <= chunkSize) {
    return [text.trim()]
  }

  const chunks: string[] = []
  let start = 0

  while (start < text.length) {
    let end = start + chunkSize

    if (end >= text.length) {
      chunks.push(text.slice(start).trim())
      break
    }

    let breakPoint = text.lastIndexOf(". ", end)
    if (breakPoint <= start || breakPoint < start + chunkSize * 0.5) {
      breakPoint = text.lastIndexOf("\n", end)
    }
    if (breakPoint <= start || breakPoint < start + chunkSize * 0.5) {
      breakPoint = text.lastIndexOf(" ", end)
    }
    if (breakPoint <= start) {
      breakPoint = end
    }

    chunks.push(text.slice(start, breakPoint + 1).trim())
    start = breakPoint + 1 - overlap

    if (start < 0) start = 0
  }

  return chunks.filter((c) => c.length > 0)
}

export class RAGProvider implements Provider {
  name = "rag"
  prompts = RAG_PROMPTS
  concurrency = {
    default: 200,
    ingest: 200,
    indexing: 200,
  }

  private searchEngine = new HybridSearchEngine()
  private graphs = new Map<string, EntityGraph>()
  private openai: ReturnType<typeof createOpenAI> | null = null
  private apiKey: string = ""
  private containerLock = new ContainerLock()
  private cacheLoading = new Map<string, Promise<boolean>>()
  private pgStore: any = null
  cfg: RAGConfig = loadConfig()

  private getGraph(containerTag: string): EntityGraph {
    if (!this.graphs.has(containerTag)) {
      this.graphs.set(containerTag, new EntityGraph())
    }
    return this.graphs.get(containerTag)!
  }

  private getCacheDir(containerTag: string): string {
    return `${this.cfg.cacheDir}/${containerTag}`
  }

  private async saveToCache(containerTag: string): Promise<void> {
    const dir = this.getCacheDir(containerTag)
    await mkdir(dir, { recursive: true })

    await this.containerLock.readLock(containerTag)
    let searchJson: string | null = null
    let graphJson: string | null = null
    let searchCount = 0
    let graphNodeCount = 0
    let graphEdgeCount = 0
    try {
      const searchData = this.searchEngine.save(containerTag)
      if (searchData) {
        searchJson = JSON.stringify(searchData)
        searchCount = searchData.chunks.length
      }
      const graph = this.graphs.get(containerTag)
      if (graph && graph.nodeCount > 0) {
        const graphData = graph.save()
        graphJson = JSON.stringify(graphData)
        graphNodeCount = graphData.nodes.length
        graphEdgeCount = graphData.edges.length
      }
    } finally {
      this.containerLock.readUnlock(containerTag)
    }

    const writes: Promise<void>[] = []
    if (searchJson) {
      writes.push(writeFile(`${dir}/search.json`, searchJson))
      logger.info(`[cache] Saved search index for ${containerTag} (${searchCount} chunks)`)
    }
    if (graphJson) {
      writes.push(writeFile(`${dir}/graph.json`, graphJson))
      logger.info(`[cache] Saved entity graph for ${containerTag} (${graphNodeCount} nodes, ${graphEdgeCount} edges)`)
    }
    await Promise.all(writes)
  }

  private loadFromCache(containerTag: string): Promise<boolean> {
    const existing = this.cacheLoading.get(containerTag)
    if (existing) return existing

    const promise = this.doLoadFromCache(containerTag).finally(() => {
      this.cacheLoading.delete(containerTag)
    })
    this.cacheLoading.set(containerTag, promise)
    return promise
  }

  private async doLoadFromCache(containerTag: string): Promise<boolean> {
    const dir = this.getCacheDir(containerTag)
    const searchPath = `${dir}/search.json`

    try {
      const raw = await readFile(searchPath, "utf8")
      const searchData = JSON.parse(raw)

      const graphRaw = await readFile(`${dir}/graph.json`, "utf8").catch(() => null)
      const graphData = graphRaw ? JSON.parse(graphRaw) as SerializedGraph : null

      await this.containerLock.writeLock(containerTag)
      try {
        this.searchEngine.load(containerTag, searchData)
        if (graphData) {
          const graph = this.getGraph(containerTag)
          graph.load(graphData)
        }
      } finally {
        this.containerLock.writeUnlock(containerTag)
      }

      logger.info(`[cache] Loaded index for ${containerTag} (${this.searchEngine.getChunkCount(containerTag)} chunks)`)
      return true
    } catch (e: any) {
      if (e?.code === "ENOENT") return false
      logger.warn(`[cache] Failed to load cache for ${containerTag}: ${e}`)
      return false
    }
  }

  async initialize(config: ProviderConfig): Promise<void> {
    this.apiKey = config.apiKey
    if (!this.apiKey) {
      throw new Error("RAG provider requires OPENAI_API_KEY for embeddings")
    }
    this.openai = createOpenAI({ apiKey: this.apiKey })

    const dbUrl = process.env.DATABASE_URL
    if (dbUrl) {
      const { PgStore } = await import("./pg")
      this.pgStore = new PgStore(dbUrl)
      await this.pgStore.initialize()
      logger.info("Using PostgreSQL + pgvector backend")
    }

    const flags = [
      this.cfg.enableGraphRAG && "graph",
      this.cfg.enableReranker && "reranker",
      this.cfg.enableQueryRewrite && "rewrite",
      this.cfg.enableQueryDecomposition && "decompose",
    ].filter(Boolean).join(" + ")
    logger.info(`Initialized RAG provider (${flags || "hybrid search only"})`)
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    if (!this.openai) throw new Error("Provider not initialized")

    const allChunks: Array<{
      text: string
      sessionId: string
      chunkIndex: number
      date: string
      metadata?: Record<string, unknown>
    }> = []

    for (const session of sessions) {
      const isoDate = (session.metadata?.date as string) || "unknown"
      const dateStr = isoDate !== "unknown" ? isoDate.split("T")[0] : "unknown"

      const conversationText = session.messages
        .map((m) => `${m.role}: ${m.content}`)
        .join("\n")

      const textChunks = chunkText(conversationText, this.cfg.chunkSize, this.cfg.chunkOverlap)

      for (let i = 0; i < textChunks.length; i++) {
        allChunks.push({
          text: textChunks[i],
          sessionId: session.sessionId,
          chunkIndex: i,
          date: dateStr,
          metadata: {
            ...session.metadata,
            memoryDate: dateStr,
          },
        })
      }
    }

    logger.info(`[ingest] ${options.containerTag}: ${sessions.length} sessions, ${allChunks.length} chunks`)

    if (allChunks.length === 0) {
      return { documentIds: [] }
    }

    const embeddingModel = this.openai.embedding(this.cfg.embeddingModel)

    const embedBatch = async (texts: string[]): Promise<number[][]> => {
      for (let attempt = 0; ; attempt++) {
        try {
          const result = await embedMany({ model: embeddingModel, values: texts })
          return result.embeddings
        } catch (e) {
          if (attempt >= 2) throw e
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)))
        }
      }
      throw new Error("unreachable")
    }

    const embedded: Chunk[] = []
    for (let i = 0; i < allChunks.length; i += EMBEDDING_BATCH_SIZE) {
      const batch = allChunks.slice(i, i + EMBEDDING_BATCH_SIZE)
      const embeddings = await embedBatch(batch.map((c) => c.text))

      for (let j = 0; j < batch.length; j++) {
        const chunk = batch[j]
        embedded.push({
          id: `${options.containerTag}_${chunk.sessionId}_${chunk.chunkIndex}`,
          content: chunk.text,
          sessionId: chunk.sessionId,
          chunkIndex: chunk.chunkIndex,
          embedding: embeddings[j],
          date: chunk.date,
          metadata: chunk.metadata,
        })
      }

      logger.debug(`Embedded chunk batch ${Math.floor(i / EMBEDDING_BATCH_SIZE) + 1}/${Math.ceil(allChunks.length / EMBEDDING_BATCH_SIZE)} (${batch.length} chunks)`)
    }

    if (this.pgStore) {
      await this.pgStore.addChunks(options.containerTag, embedded)
    } else {
      await this.containerLock.writeLock(options.containerTag)
      try {
        this.searchEngine.addChunks(options.containerTag, embedded)
      } finally {
        this.containerLock.writeUnlock(options.containerTag)
      }
    }

    if (!this.pgStore) {
      await this.saveToCache(options.containerTag)
    }

    const documentIds = embedded.map((c) => c.id)
    logger.debug(`Ingested ${sessions.length} session(s) as ${embedded.length} chunks for ${options.containerTag}`)

    return { documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    if (!this.openai) throw new Error("Provider not initialized")

    let searchQuery = query
    if (this.cfg.enableQueryRewrite) {
      const { rewriteQuery } = await import("./rewrite")
      searchQuery = await rewriteQuery(this.openai, query)
    }

    const embeddingModel = this.openai.embedding(this.cfg.embeddingModel)
    let queryEmbedding: number[]
    for (let attempt = 0; ; attempt++) {
      try {
        const result = await embed({ model: embeddingModel, value: searchQuery })
        queryEmbedding = result.embedding
        break
      } catch (e) {
        if (attempt >= 2) throw e
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)))
      }
    }

    const limit = options.limit || 10
    const overfetchLimit = this.cfg.enableReranker ? Math.max(limit, this.cfg.rerankOverfetch) : limit

    let hybridResults: SearchResult[]
    let graphResult: GraphSearchResult | null = null

    if (this.pgStore) {
      const promises: Promise<any>[] = [
        this.pgStore.search(options.containerTag, queryEmbedding, searchQuery, overfetchLimit),
        this.cfg.enableGraphRAG
          ? this.pgStore.findEntitiesInQuery(options.containerTag, query)
          : Promise.resolve([]),
      ]

      const [searchResults, queryEntities] = await Promise.all(promises)
      hybridResults = searchResults

      if (this.cfg.enableGraphRAG && queryEntities.length > 0) {
        graphResult = await this.pgStore.getContext(options.containerTag, queryEntities, 2)
        logger.debug(`Graph: found ${queryEntities.length} entities, added ${graphResult!.entities.length} nodes + ${graphResult!.relationships.length} edges`)
      }
    } else {
      if (!this.searchEngine.hasData(options.containerTag)) {
        await this.loadFromCache(options.containerTag)
      }

      await this.containerLock.readLock(options.containerTag)
      try {
        hybridResults = this.searchEngine.search(options.containerTag, queryEmbedding, searchQuery, overfetchLimit)

        if (this.cfg.enableGraphRAG) {
          const graph = this.graphs.get(options.containerTag)
          if (graph && graph.nodeCount > 0) {
            const queryEntities = graph.findEntitiesInQuery(query)
            if (queryEntities.length > 0) {
              graphResult = graph.getContext(queryEntities, 2)
              logger.debug(`Graph: found ${queryEntities.length} entities, added ${graphResult.entities.length} nodes + ${graphResult.relationships.length} edges`)
            }
          }
        }
      } finally {
        this.containerLock.readUnlock(options.containerTag)
      }
    }

    logger.debug(`Hybrid search: ${hybridResults.length} results for "${query.substring(0, 50)}..."`)

    if (this.cfg.enableQueryDecomposition && isCountingQuery(query)) {
      const subQueries = await decomposeQuery(this.openai!, query)
      const existingKeys = new Set(hybridResults.map(r => `${r.sessionId}_${r.chunkIndex}`))

      const subResultsArrays = await Promise.all(
        subQueries.slice(1).map(async (subQuery) => {
          const subEmbed = await embed({ model: embeddingModel, value: subQuery })
          if (this.pgStore) {
            return this.pgStore.search(options.containerTag, subEmbed.embedding, subQuery, overfetchLimit)
          }
          await this.containerLock.readLock(options.containerTag)
          try {
            return this.searchEngine.search(options.containerTag, subEmbed.embedding, subQuery, overfetchLimit)
          } finally {
            this.containerLock.readUnlock(options.containerTag)
          }
        })
      )

      for (const subResults of subResultsArrays) {
        for (const r of subResults) {
          const key = `${r.sessionId}_${r.chunkIndex}`
          if (!existingKeys.has(key)) {
            existingKeys.add(key)
            hybridResults.push(r)
          }
        }
      }

      hybridResults.sort((a, b) => b.score - a.score)
      logger.debug(`[decompose] Merged ${hybridResults.length} total results for counting query`)
    }

    let finalChunks = hybridResults.slice(0, this.cfg.enableReranker ? Math.max(limit, this.cfg.rerankOverfetch) : limit)
    if (this.cfg.enableReranker && finalChunks.length > limit) {
      const { rerankResults } = await import("./reranker")
      finalChunks = await rerankResults(this.openai, query, finalChunks, limit)
    } else {
      finalChunks = finalChunks.slice(0, limit)
    }

    const combinedResults: unknown[] = [...finalChunks]

    if (this.cfg.enableGraphRAG && graphResult) {
      for (const entity of graphResult.entities) {
        combinedResults.push({
          content: entity.summary,
          _type: "entity",
          name: entity.name,
          entityType: entity.type,
          score: 0,
          vectorScore: 0,
          bm25Score: 0,
          sessionId: "",
          chunkIndex: -1,
        })
      }

      for (const rel of graphResult.relationships) {
        combinedResults.push({
          content: `${rel.source} ${rel.relation} ${rel.target}`,
          _type: "relationship",
          source: rel.source,
          target: rel.target,
          relation: rel.relation,
          date: rel.date,
          score: 0,
          vectorScore: 0,
          bm25Score: 0,
          sessionId: "",
          chunkIndex: -1,
        })
      }
    }

    return combinedResults
  }

  async clear(containerTag: string): Promise<void> {
    if (this.pgStore) {
      await this.pgStore.clear(containerTag)
    } else {
      await this.containerLock.writeLock(containerTag)
      try {
        this.searchEngine.clear(containerTag)
        this.graphs.get(containerTag)?.clear()
        this.graphs.delete(containerTag)
      } finally {
        this.containerLock.writeUnlock(containerTag)
      }
    }
    logger.info(`Cleared RAG data for: ${containerTag}`)
  }
}

export default RAGProvider
