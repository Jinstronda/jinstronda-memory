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
import { extractMemories, parseExtractionOutput } from "../../prompts/extraction"
import { rerankResults } from "./reranker"
import { EntityGraph } from "./graph"
import type { SerializedGraph, GraphSearchResult } from "./graph"
import { ContainerLock } from "./lock"

const CHUNK_SIZE = 1600
const CHUNK_OVERLAP = 320
const EMBEDDING_BATCH_SIZE = 100
const EMBEDDING_MODEL = "text-embedding-3-small"
const RERANK_OVERFETCH = 40
const EXTRACTION_CONCURRENCY = 10
const MAX_GLOBAL_EXTRACTIONS = 300
const CACHE_DIR = "./data/cache/rag"

function chunkText(text: string, chunkSize: number = CHUNK_SIZE, overlap: number = CHUNK_OVERLAP): string[] {
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
  private extractionCache = new Map<string, string>()
  private extractionInFlight = new Map<string, Promise<string>>()
  private activeGlobalExtractions = 0
  private extractionQueue: Array<() => void> = []
  private containerLock = new ContainerLock()
  private cacheLoading = new Map<string, Promise<boolean>>()
  private pgStore: any = null

  private async acquireExtractionSlot(): Promise<void> {
    if (this.activeGlobalExtractions < MAX_GLOBAL_EXTRACTIONS) {
      this.activeGlobalExtractions++
      return
    }
    return new Promise((resolve) => {
      this.extractionQueue.push(() => {
        this.activeGlobalExtractions++
        resolve()
      })
    })
  }

  private releaseExtractionSlot(): void {
    this.activeGlobalExtractions--
    const next = this.extractionQueue.shift()
    if (next) next()
  }

  private getGraph(containerTag: string): EntityGraph {
    if (!this.graphs.has(containerTag)) {
      this.graphs.set(containerTag, new EntityGraph())
    }
    return this.graphs.get(containerTag)!
  }

  private getCacheDir(containerTag: string): string {
    return `${CACHE_DIR}/${containerTag}`
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

    if (searchJson) {
      await writeFile(`${dir}/search.json`, searchJson)
      logger.info(`[cache] Saved search index for ${containerTag} (${searchCount} chunks)`)
    }
    if (graphJson) {
      await writeFile(`${dir}/graph.json`, graphJson)
      logger.info(`[cache] Saved entity graph for ${containerTag} (${graphNodeCount} nodes, ${graphEdgeCount} edges)`)
    }
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

      let graphData: SerializedGraph | null = null
      const graphPath = `${dir}/graph.json`
      try {
        graphData = JSON.parse(await readFile(graphPath, "utf8")) as SerializedGraph
      } catch {}

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
      throw new Error("RAG provider requires OPENAI_API_KEY for memory extraction and embeddings")
    }
    this.openai = createOpenAI({ apiKey: this.apiKey })

    const dbUrl = process.env.DATABASE_URL
    if (dbUrl) {
      const { PgStore } = await import("./pg")
      this.pgStore = new PgStore(dbUrl)
      await this.pgStore.initialize()
      logger.info("Using PostgreSQL + pgvector backend")
    }

    logger.info("Initialized RAG provider (hybrid search + entity graph + LLM reranker)")
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    if (!this.openai) throw new Error("Provider not initialized")

    const graph = this.getGraph(options.containerTag)

    const allChunks: Array<{
      text: string
      sessionId: string
      chunkIndex: number
      date: string
      metadata?: Record<string, unknown>
    }> = []

    let activeExtractions = 0
    let completedExtractions = 0
    let cachedHits = 0
    let dedupHits = 0

    const extractSession = async (session: UnifiedSession): Promise<string> => {
      if (this.extractionCache.has(session.sessionId)) {
        cachedHits++
        return this.extractionCache.get(session.sessionId)!
      }
      if (this.extractionInFlight.has(session.sessionId)) {
        dedupHits++
        return this.extractionInFlight.get(session.sessionId)!
      }
      activeExtractions++
      const doExtract = async (): Promise<string> => {
        await this.acquireExtractionSlot()
        try {
          logger.info(`[extract] START ${session.sessionId} (active: ${activeExtractions}, global: ${this.activeGlobalExtractions}/${MAX_GLOBAL_EXTRACTIONS}, queue: ${this.extractionQueue.length})`)
          return await extractMemories(this.openai!, session)
        } finally {
          this.releaseExtractionSlot()
        }
      }
      const promise = doExtract()
      this.extractionInFlight.set(session.sessionId, promise)
      try {
        const result = await promise
        this.extractionCache.set(session.sessionId, result)
        activeExtractions--
        completedExtractions++
        if (completedExtractions % 10 === 0 || completedExtractions <= 3) {
          logger.info(`[extract] DONE ${session.sessionId} (completed: ${completedExtractions}, cached: ${cachedHits}, dedup: ${dedupHits}, global: ${this.activeGlobalExtractions}/${MAX_GLOBAL_EXTRACTIONS})`)
        }
        return result
      } catch (e) {
        activeExtractions--
        throw e
      } finally {
        this.extractionInFlight.delete(session.sessionId)
      }
    }

    logger.info(`[ingest] ${options.containerTag}: ${sessions.length} sessions, extraction concurrency: ${EXTRACTION_CONCURRENCY}`)

    const extractions: string[] = []
    for (let i = 0; i < sessions.length; i += EXTRACTION_CONCURRENCY) {
      const batch = sessions.slice(i, i + EXTRACTION_CONCURRENCY)
      const batchNum = Math.floor(i / EXTRACTION_CONCURRENCY) + 1
      const totalBatches = Math.ceil(sessions.length / EXTRACTION_CONCURRENCY)
      logger.info(`[ingest] ${options.containerTag}: extraction batch ${batchNum}/${totalBatches} (${batch.length} sessions)`)
      const results = await Promise.all(batch.map(extractSession))
      extractions.push(...results)
    }

    // Parse extractions: build graph + prepare chunk text
    let totalEntities = 0
    let totalRelationships = 0

    const parsedSessions = sessions.map((session, si) => {
      const rawExtraction = extractions[si]
      const isoDate = (session.metadata?.date as string) || "unknown"
      const dateStr = isoDate !== "unknown" ? isoDate.split("T")[0] : "unknown"
      const parsed = parseExtractionOutput(rawExtraction)
      return { session, parsed, dateStr }
    })

    if (this.pgStore) {
      for (const { session, parsed } of parsedSessions) {
        for (const entity of parsed.entities) {
          await this.pgStore.addEntity(options.containerTag, entity.name, entity.type, entity.summary, session.sessionId)
        }
        for (const rel of parsed.relationships) {
          await this.pgStore.addRelationship(options.containerTag, {
            source: rel.source,
            target: rel.target,
            relation: rel.relation,
            date: rel.date,
            sessionId: session.sessionId,
          })
        }
        totalEntities += parsed.entities.length
        totalRelationships += parsed.relationships.length
      }
    } else {
      await this.containerLock.writeLock(options.containerTag)
      try {
        for (const { session, parsed } of parsedSessions) {
          for (const entity of parsed.entities) {
            graph.addEntity(entity.name, entity.type, entity.summary, session.sessionId)
          }
          for (const rel of parsed.relationships) {
            graph.addRelationship({
              source: rel.source,
              target: rel.target,
              relation: rel.relation,
              date: rel.date,
              sessionId: session.sessionId,
            })
          }
          totalEntities += parsed.entities.length
          totalRelationships += parsed.relationships.length
        }
      } finally {
        this.containerLock.writeUnlock(options.containerTag)
      }
    }

    for (const { session, parsed, dateStr } of parsedSessions) {
      const dateHeader = `# Memories from ${dateStr}\n\n`
      const content = dateHeader + parsed.memoriesText
      const textChunks = chunkText(content)

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

    if (this.pgStore) {
      logger.info(`[ingest] ${options.containerTag}: graph built with ${totalEntities} entities, ${totalRelationships} relationships (pg backend)`)
    } else {
      logger.info(`[ingest] ${options.containerTag}: graph built with ${totalEntities} entities, ${totalRelationships} relationships (${graph.nodeCount} unique nodes, ${graph.edgeCount} edges)`)
    }

    if (allChunks.length === 0) {
      return { documentIds: [] }
    }

    // Generate embeddings in batches
    const embeddedChunks: Chunk[] = []
    const embeddingModel = this.openai.embedding(EMBEDDING_MODEL)

    for (let i = 0; i < allChunks.length; i += EMBEDDING_BATCH_SIZE) {
      const batch = allChunks.slice(i, i + EMBEDDING_BATCH_SIZE)
      const texts = batch.map((c) => c.text)

      let embeddings: number[][]
      for (let attempt = 0; ; attempt++) {
        try {
          const result = await embedMany({ model: embeddingModel, values: texts })
          embeddings = result.embeddings
          break
        } catch (e) {
          if (attempt >= 2) throw e
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)))
        }
      }

      for (let j = 0; j < batch.length; j++) {
        const chunk = batch[j]
        const id = `${options.containerTag}_${chunk.sessionId}_${chunk.chunkIndex}`
        embeddedChunks.push({
          id,
          content: chunk.text,
          sessionId: chunk.sessionId,
          chunkIndex: chunk.chunkIndex,
          embedding: embeddings[j],
          date: chunk.date,
          metadata: chunk.metadata,
        })
      }

      logger.debug(
        `Embedded batch ${Math.floor(i / EMBEDDING_BATCH_SIZE) + 1}/${Math.ceil(allChunks.length / EMBEDDING_BATCH_SIZE)} (${batch.length} chunks)`
      )
    }

    if (this.pgStore) {
      await this.pgStore.addChunks(options.containerTag, embeddedChunks)
    } else {
      await this.containerLock.writeLock(options.containerTag)
      try {
        this.searchEngine.addChunks(options.containerTag, embeddedChunks)
      } finally {
        this.containerLock.writeUnlock(options.containerTag)
      }
      await this.saveToCache(options.containerTag)
    }

    const documentIds = embeddedChunks.map((c) => c.id)
    logger.debug(
      `Ingested ${sessions.length} session(s) as ${embeddedChunks.length} chunks for ${options.containerTag}`
    )

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

    const embeddingModel = this.openai.embedding(EMBEDDING_MODEL)
    let queryEmbedding: number[]
    for (let attempt = 0; ; attempt++) {
      try {
        const result = await embed({ model: embeddingModel, value: query })
        queryEmbedding = result.embedding
        break
      } catch (e) {
        if (attempt >= 2) throw e
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)))
      }
    }

    const limit = options.limit || 10
    const overfetchLimit = Math.max(limit, RERANK_OVERFETCH)

    let hybridResults: SearchResult[]
    let graphResult: GraphSearchResult | null = null

    if (this.pgStore) {
      hybridResults = await this.pgStore.search(options.containerTag, queryEmbedding, query, overfetchLimit)
      const nodeCount = await this.pgStore.getNodeCount(options.containerTag)
      if (nodeCount > 0) {
        const queryEntities = await this.pgStore.findEntitiesInQuery(options.containerTag, query)
        if (queryEntities.length > 0) {
          graphResult = await this.pgStore.getContext(options.containerTag, queryEntities, 2)
          logger.debug(`Graph: found ${queryEntities.length} entities, added ${graphResult!.entities.length} nodes + ${graphResult!.relationships.length} edges`)
        }
      }
    } else {
      if (!this.searchEngine.hasData(options.containerTag)) {
        await this.loadFromCache(options.containerTag)
      }

      await this.containerLock.readLock(options.containerTag)
      try {
        hybridResults = this.searchEngine.search(options.containerTag, queryEmbedding, query, overfetchLimit)

        const graph = this.graphs.get(options.containerTag)
        if (graph && graph.nodeCount > 0) {
          const queryEntities = graph.findEntitiesInQuery(query)
          if (queryEntities.length > 0) {
            graphResult = graph.getContext(queryEntities, 2)
            logger.debug(`Graph: found ${queryEntities.length} entities, added ${graphResult.entities.length} nodes + ${graphResult.relationships.length} edges`)
          }
        }
      } finally {
        this.containerLock.readUnlock(options.containerTag)
      }
    }

    if (this.pgStore) {
      const chunkCount = await this.pgStore.getChunkCount(options.containerTag)
      logger.debug(`Hybrid search: ${hybridResults.length} results for "${query.substring(0, 50)}..." (${chunkCount} total chunks, pg)`)
    } else {
      logger.debug(`Hybrid search: ${hybridResults.length} results for "${query.substring(0, 50)}..." (${this.searchEngine.getChunkCount(options.containerTag)} total chunks)`)
    }

    let finalChunks = hybridResults
    if (hybridResults.length > limit) {
      finalChunks = await rerankResults(this.openai, query, hybridResults, limit)
    }

    const combinedResults: unknown[] = [...finalChunks]

    if (graphResult) {
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
