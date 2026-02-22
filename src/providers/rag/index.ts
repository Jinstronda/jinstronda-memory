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
import { isCountingQuery, decomposeQuery } from "./decompose"
import { EntityGraph } from "./graph"
import type { SerializedGraph, GraphSearchResult } from "./graph"
import { ContainerLock } from "./lock"
import { InMemoryFactStore } from "./facts"
import type { AtomicFact, FactSearchResult } from "./facts"

import { buildProfile, formatProfileContext } from "./profile"
import type { UserProfile } from "./profile"

const CHUNK_SIZE = 1600
const CHUNK_OVERLAP = 320
const EMBEDDING_BATCH_SIZE = 100
const EMBEDDING_MODEL = "text-embedding-3-large"
const RERANK_OVERFETCH = 40
const EXTRACTION_CONCURRENCY = 10
const MAX_GLOBAL_EXTRACTIONS = 300
const FACT_SEARCH_LIMIT = 30
const FACT_SESSION_BOOST = 0.1
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
  private factStore = new InMemoryFactStore()
  private profiles = new Map<string, UserProfile>()
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
    let factsJson: string | null = null
    let profileJson: string | null = null
    let searchCount = 0
    let graphNodeCount = 0
    let graphEdgeCount = 0
    let factCount = 0
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
      const factsData = this.factStore.save(containerTag)
      if (factsData) {
        factsJson = JSON.stringify(factsData)
        factCount = factsData.facts.length
      }
      const profile = this.profiles.get(containerTag)
      if (profile && profile.facts.length > 0) {
        profileJson = JSON.stringify(profile)
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
    if (factsJson) {
      writes.push(writeFile(`${dir}/facts.json`, factsJson))
      logger.info(`[cache] Saved ${factCount} atomic facts for ${containerTag}`)
    }
    if (profileJson) {
      writes.push(writeFile(`${dir}/profile.json`, profileJson))
      logger.info(`[cache] Saved user profile for ${containerTag}`)
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

      const [graphRaw, factsRaw, profileRaw] = await Promise.all([
        readFile(`${dir}/graph.json`, "utf8").catch(() => null),
        readFile(`${dir}/facts.json`, "utf8").catch(() => null),
        readFile(`${dir}/profile.json`, "utf8").catch(() => null),
      ])

      const graphData = graphRaw ? JSON.parse(graphRaw) as SerializedGraph : null
      const factsData = factsRaw ? JSON.parse(factsRaw) as { facts: AtomicFact[] } : null
      const profileData = profileRaw ? JSON.parse(profileRaw) as UserProfile : null

      await this.containerLock.writeLock(containerTag)
      try {
        this.searchEngine.load(containerTag, searchData)
        if (graphData) {
          const graph = this.getGraph(containerTag)
          graph.load(graphData)
        }
        if (factsData) {
          this.factStore.load(containerTag, factsData)
        }
        if (profileData) {
          this.profiles.set(containerTag, profileData)
        }
      } finally {
        this.containerLock.writeUnlock(containerTag)
      }

      const factCount = this.factStore.getFactCount(containerTag)
      logger.info(`[cache] Loaded index for ${containerTag} (${this.searchEngine.getChunkCount(containerTag)} chunks, ${factCount} facts)`)
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
      eventDate?: string
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

    // Single pass: extract chunks + atomic facts from parsed sessions
    const rawFacts: Array<{ text: string; sessionId: string; date: string; eventDate?: string; factIndex: number }> = []
    for (const { session, parsed, dateStr } of parsedSessions) {
      const dateHeader = `# Memories from ${dateStr}\n\n`
      const content = dateHeader + parsed.memoriesText
      const textChunks = chunkText(content)

      for (let i = 0; i < textChunks.length; i++) {
        const dateMatches = textChunks[i].match(/\[(\d{4}-\d{2}-\d{2})\]/g)
        const eventDatesInChunk = dateMatches?.map((d) => d.slice(1, -1)).sort() || []
        const eventDate = eventDatesInChunk[0]

        allChunks.push({
          text: textChunks[i],
          sessionId: session.sessionId,
          chunkIndex: i,
          date: dateStr,
          eventDate,
          metadata: {
            ...session.metadata,
            memoryDate: dateStr,
            eventDate,
          },
        })
      }

      const lines = parsed.memoriesText.split("\n").map((l) => l.trim()).filter((l) => l.length > 5)
      for (let i = 0; i < lines.length; i++) {
        const dateMatch = lines[i].match(/^\[(\d{4}-\d{2}-\d{2})\]/)
        rawFacts.push({
          text: lines[i],
          sessionId: session.sessionId,
          date: dateStr,
          eventDate: dateMatch ? dateMatch[1] : undefined,
          factIndex: i,
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

    const embeddingModel = this.openai.embedding(EMBEDDING_MODEL)

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

    // Run chunk embedding, fact embedding, and profile building in parallel
    const embedAndStoreChunks = async (): Promise<Chunk[]> => {
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
            eventDate: chunk.eventDate,
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

      return embedded
    }

    const embedAndStoreFacts = async (): Promise<AtomicFact[]> => {
      if (rawFacts.length === 0) return []

      const embedded: AtomicFact[] = []
      for (let i = 0; i < rawFacts.length; i += EMBEDDING_BATCH_SIZE) {
        const batch = rawFacts.slice(i, i + EMBEDDING_BATCH_SIZE)
        const embeddings = await embedBatch(batch.map((f) => f.text))

        for (let j = 0; j < batch.length; j++) {
          const fact = batch[j]
          embedded.push({
            id: `${options.containerTag}_${fact.sessionId}_fact_${fact.factIndex}`,
            content: fact.text,
            sessionId: fact.sessionId,
            date: fact.date,
            eventDate: fact.eventDate,
            factIndex: fact.factIndex,
            embedding: embeddings[j],
          })
        }

        logger.debug(`Embedded fact batch ${Math.floor(i / EMBEDDING_BATCH_SIZE) + 1}/${Math.ceil(rawFacts.length / EMBEDDING_BATCH_SIZE)} (${batch.length} facts)`)
      }

      if (this.pgStore) {
        await this.pgStore.addFacts(options.containerTag, embedded)
      } else {
        await this.containerLock.writeLock(options.containerTag)
        try {
          this.factStore.addFacts(options.containerTag, embedded)
        } finally {
          this.containerLock.writeUnlock(options.containerTag)
        }
      }

      logger.info(`[ingest] ${options.containerTag}: stored ${embedded.length} atomic facts`)
      return embedded
    }

    const buildProfileAsync = async (): Promise<void> => {
      const allMemoriesText = parsedSessions.map((s) => s.parsed.memoriesText).join("\n\n")
      const existingProfile = this.profiles.get(options.containerTag)
      const profile = await buildProfile(this.openai!, allMemoriesText, existingProfile)
      if (profile.facts.length > 0) {
        this.profiles.set(options.containerTag, profile)
        logger.info(`[ingest] ${options.containerTag}: user profile with ${profile.facts.length} facts`)
      }
    }

    const [embeddedChunks, embeddedFacts] = await Promise.all([
      embedAndStoreChunks(),
      embedAndStoreFacts(),
      buildProfileAsync(),
    ])

    if (!this.pgStore) {
      await this.saveToCache(options.containerTag)
    }

    const documentIds = embeddedChunks.map((c) => c.id)
    logger.debug(
      `Ingested ${sessions.length} session(s) as ${embeddedChunks.length} chunks + ${embeddedFacts.length} facts for ${options.containerTag}`
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
    const factMatchedSessions = new Set<string>()
    let factResults: FactSearchResult[] = []

    if (this.pgStore) {
      const graphEntityPromise = this.pgStore.findEntitiesInQuery(options.containerTag, query)

      const [fr, searchResults, queryEntities] = await Promise.all([
        this.pgStore.searchFacts(options.containerTag, queryEmbedding, FACT_SEARCH_LIMIT),
        this.pgStore.search(options.containerTag, queryEmbedding, query, overfetchLimit),
        graphEntityPromise,
      ])

      factResults = fr
      hybridResults = searchResults
      for (const f of factResults) factMatchedSessions.add(f.sessionId)
      if (factResults.length > 0) {
        logger.debug(`[facts] Found ${factResults.length} matching facts from ${factMatchedSessions.size} sessions (pg)`)
      }

      if (queryEntities.length > 0) {
        graphResult = await this.pgStore.getContext(options.containerTag, queryEntities, 2)
        logger.debug(`Graph: found ${queryEntities.length} entities, added ${graphResult!.entities.length} nodes + ${graphResult!.relationships.length} edges`)
      }
    } else {
      if (!this.searchEngine.hasData(options.containerTag)) {
        await this.loadFromCache(options.containerTag)
      }

      await this.containerLock.readLock(options.containerTag)
      try {
        factResults = this.factStore.search(options.containerTag, queryEmbedding, FACT_SEARCH_LIMIT)
        for (const f of factResults) factMatchedSessions.add(f.sessionId)
        if (factResults.length > 0) {
          logger.debug(`[facts] Found ${factResults.length} matching facts from ${factMatchedSessions.size} sessions`)
        }

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

    // Boost chunks from fact-matched sessions
    if (factMatchedSessions.size > 0) {
      for (const r of hybridResults) {
        if (factMatchedSessions.has(r.sessionId)) {
          r.score += FACT_SESSION_BOOST
        }
      }
      hybridResults.sort((a, b) => b.score - a.score)
    }

    // Include parent chunks for top matched facts
    if (factResults.length > 0) {
      const existingKeys = new Set(hybridResults.map(r => `${r.sessionId}_${r.chunkIndex}`))
      const topFacts = factResults.slice(0, 10)
      const uniqueSessions = [...new Set(topFacts.map(f => f.sessionId))]

      const sessionChunksMap = new Map<string, Array<{ content: string; sessionId: string; chunkIndex: number; date?: string; eventDate?: string; metadata?: Record<string, unknown> }>>()
      if (this.pgStore) {
        const results = await Promise.all(uniqueSessions.map(sid => this.pgStore!.getChunksBySession(options.containerTag, sid)))
        for (let i = 0; i < uniqueSessions.length; i++) sessionChunksMap.set(uniqueSessions[i], results[i])
      } else {
        for (const sid of uniqueSessions) sessionChunksMap.set(sid, this.searchEngine.getChunksBySession(options.containerTag, sid))
      }

      let injected = 0
      for (const fact of topFacts) {
        const chunks = sessionChunksMap.get(fact.sessionId) || []
        for (const chunk of chunks) {
          if (!chunk.content.includes(fact.content)) continue
          const key = `${chunk.sessionId}_${chunk.chunkIndex}`
          if (existingKeys.has(key)) continue
          existingKeys.add(key)
          hybridResults.push({
            content: chunk.content,
            score: fact.score,
            vectorScore: 0,
            bm25Score: 0,
            sessionId: chunk.sessionId,
            chunkIndex: chunk.chunkIndex,
            date: chunk.date,
            eventDate: chunk.eventDate,
            metadata: chunk.metadata,
          })
          injected++
        }
      }

      if (injected > 0) {
        hybridResults.sort((a, b) => b.score - a.score)
        logger.debug(`[facts] Injected ${injected} parent chunks for matched facts`)
      }
    }

    logger.debug(`Hybrid search: ${hybridResults.length} results for "${query.substring(0, 50)}..."`)

    if (isCountingQuery(query)) {
      const subQueries = await decomposeQuery(this.openai!, query)
      const existingKeys = new Set(hybridResults.map(r => `${r.sessionId}_${r.chunkIndex}`))

      // Parallel: embed + search all sub-queries concurrently
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

    // Inject user profile as always-present context
    const profile = this.profiles.get(options.containerTag)
    if (profile && profile.facts.length > 0) {
      combinedResults.push({
        content: formatProfileContext(profile),
        _type: "profile",
        score: 0,
        vectorScore: 0,
        bm25Score: 0,
        sessionId: "",
        chunkIndex: -1,
      })
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
        this.factStore.clear(containerTag)
        this.graphs.get(containerTag)?.clear()
        this.graphs.delete(containerTag)
      } finally {
        this.containerLock.writeUnlock(containerTag)
      }
    }
    this.profiles.delete(containerTag)
    logger.info(`Cleared RAG data for: ${containerTag}`)
  }

  getProfile(containerTag: string): UserProfile | null {
    return this.profiles.get(containerTag) ?? null
  }
}

export default RAGProvider
