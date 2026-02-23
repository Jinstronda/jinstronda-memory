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
import { MEM0_PROMPTS } from "../mem0/prompts"

const INGEST_TIMEOUT_MS = 10 * 60 * 1000 // 10 minutes per session
const SEARCH_TIMEOUT_MS = 60 * 1000 // 1 minute

export class Mem0LocalProvider implements Provider {
  name = "mem0-local"
  prompts = MEM0_PROMPTS
  concurrency = { default: 200 }
  private baseUrl = ""

  async initialize(config: ProviderConfig): Promise<void> {
    this.baseUrl = (config.baseUrl || "http://localhost:8080").replace(/\/$/, "")

    const res = await fetch(`${this.baseUrl}/health`)
    if (!res.ok) throw new Error(`mem0-local server not reachable at ${this.baseUrl}`)

    logger.info(`Initialized mem0-local provider at ${this.baseUrl}`)
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const ids: string[] = []

    for (const session of sessions) {
      const content = session.messages
        .map((m) => `${m.role}: ${m.content}`)
        .join("\n")

      const res = await fetch(`${this.baseUrl}/v1/memories/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(INGEST_TIMEOUT_MS),
        body: JSON.stringify({
          messages: content,
          user_id: options.containerTag,
          metadata: {
            sessionId: session.sessionId,
            timestamp: session.metadata?.date,
            ...session.metadata,
            ...options.metadata,
          },
          infer: true,
        }),
      })

      if (!res.ok) {
        const text = await res.text().catch(() => "")
        logger.warn(`ingest failed for session ${session.sessionId}: ${res.status} ${text}`)
        continue
      }

      const data = (await res.json()) as { ids?: string[]; results?: Array<{ id?: string }> }
      const resultIds = data.ids ?? data.results?.map((r) => r.id).filter(Boolean) ?? []
      ids.push(...(resultIds as string[]))
    }

    return { documentIds: ids }
  }

  async awaitIndexing(
    _result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    onProgress?.({ completedIds: _result.documentIds, failedIds: [], total: _result.documentIds.length })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const res = await fetch(`${this.baseUrl}/v1/memories/search/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
      body: JSON.stringify({
        query,
        user_id: options.containerTag,
        top_k: options.limit || 30,
        rerank: true,
      }),
    })

    if (!res.ok) {
      const text = await res.text().catch(() => "")
      logger.warn(`search failed: ${res.status} ${text}`)
      return []
    }

    const data = (await res.json()) as { results?: unknown[] }
    return data.results ?? []
  }

  async clear(containerTag: string): Promise<void> {
    await fetch(`${this.baseUrl}/v1/memories/?user_id=${encodeURIComponent(containerTag)}`, {
      method: "DELETE",
    })
    logger.info(`Cleared memories for user: ${containerTag}`)
  }
}

export default Mem0LocalProvider
