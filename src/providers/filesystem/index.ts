import { mkdir, readdir, readFile, writeFile, rm } from "node:fs/promises"
import { join } from "node:path"
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
import { FILESYSTEM_PROMPTS } from "./prompts"

const BASE_DIR = join(process.cwd(), "data", "providers", "filesystem")

/**
 * Format a conversation session as a human-readable Markdown file.
 * Mirrors how Claude Code stores context in CLAUDE.md / MEMORY.md files.
 */
function formatSessionAsMarkdown(session: UnifiedSession): string {
  const lines: string[] = []
  lines.push(`# Session: ${session.sessionId}`)
  lines.push("")

  if (session.metadata?.formattedDate) {
    lines.push(`**Date:** ${session.metadata.formattedDate}`)
  } else if (session.metadata?.date) {
    lines.push(`**Date:** ${session.metadata.date}`)
  }

  if (session.metadata) {
    const metaEntries = Object.entries(session.metadata).filter(
      ([k]) => !["date", "formattedDate"].includes(k)
    )
    if (metaEntries.length > 0) {
      for (const [key, value] of metaEntries) {
        if (typeof value === "string" || typeof value === "number") {
          lines.push(`**${key}:** ${value}`)
        }
      }
    }
  }

  lines.push("")
  lines.push("---")
  lines.push("")

  for (const msg of session.messages) {
    const speaker = msg.speaker || msg.role
    const timestamp = msg.timestamp ? ` [${msg.timestamp}]` : ""
    lines.push(`**${speaker}** (${msg.role})${timestamp}: ${msg.content}`)
    lines.push("")
  }

  return lines.join("\n")
}

/**
 * Simple tokenizer: lowercase, split on non-alphanumeric, filter short tokens.
 * Deliberately kept simple to represent the filesystem-based approach.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1)
}

/**
 * Score a document against query terms using simple term matching.
 * Returns a score between 0 and 1 representing the fraction of query terms found,
 * with a small frequency bonus for repeated matches.
 */
function scoreDocument(queryTerms: string[], docText: string): { score: number; matchCount: number } {
  if (queryTerms.length === 0) return { score: 0, matchCount: 0 }

  const docLower = docText.toLowerCase()
  let matchCount = 0
  let totalFrequency = 0

  for (const term of queryTerms) {
    if (docLower.includes(term)) {
      matchCount++
      // Count occurrences for frequency bonus
      let idx = 0
      let count = 0
      while ((idx = docLower.indexOf(term, idx)) !== -1) {
        count++
        idx += term.length
      }
      totalFrequency += count
    }
  }

  const termCoverage = matchCount / queryTerms.length
  const frequencyBonus = Math.min(totalFrequency / 100, 0.1)

  return {
    score: Math.min(termCoverage + frequencyBonus, 1.0),
    matchCount,
  }
}

/**
 * Filesystem Memory Provider
 *
 * Implements the Claude Code MEMORY.md / CLAUDE.md approach to memory:
 * - Stores conversations as plain Markdown files on the filesystem
 * - No embeddings, no vector database, no pre-processing
 * - Search is simple text matching across files
 * - The LLM reasons over raw conversation text
 *
 * This represents the simplest possible memory approach: write to files, read from files.
 * Research from Letta shows this approach scores ~74% on LoCoMo, demonstrating that
 * LLMs can effectively reason over raw text without sophisticated retrieval.
 */
export class FilesystemProvider implements Provider {
  name = "filesystem"
  prompts = FILESYSTEM_PROMPTS
  concurrency = {
    default: 50,
    ingest: 100,
  }

  async initialize(_config: ProviderConfig): Promise<void> {
    await mkdir(BASE_DIR, { recursive: true })
    logger.info("Initialized Filesystem memory provider (CLAUDE.md-style)")
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const containerDir = join(BASE_DIR, sanitizePath(options.containerTag))
    const sessionsDir = join(containerDir, "sessions")
    await mkdir(sessionsDir, { recursive: true })

    const documentIds: string[] = []

    for (const session of sessions) {
      const markdown = formatSessionAsMarkdown(session)
      const safeId = sanitizePath(session.sessionId)
      const filePath = join(sessionsDir, `${safeId}.md`)
      await writeFile(filePath, markdown, "utf-8")
      documentIds.push(safeId)
      logger.debug(`Ingested session ${session.sessionId} as markdown`)
    }

    return { documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    // Filesystem indexing is instant - no async processing needed
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const containerDir = join(BASE_DIR, sanitizePath(options.containerTag))
    const sessionsDir = join(containerDir, "sessions")

    let files: string[]
    try {
      files = await readdir(sessionsDir)
    } catch {
      logger.warn(`No sessions directory found for ${options.containerTag}`)
      return []
    }

    const mdFiles = files.filter((f) => f.endsWith(".md"))
    if (mdFiles.length === 0) return []

    const queryTerms = tokenize(query)

    const scored: Array<{
      sessionId: string
      content: string
      score: number
      matchCount: number
    }> = []

    for (const file of mdFiles) {
      const content = await readFile(join(sessionsDir, file), "utf-8")
      const { score, matchCount } = scoreDocument(queryTerms, content)
      scored.push({
        sessionId: file.replace(".md", ""),
        content,
        score,
        matchCount,
      })
    }

    // Sort by score (desc), then by matchCount (desc) as tiebreaker
    scored.sort((a, b) => b.score - a.score || b.matchCount - a.matchCount)

    const limit = options.limit || 10

    // Return top results; include score=0 results only if we have fewer than limit scored results
    const scoredResults = scored.filter((r) => r.score > 0)
    if (scoredResults.length >= limit) {
      return scoredResults.slice(0, limit)
    }

    // Fill remaining slots with unscored results (chronological order fallback)
    const unscoredResults = scored.filter((r) => r.score === 0)
    return [...scoredResults, ...unscoredResults].slice(0, limit)
  }

  async clear(containerTag: string): Promise<void> {
    const containerDir = join(BASE_DIR, sanitizePath(containerTag))
    try {
      await rm(containerDir, { recursive: true, force: true })
      logger.info(`Cleared filesystem data for: ${containerTag}`)
    } catch (e) {
      logger.warn(`Failed to clear filesystem data: ${e}`)
    }
  }
}

/** Sanitize a string for safe use as a filesystem path component */
function sanitizePath(input: string): string {
  return input.replace(/[^a-zA-Z0-9_.-]/g, "_")
}

export default FilesystemProvider
