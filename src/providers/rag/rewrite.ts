import { createOpenAI } from "@ai-sdk/openai"
import { generateText } from "ai"
import { logger } from "../../utils/logger"

const REWRITE_MODEL = "gpt-5-nano"

export async function rewriteQuery(
  openai: ReturnType<typeof createOpenAI>,
  query: string
): Promise<string> {
  try {
    const { text } = await generateText({
      model: openai(REWRITE_MODEL),
      prompt: `You are a search query rewriter for a memory/conversation retrieval system.

Rewrite the following query to maximize recall. Expand with key terms, synonyms, and related concepts. Preserve proper nouns, dates, and key entities. Keep it as a single line.

Output ONLY the rewritten query, nothing else.

Query: ${query}`,
    } as Parameters<typeof generateText>[0])

    const rewritten = text.trim()

    if (!rewritten || rewritten.length > 500) {
      logger.warn(`[rewrite] Bad output (len=${rewritten.length}), using original`)
      return query
    }

    logger.debug(`[rewrite] "${query.substring(0, 50)}..." -> "${rewritten.substring(0, 80)}..."`)
    return rewritten
  } catch (e) {
    logger.warn(`[rewrite] Failed, using original query: ${e}`)
    return query
  }
}
