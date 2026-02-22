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
      prompt: `Rewrite this search query to maximize recall. Expand with key terms and synonyms. Keep it under 200 chars. Output ONLY the rewritten query.\n\nQuery: ${query}`,
    } as Parameters<typeof generateText>[0])

    const rewritten = text.trim()
    if (!rewritten || rewritten.length > 500) {
      logger.warn(`[rewrite] Bad output (len=${rewritten.length}), using original`)
      return query
    }

    logger.debug(`[rewrite] "${query.substring(0, 50)}..." -> "${rewritten.substring(0, 80)}..."`)
    return rewritten
  } catch (e) {
    logger.warn(`[rewrite] Failed, using original: ${e}`)
    return query
  }
}
