import { createOpenAI } from "@ai-sdk/openai"
import { generateText } from "ai"
import { logger } from "../../utils/logger"

const DECOMPOSE_MODEL = "gpt-5-nano"

export function isCountingQuery(query: string): boolean {
  const q = query.toLowerCase()
  return /\b(how many|how much|list all|what are all|count|total|every|each of)\b/.test(q)
    || /\b(all the|all my|everything)\b/.test(q)
}

export async function decomposeQuery(
  openai: ReturnType<typeof createOpenAI>,
  query: string
): Promise<string[]> {
  const queries = [query]

  try {
    const { text } = await generateText({
      model: openai(DECOMPOSE_MODEL),
      prompt: `You are a search query decomposer. The user wants to count or list items from their conversation history.

Original question: ${query}

Generate 3-5 alternative search queries that would find different instances of what's being counted. Each query should use different wording, synonyms, or focus on different aspects.

Example: "How many books did I mention reading?" ->
- books I was reading
- novels I mentioned
- reading list recommendations
- book titles discussed

Return ONLY the queries, one per line. No numbering, no explanations.`,
    } as Parameters<typeof generateText>[0])

    const subQueries = text.trim().split('\n')
      .map(l => l.replace(/^[-*\d.)\s]+/, '').trim())
      .filter(l => l.length > 3 && l.length < 200)
      .slice(0, 5)

    queries.push(...subQueries)
    logger.debug(`[decompose] "${query.substring(0, 50)}..." -> ${queries.length} queries`)
  } catch (e) {
    logger.warn(`[decompose] Failed, using original query only: ${e}`)
  }

  return queries
}
