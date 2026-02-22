import { createOpenAI } from "@ai-sdk/openai"
import { generateText } from "ai"
import type { SearchResult } from "./search"
import { logger } from "../../utils/logger"

const RERANKER_MODEL = "gpt-5-nano"

interface RerankScore {
  index: number
  score: number
}

export async function rerankResults(
  openai: ReturnType<typeof createOpenAI>,
  query: string,
  results: SearchResult[],
  topK: number
): Promise<SearchResult[]> {
  if (results.length <= topK) return results

  const candidateList = results
    .map((r, i) => `[${i}]${r.date ? ` [${r.date}]` : ""} ${r.content.substring(0, 400)}`)
    .join("\n")

  const prompt = `Rerank by relevance to query. Return JSON array: [{"index":0,"score":8},...]
Query: ${query}
${candidateList}`

  try {
    const { text } = await generateText({
      model: openai(RERANKER_MODEL),
      prompt,
    } as Parameters<typeof generateText>[0])

    const jsonMatch = text.match(/\[[\s\S]*\]/)
    if (!jsonMatch) {
      logger.warn("Reranker: parse failed, returning hybrid order")
      return results.slice(0, topK)
    }

    const scores: RerankScore[] = JSON.parse(jsonMatch[0])
    const scoreMap = new Map<number, number>()
    for (const s of scores) scoreMap.set(s.index, s.score)

    const reranked = results
      .map((r, i) => ({ result: r, rerankScore: scoreMap.get(i) ?? 0 }))
      .sort((a, b) => b.rerankScore - a.rerankScore)
      .slice(0, topK)
      .map(({ result, rerankScore }) => ({
        ...result,
        rerankScore,
        score: rerankScore / 10,
      }))

    logger.debug(`Reranked ${results.length} -> ${reranked.length} (top: ${reranked[0]?.rerankScore ?? 0})`)
    return reranked
  } catch (e) {
    logger.warn(`Reranker failed: ${e instanceof Error ? e.message : e}`)
    return results.slice(0, topK)
  }
}
