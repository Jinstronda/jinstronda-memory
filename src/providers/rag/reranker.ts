import { createOpenAI } from "@ai-sdk/openai"
import { generateText } from "ai"
import type { SearchResult } from "./search"
import { logger } from "../../utils/logger"

const RERANKER_MODEL = "gpt-5-nano"

interface RerankScore {
  index: number
  score: number
}

function detectQueryType(query: string): string {
  const q = query.toLowerCase()
  if (/\b(when|what (date|time|day|month|year)|how long ago|how recently|last time|first time|before|after)\b/.test(q)) {
    return "temporal"
  }
  if (/\b(change|update|move|switch|new|current|now|still|anymore|used to|latest)\b/.test(q)) {
    return "knowledge-update"
  }
  if (/\bwhat .+ (of|for) .+ (the|my|a) .+\b/.test(q) || /\b\w+'s \w+'s\b/.test(q)) {
    return "multi-hop"
  }
  if (/\b(favorite|prefer|like|enjoy|love|hate|dislike|opinion)\b/.test(q)) {
    return "preference"
  }
  if (/\b(you (said|told|recommended|suggested|mentioned)|did you|your (advice|recommendation|suggestion))\b/.test(q)) {
    return "assistant-recall"
  }
  if (/\b(who|what|where|which|name|tell me about)\b/.test(q)) {
    return "factual"
  }
  return "general"
}

const TYPE_INSTRUCTIONS: Record<string, string> = {
  temporal:
    "TEMPORAL question about when something happened. Prioritize passages with specific dates, times, or temporal references. Date metadata in brackets is the event date.",
  "knowledge-update":
    "KNOWLEDGE-UPDATE question about information that may have changed over time. Prioritize passages with the most recent dates, as they reflect the current state.",
  "multi-hop":
    "MULTI-HOP question requiring info from multiple sources or relationship chains. Prioritize passages containing entity names and relationships from the query.",
  preference:
    "PREFERENCE question about likes, dislikes, or opinions. Prioritize passages with explicit preference statements.",
  "assistant-recall":
    "Question about what the ASSISTANT said or recommended. Prioritize passages containing assistant/AI responses rather than user statements.",
  factual:
    "FACTUAL question about a specific person, place, thing, or event. Prioritize passages directly mentioning the queried entities.",
  general: "Score based on how directly the passage answers or provides evidence for the query.",
}

export async function rerankResults(
  openai: ReturnType<typeof createOpenAI>,
  query: string,
  results: SearchResult[],
  topK: number
): Promise<SearchResult[]> {
  if (results.length <= topK) return results

  const queryType = detectQueryType(query)

  const candidateList = results
    .map((r, i) => {
      const dateInfo = r.date ? ` [Date: ${r.date}]` : ""
      return `[${i}]${dateInfo} ${r.content.substring(0, 1000)}`
    })
    .join("\n\n")

  const prompt = `You are a search result reranker. Score each passage's relevance to the query from 0 to 10.

Query: ${query}
Query Type: ${queryType}
${TYPE_INSTRUCTIONS[queryType]}

Candidates:
${candidateList}

10 = directly contains the answer, 0 = irrelevant.
Return ALL candidates as a JSON array sorted by score descending:
[{"index": 0, "score": 8}, {"index": 3, "score": 7}, ...]`

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const { text } = await generateText({
        model: openai(RERANKER_MODEL),
        prompt,
      } as Parameters<typeof generateText>[0])

      const jsonMatch = text.match(/\[[\s\S]*\]/)
      if (!jsonMatch) {
        if (attempt < 2) {
          logger.warn(`Reranker: parse failed attempt ${attempt + 1}, retrying`)
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)))
          continue
        }
        logger.warn("Reranker: parse failed after retries, returning hybrid order")
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

      logger.debug(`Reranked ${results.length} -> ${reranked.length} (type: ${queryType}, top: ${reranked[0]?.rerankScore ?? 0})`)
      return reranked
    } catch (e) {
      if (attempt < 2) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)))
        continue
      }
      logger.warn(`Reranker failed after retries: ${e instanceof Error ? e.message : e}`)
      return results.slice(0, topK)
    }
  }

  return results.slice(0, topK)
}
