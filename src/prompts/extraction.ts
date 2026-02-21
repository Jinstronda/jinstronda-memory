import { createOpenAI } from "@ai-sdk/openai"
import { generateText } from "ai"
import type { UnifiedSession } from "../types/unified"
import { logger } from "../utils/logger"

const EXTRACTION_MODEL = "gpt-5-nano"

export interface ParsedExtraction {
  memoriesText: string
  entities: Array<{ name: string; type: string; summary: string }>
  relationships: Array<{ source: string; relation: string; target: string; date?: string }>
  eventDates: Map<number, string>
}

export function buildExtractionPrompt(session: UnifiedSession): string {
  const speakerA = (session.metadata?.speakerA as string) || "Speaker A"
  const speakerB = (session.metadata?.speakerB as string) || "Speaker B"
  const date =
    (session.metadata?.formattedDate as string) ||
    (session.metadata?.date as string) ||
    "Unknown date"

  const conversation = session.messages
    .map((m) => {
      const speaker = m.speaker || m.role
      const ts = m.timestamp ? ` [${m.timestamp}]` : ""
      return `${speaker}${ts}: ${m.content}`
    })
    .join("\n")

  return `You are a memory extraction system. Extract all important information from this conversation into a structured format for later retrieval.

<context>
Conversation Date: ${date}
Participants: ${speakerA}, ${speakerB}
</context>

<conversation>
${conversation}
</conversation>

Extract information in this exact format:

<memories>
Write each memory as a standalone, self-contained fact. Anyone reading a single line should understand it without needing other lines for context.

Rules:
- Prefix every memory with the date in [YYYY-MM-DD] format
- Resolve ALL relative dates to absolute dates using conversation date ${date}
  "yesterday" = day before ${date}, "last week" = 7 days before, "next Friday" = calculate from ${date}, "two months ago" = calculate from ${date}
- If a message has an explicit timestamp, use that exact date
- Use actual names from the conversation, never "the user" or "the assistant"
- Include specific numbers, dates, locations, proper nouns
- One fact per line, no bullet points or dashes
- Cover: biographical facts, preferences, opinions, events, plans, decisions, activities, emotions, routines, skills, health, work, relationships
- For recurring events or habits, note the pattern (e.g., "every Tuesday", "weekly")
</memories>

<entities>
List each unique person, organization, location, or notable object mentioned.
One per line, pipe-delimited: name|type|one-sentence summary of all known facts

Types: person, organization, location, object

Use the entity's actual name. Summary should combine ALL known facts about this entity from the conversation.
Do not use the pipe character (|) within any field value. Use a comma or semicolon instead.
</entities>

<relationships>
List relationships between entities.
One per line, pipe-delimited: source|relation|target|date-or-timeframe

Relations: married_to, partner_of, parent_of, child_of, sibling_of, friend_of, colleague_of, works_at, lives_in, studies_at, member_of, owns, manages, reports_to, visited, likes, dislikes, etc.

Only include relationships explicitly stated or strongly implied. Include temporal info when known. Use entity names exactly as in the entities section.
</relationships>`
}

export function parseExtractionOutput(rawText: string): ParsedExtraction {
  const memoriesMatch = rawText.match(/<memories>([\s\S]*?)<\/memories>/)
  const memoriesText = memoriesMatch
    ? memoriesMatch[1].trim()
    : rawText
        .replace(/<entities>[\s\S]*?<\/entities>/g, "")
        .replace(/<relationships>[\s\S]*?<\/relationships>/g, "")
        .trim()

  const entities: ParsedExtraction["entities"] = []
  const entitiesMatch = rawText.match(/<entities>([\s\S]*?)<\/entities>/)
  if (entitiesMatch) {
    for (const line of entitiesMatch[1].trim().split("\n")) {
      if (!line.trim() || !line.includes("|")) continue
      const parts = line.split("|").map((p) => p.trim())
      if (parts.length >= 3 && parts[0] && parts[1] && parts[2]) {
        entities.push({ name: parts[0], type: parts[1], summary: parts.slice(2).join("|") })
      }
    }
  }

  const relationships: ParsedExtraction["relationships"] = []
  const relsMatch = rawText.match(/<relationships>([\s\S]*?)<\/relationships>/)
  if (relsMatch) {
    for (const line of relsMatch[1].trim().split("\n")) {
      if (!line.trim() || !line.includes("|")) continue
      const parts = line.split("|").map((p) => p.trim())
      if (parts.length >= 3 && parts[0] && parts[1] && parts[2]) {
        relationships.push({
          source: parts[0],
          relation: parts[1],
          target: parts[2],
          date: parts[3] || undefined,
        })
      }
    }
  }

  const eventDates = new Map<number, string>()
  const memoryLines = memoriesText.split("\n").filter((l) => l.trim())
  for (let i = 0; i < memoryLines.length; i++) {
    const dateMatch = memoryLines[i].match(/^\[(\d{4}-\d{2}-\d{2})\]/)
    if (dateMatch) {
      eventDates.set(i, dateMatch[1])
    }
  }

  return { memoriesText, entities, relationships, eventDates }
}

export async function extractMemories(
  openai: ReturnType<typeof createOpenAI>,
  session: UnifiedSession,
  maxRetries = 5
): Promise<string> {
  const prompt = buildExtractionPrompt(session)

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      if (attempt > 0) {
        logger.info(`[extraction] Retry ${attempt}/${maxRetries} for ${session.sessionId}`)
      }
      const { text } = await generateText({
        model: openai(EXTRACTION_MODEL),
        prompt,
      } as Parameters<typeof generateText>[0])
      return text.trim()
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e)
      const errCode = (e as any)?.code || (e as any)?.status || "unknown"
      logger.warn(
        `[extraction] ${session.sessionId} attempt ${attempt + 1}/${maxRetries} failed: [${errCode}] ${errMsg.substring(0, 150)}`
      )
      if (attempt === maxRetries - 1) throw e
      const delay = 2000 * Math.pow(2, attempt)
      logger.info(`[extraction] Waiting ${delay}ms before retry...`)
      await new Promise((r) => setTimeout(r, delay))
    }
  }
  throw new Error("unreachable")
}
