import { createOpenAI } from "@ai-sdk/openai"
import { generateText } from "ai"
import { logger } from "../../utils/logger"

const PROFILE_MODEL = "gpt-5-nano"
const WORD_OVERLAP_THRESHOLD = 0.6

export interface UserProfile {
  facts: string[]
}

function emptyProfile(): UserProfile {
  return { facts: [] }
}

function extractWords(text: string): Set<string> {
  return new Set(
    text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(w => w.length > 1)
  )
}

function wordOverlap(a: string, b: string): number {
  const wordsA = extractWords(a)
  const wordsB = extractWords(b)
  if (wordsA.size === 0 || wordsB.size === 0) return 0

  let shared = 0
  for (const w of wordsA) {
    if (wordsB.has(w)) shared++
  }

  return shared / Math.min(wordsA.size, wordsB.size)
}

export function mergeProfiles(existing: UserProfile, incoming: UserProfile): UserProfile {
  const merged = [...existing.facts]

  for (const newFact of incoming.facts) {
    const overlapIdx = merged.findIndex(f => wordOverlap(f, newFact) >= WORD_OVERLAP_THRESHOLD)
    if (overlapIdx !== -1) {
      merged[overlapIdx] = newFact
    } else {
      merged.push(newFact)
    }
  }

  return { facts: merged }
}

export async function buildProfile(
  openai: ReturnType<typeof createOpenAI>,
  memoriesText: string,
  existingProfile?: UserProfile
): Promise<UserProfile> {
  if (!memoriesText.trim()) return existingProfile || emptyProfile()

  try {
    const { text } = await generateText({
      model: openai(PROFILE_MODEL),
      prompt: `Extract static/biographical facts about the primary user from the memories below. Only include facts that are clearly stated, not inferred.

Look for: name, age, birthday, location (city/country), occupation, employer, relationships (spouse, partner, children, parents, siblings, friends, pets), hobbies, key preferences.

Each fact should be a self-contained statement. One fact per line. No numbering, no bullet markers, no explanations.

Example output:
User's name is Sarah
Lives in San Francisco
Works at Google as a software engineer
Married to Tom
Has a dog named Max
Birthday is March 15

Memories:
${memoriesText}`,
    } as Parameters<typeof generateText>[0])

    const facts = text.trim().split("\n")
      .map(l => l.replace(/^[-*\d.)\s]+/, "").trim())
      .filter(l => l.length > 3 && l.length < 300)

    if (facts.length === 0) return existingProfile || emptyProfile()

    const incoming: UserProfile = { facts }

    if (existingProfile && existingProfile.facts.length > 0) {
      return mergeProfiles(existingProfile, incoming)
    }

    return incoming
  } catch (e) {
    logger.warn(`[profile] Extraction failed: ${e instanceof Error ? e.message : e}`)
    return existingProfile || emptyProfile()
  }
}

export function formatProfileContext(profile: UserProfile): string {
  if (!profile.facts.length) return ""

  const lines = profile.facts.map(f => `- ${f}`).join("\n")
  return `<user_profile>\n${lines}\n</user_profile>`
}
