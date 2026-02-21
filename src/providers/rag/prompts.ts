import type { ProviderPrompts } from "../../types/prompts"

interface RAGSearchResult {
  content: string
  score: number
  vectorScore: number
  bm25Score: number
  rerankScore?: number
  sessionId: string
  chunkIndex: number
  date?: string
  eventDate?: string
  metadata?: Record<string, unknown>
  _type?: "chunk" | "entity" | "relationship" | "profile"
  name?: string
  entityType?: string
  source?: string
  target?: string
  relation?: string
}

function buildRAGContext(context: unknown[]): { chunks: string; graphContext: string; profileContext: string } {
  const results = context as RAGSearchResult[]

  const chunks: string[] = []
  const entities: string[] = []
  const relationships: string[] = []
  let profileContext = ""

  for (const result of results) {
    if (result._type === "profile") {
      profileContext = result.content
    } else if (result._type === "entity") {
      entities.push(`- ${result.name} (${result.entityType}): ${result.content}`)
    } else if (result._type === "relationship") {
      const dateStr = result.date ? ` [${result.date}]` : ""
      relationships.push(`- ${result.source} -${result.relation}-> ${result.target}${dateStr}`)
    } else {
      const date = result.date || (result.metadata?.memoryDate as string) || undefined
      const dateStr = date ? ` date="${date}"` : ""
      const eventDateStr = result.eventDate ? ` eventDate="${result.eventDate}"` : ""
      chunks.push(`<memory session="${result.sessionId}"${dateStr}${eventDateStr}>
${result.content}
</memory>`)
    }
  }

  let graphContext = ""
  if (entities.length > 0 || relationships.length > 0) {
    const parts: string[] = []
    if (entities.length > 0) {
      parts.push(`<entities>\n${entities.join("\n")}\n</entities>`)
    }
    if (relationships.length > 0) {
      parts.push(`<relationships>\n${relationships.join("\n")}\n</relationships>`)
    }
    graphContext = parts.join("\n")
  }

  return {
    chunks: chunks.length > 0 ? chunks.join("\n\n") : "No relevant memory chunks retrieved.",
    graphContext,
    profileContext,
  }
}

export function buildRAGAnswerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  const { chunks, graphContext, profileContext } = buildRAGContext(context)

  const profileSection = profileContext
    ? `\n${profileContext}\n`
    : ""

  const graphSection = graphContext
    ? `\n<knowledge_graph>\n${graphContext}\n</knowledge_graph>\n`
    : ""

  return `You are a question-answering system. Based on the retrieved context below, answer the question.

Question: ${question}
Question Date: ${questionDate || "Not specified"}
${profileSection}${graphSection}
<retrieved_memories>
${chunks}
</retrieved_memories>

**Understanding the Context:**
The context contains memories extracted from past conversations. Each memory has:
1. **Content**: extracted facts, preferences, events, and details from conversations
2. **Date**: when the event/conversation occurred (in YYYY-MM-DD format)
3. **User Profile** (if present): static biographical facts about the user (name, location, relationships, etc.)
4. **Knowledge Graph** (if present): entity relationships useful for multi-hop questions

**How to Answer:**

1. Base your answer ONLY on the provided context. Never use outside knowledge.

2. **Temporal questions** (how many days/weeks/months ago, what order, when):
   - The Question Date above is YOUR reference point. Calculate ALL relative time differences from the Question Date, NOT from today.
   - Example: If Question Date is 2023-06-22 and a memory is dated 2023-06-15, that event was 7 days ago.
   - Show your date math explicitly: "Event date: 2023-06-15. Question date: 2023-06-22. Difference: 7 days."
   - If the question asks about something that hasn't happened yet according to the memories, say so.

3. **Suggestion/preference questions** (recommend, suggest, any tips, what should I):
   - Use the retrieved memories to understand the user's preferences, experiences, and interests.
   - Synthesize personalized suggestions based on what you know about the user.
   - Example: If memories show the user owns a Sony camera, suggest Sony-compatible accessories.
   - Example: If memories show the user made a great beef stew in their slow cooker, use that as a basis for advice.
   - Do NOT say "I don't know" if the context contains relevant preferences or experiences you can build on.

4. **Knowledge-update questions** (information that changes over time):
   - When the same topic appears at different dates with different values, the MOST RECENT memory is the current state.
   - Earlier values are outdated. Answer with the latest value only.

5. **Counting/aggregation questions** (how many, how much, total):
   - List EACH item individually before giving a total. Do not estimate.
   - Cross-check: scan ALL provided memories for mentions, not just the first few.
   - If the question asks "how many X", enumerate every X you find, then count them.

6. **Multi-hop/relationship questions**:
   - Use the knowledge graph to trace entity connections.
   - Follow relationship chains (e.g., "What does X's wife do?" = find spouse, then find their job).

7. **When to say "I don't know"**:
   - ONLY when the context genuinely contains no relevant information about the topic.
   - If the question asks about entity X but memories only mention entity Y, that is not enough.
   - Do NOT say "I don't know" just because the question asks for a suggestion or opinion. If you have context about the user's preferences, use it.

**Response Format:**
Think step by step, then provide your answer.

Reasoning:
[Your step-by-step reasoning here. For temporal questions, show date calculations explicitly.]

Answer:
[Your final answer here. Be specific: use names, dates, numbers.]`
}

export const RAG_PROMPTS: ProviderPrompts = {
  answerPrompt: buildRAGAnswerPrompt,
}

export default RAG_PROMPTS
