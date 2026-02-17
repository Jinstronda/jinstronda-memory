import type { ProviderPrompts } from "../../types/prompts"

interface FilesystemResult {
  sessionId: string
  content: string
  score: number
  matchCount: number
}

function buildFilesystemContext(context: unknown[]): string {
  const results = context as FilesystemResult[]

  if (results.length === 0) {
    return "No relevant conversation sessions were found."
  }

  return results
    .map((result, i) => {
      const header = `=== Session ${i + 1}: ${result.sessionId} (relevance: ${(result.score * 100).toFixed(0)}%) ===`
      return `${header}\n${result.content}`
    })
    .join("\n\n---\n\n")
}

export function buildFilesystemAnswerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  const retrievedContext = buildFilesystemContext(context)

  return `You are a question-answering system. You have access to raw conversation transcripts stored as plain text files. Based on the retrieved sessions below, answer the question.

Question: ${question}
Question Date: ${questionDate || "Not specified"}

Retrieved Conversation Sessions:
${retrievedContext}

**Understanding the Context:**
The context above contains raw conversation transcripts from a file-based memory system (similar to CLAUDE.md / MEMORY.md). Each session is a complete conversation with timestamps and speaker labels.

**How to Answer:**
1. Carefully read through each conversation session
2. Look for specific mentions of facts, events, dates, names, preferences, and details relevant to the question
3. Pay attention to speaker names and roles to understand who said what
4. Consider temporal information - dates on sessions help establish when things happened
5. If multiple sessions contain relevant info, synthesize across them
6. For time-based questions, calculate relative dates based on the session dates, not the current date

Instructions:
- Base your answer ONLY on the provided conversation sessions
- Read the raw text carefully - the answer is embedded in the conversation flow
- If the sessions contain enough information, provide a clear, concise answer
- If the sessions do not contain enough information, respond with "I don't know"
- Pay attention to context and nuance in the conversations

Reasoning:
[Your step-by-step reasoning process here]

Answer:
[Your final answer here]`
}

export const FILESYSTEM_PROMPTS: ProviderPrompts = {
  answerPrompt: buildFilesystemAnswerPrompt,
}

export default FILESYSTEM_PROMPTS
