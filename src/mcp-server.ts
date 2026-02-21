#!/usr/bin/env bun
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js"
import { z } from "zod"

const RAG_URL = process.env.RAG_URL || "http://localhost:3847"
const PERSONAL_CONTAINER = process.env.RAG_CONTAINER || "jinstronda"

let activeRepoContainer: string | null = null

async function ragPost(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(`${RAG_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`RAG ${path} failed (${res.status}): ${text}`)
  }
  return res.json()
}

async function ragGet(path: string): Promise<unknown> {
  const res = await fetch(`${RAG_URL}${path}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`RAG ${path} failed (${res.status}): ${text}`)
  }
  return res.json()
}

type SearchResult = { content: string; score: number; type: string }
type SearchResponse = { results: SearchResult[]; profile?: string }

async function searchContainer(containerTag: string, query: string, limit: number): Promise<SearchResponse> {
  return ragPost("/search", { containerTag, query, limit }) as Promise<SearchResponse>
}

const server = new McpServer({
  name: "rag-memory",
  version: "2.0.0",
})

server.tool(
  "memory_search",
  "Search long-term memory (personal + repo). USE PROACTIVELY: at session start to load context, before architecture decisions to check past approaches, when user references anything from the past. Returns conversation chunks, entity graph, relationships, and user profile. Use natural language queries.",
  {
    query: z.string().describe("The search query"),
    limit: z.number().optional().default(10).describe("Max results per container"),
    scope: z.enum(["all", "personal", "repo"]).optional().default("all").describe("Search scope: all (personal + repo), personal only, or repo only"),
  },
  async ({ query, limit, scope }) => {
    const containers: string[] = []
    if (scope !== "repo") containers.push(PERSONAL_CONTAINER)
    if (scope !== "personal" && activeRepoContainer) containers.push(activeRepoContainer)

    if (containers.length === 0) {
      return { content: [{ type: "text" as const, text: "No active containers. Use memory_use_repo to set a repo memory, or search with scope: 'personal'." }] }
    }

    const responses = await Promise.all(containers.map(c => searchContainer(c, query, limit)))

    let text = ""
    for (let i = 0; i < containers.length; i++) {
      const { results, profile } = responses[i]
      if (containers.length > 1) text += `## ${containers[i]}\n\n`
      if (profile) text += `${profile}\n\n`
      if (results.length === 0) {
        text += "No results.\n\n"
        continue
      }
      for (const r of results) {
        const label = r.type === "chunk" ? "" : `[${r.type}] `
        text += `${label}${r.content}\n\n---\n\n`
      }
    }
    return { content: [{ type: "text" as const, text: text.trim() }] }
  }
)

server.tool(
  "memory_store",
  "Store a memory for later retrieval. USE WHEN: you solve a hard bug, discover an important pattern, make a key decision, or context is getting long and compaction is near. Include dates. Be concise but searchable. Store things that would be annoying to rediscover.",
  {
    content: z.string().describe("The content to store"),
    target: z.enum(["auto", "personal", "repo"]).optional().default("auto").describe("Where to store: auto (repo if active, else personal), personal, or repo"),
    sessionId: z.string().optional().describe("Session ID (auto-generated if omitted)"),
    date: z.string().optional().describe("Date in YYYY-MM-DD format"),
  },
  async ({ content, target, sessionId, date }) => {
    let container: string
    if (target === "personal") {
      container = PERSONAL_CONTAINER
    } else if (target === "repo") {
      if (!activeRepoContainer) {
        return { content: [{ type: "text" as const, text: "No active repo. Use memory_use_repo first." }] }
      }
      container = activeRepoContainer
    } else {
      container = activeRepoContainer || PERSONAL_CONTAINER
    }

    const sid = sessionId || `claude_${Date.now()}`
    const body: Record<string, unknown> = {
      containerTag: container,
      sessionId: sid,
      messages: [{ role: "user", content }],
    }
    if (date) body.date = date

    await ragPost("/ingest", body)
    return { content: [{ type: "text" as const, text: `Stored in "${container}" (session: ${sid})` }] }
  }
)

server.tool(
  "memory_use_repo",
  "Set active repo memory. CALL THIS AT THE START OF EVERY SESSION with the current project name. Each repo gets isolated memory. Auto-creates on first store. After calling this, memory_search queries both personal and repo memory.",
  {
    repoName: z.string().describe("Repository or project name (e.g., 'memorybench', 'robo-advisor', 'smart-agents')"),
  },
  async ({ repoName }) => {
    const tag = `repo-${repoName.toLowerCase().replace(/[^a-z0-9-]/g, "-")}`
    activeRepoContainer = tag
    return { content: [{ type: "text" as const, text: `Active repo memory: "${tag}". Search now queries both personal ("${PERSONAL_CONTAINER}") and repo ("${tag}").` }] }
  }
)

server.tool(
  "memory_list_repos",
  "List all known memory containers. Shows personal memory and all repo-specific memories with which one is currently active.",
  {},
  async () => {
    const result = await ragGet("/containers") as { containers: string[] }
    const personal = result.containers.filter(c => !c.startsWith("repo-"))
    const repos = result.containers.filter(c => c.startsWith("repo-"))

    let text = `Active repo: ${activeRepoContainer || "(none)"}\n\n`
    text += `Personal: ${personal.join(", ") || "(none)"}\n`
    text += `Repos: ${repos.join(", ") || "(none)"}`
    return { content: [{ type: "text" as const, text }] }
  }
)

server.tool(
  "memory_profile",
  "Get user's biographical profile: name, location, job, relationships, preferences. Extracted from personal memories. Use at session start to personalize responses.",
  {},
  async () => {
    const result = await ragGet(`/profile/${encodeURIComponent(PERSONAL_CONTAINER)}`) as { facts: string[] }

    if (!result.facts || result.facts.length === 0) {
      return { content: [{ type: "text" as const, text: "No profile data found." }] }
    }

    const text = result.facts.map(f => `- ${f}`).join("\n")
    return { content: [{ type: "text" as const, text }] }
  }
)

const transport = new StdioServerTransport()
await server.connect(transport)
