// RAG HTTP server. Run with: bun run src/server.ts
// Started as subprocess by OpenClaw plugin or standalone.
// Cache dir configurable via RAG_CACHE_DIR env (defaults to ./data/cache/rag).

import { readdir } from "fs/promises"
import { join } from "path"
import RAGProvider from "./providers/rag/index"
import type { UnifiedSession, UnifiedMessage } from "./types/unified"

const PORT = parseInt(process.env.RAG_PORT || "3847", 10)

const provider = new RAGProvider()

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  })
}

function errResponse(message: string, status = 500): Response {
  return json({ error: message }, status)
}

async function parseBody<T>(req: Request): Promise<T> {
  return (await req.json()) as T
}

async function handleIngest(req: Request): Promise<Response> {
  const body = await parseBody<{
    containerTag: string
    sessionId: string
    messages: Array<{ role: "user" | "assistant"; content: string }>
    date?: string
  }>(req)

  if (!body.containerTag || !body.sessionId || !body.messages?.length) {
    return errResponse("containerTag, sessionId, and messages required", 400)
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(body.containerTag)) return errResponse("Invalid containerTag", 400)

  const messages: UnifiedMessage[] = body.messages.map((m) => ({
    role: m.role,
    content: m.content,
  }))

  const session: UnifiedSession = {
    sessionId: body.sessionId,
    messages,
    metadata: body.date ? { date: body.date } : undefined,
  }

  const result = await provider.ingest([session], { containerTag: body.containerTag })
  return json({ documentIds: result.documentIds })
}

async function handleSearch(req: Request): Promise<Response> {
  const body = await parseBody<{
    containerTag: string
    query: string
    limit?: number
  }>(req)

  if (!body.containerTag || !body.query) {
    return errResponse("containerTag and query required", 400)
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(body.containerTag)) return errResponse("Invalid containerTag", 400)

  const raw = await provider.search(body.query, {
    containerTag: body.containerTag,
    limit: body.limit,
  })

  const results: Array<{ content: string; score: number; type: string }> = []
  for (const r of raw) {
    const item = r as Record<string, unknown>
    results.push({
      content: (item.content as string) || "",
      score: (item.score as number) || 0,
      type: (item._type as string) || "chunk",
    })
  }

  return json({ results })
}

async function handleStore(req: Request): Promise<Response> {
  const body = await parseBody<{ containerTag: string; text: string }>(req)

  if (!body.containerTag || !body.text) {
    return errResponse("containerTag and text required", 400)
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(body.containerTag)) return errResponse("Invalid containerTag", 400)

  const session: UnifiedSession = {
    sessionId: `manual_${Date.now()}`,
    messages: [{ role: "user", content: body.text }],
  }

  await provider.ingest([session], { containerTag: body.containerTag })
  return json({ ok: true })
}

async function handleClear(containerTag: string): Promise<Response> {
  await provider.clear(containerTag)
  return json({ ok: true })
}

async function init(): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) {
    console.error("OPENAI_API_KEY is required")
    process.exit(1)
  }
  await provider.initialize({ apiKey })
}

await init()

Bun.serve({
  port: PORT,

  async fetch(req) {
    const url = new URL(req.url)

    if (req.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS })
    }

    try {
      if (req.method === "GET" && url.pathname === "/health") {
        return json({ ok: true, provider: "rag" })
      }

      if (req.method === "GET" && url.pathname === "/containers") {
        const cacheDir = process.env.RAG_CACHE_DIR || join(process.cwd(), "data/cache/rag")
        const entries = await readdir(cacheDir).catch(() => [] as string[])
        const containers = entries.filter(e => !e.includes("-v1.") && !e.includes("_abs-"))
        return json({ containers })
      }

      if (req.method === "POST" && url.pathname === "/ingest") {
        return await handleIngest(req)
      }

      if (req.method === "POST" && url.pathname === "/search") {
        return await handleSearch(req)
      }

      if (req.method === "POST" && url.pathname === "/store") {
        return await handleStore(req)
      }

      const clearMatch = url.pathname.match(/^\/clear\/(.+)$/)
      if (req.method === "DELETE" && clearMatch) {
        const tag = decodeURIComponent(clearMatch[1])
        if (!/^[a-zA-Z0-9_-]+$/.test(tag)) return errResponse("Invalid containerTag", 400)
        return await handleClear(tag)
      }

      return errResponse("Not found", 404)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      console.error(`[rag-server] ${req.method} ${url.pathname}: ${msg}`)
      return errResponse(msg)
    }
  },
})

console.log(`[rag-server] Listening on http://localhost:${PORT}`)
