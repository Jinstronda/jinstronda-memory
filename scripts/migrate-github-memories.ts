// Migrate ALL memory files from the my-clawdbot GitHub repo into dual-layer system.
// Handles large files by chunking. Fully async.
// Run: bun run scripts/migrate-github-memories.ts
// Prereqs: clone repo to /tmp/my-clawdbot, both servers running.

import { readdir, readFile, stat } from "fs/promises"
import { join, basename, extname } from "path"

const RAG_URL = process.env.RAG_URL || "http://localhost:3847"
const MEM0_URL = process.env.MEM0_URL || "http://localhost:3848"
const CONTAINER_TAG = process.env.CONTAINER_TAG || "openclaw_Joaos_MacBook_Pro_local"
const USER_ID = process.env.USER_ID || "openclaw_Joaos_MacBook_Pro_local"
const REPO_MEMORY = process.env.REPO_MEMORY || "/tmp/my-clawdbot/workspace/memory"
const REPO_WORKSPACE = process.env.REPO_WORKSPACE || "/tmp/my-clawdbot/workspace"
const MAX_CHUNK = 6000
const RAG_CONCURRENCY = 100
const MEM0_CONCURRENCY = 100

let ragOk = 0, ragFail = 0, mem0Ok = 0, mem0Fail = 0, total = 0

class Semaphore {
  private count = 0
  private queue: (() => void)[] = []
  constructor(private max: number) {}
  async acquire() {
    if (this.count >= this.max) await new Promise<void>(r => this.queue.push(r))
    this.count++
  }
  release() { this.count--; this.queue.shift()?.() }
}

const ragSem = new Semaphore(RAG_CONCURRENCY)
const mem0Sem = new Semaphore(MEM0_CONCURRENCY)

async function postRAG(body: unknown) {
  await ragSem.acquire()
  try {
    const res = await fetch(`${RAG_URL}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error(`${res.status}`)
    ragOk++
  } catch (e) {
    ragFail++
  } finally {
    ragSem.release()
  }
}

async function postMem0(body: unknown) {
  await mem0Sem.acquire()
  try {
    const res = await fetch(`${MEM0_URL}/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error(`${res.status}`)
    mem0Ok++
  } catch (e) {
    mem0Fail++
  } finally {
    mem0Sem.release()
  }
}

function chunkText(text: string, maxLen: number): string[] {
  if (text.length <= maxLen) return [text]
  const chunks: string[] = []
  let start = 0
  while (start < text.length) {
    let end = Math.min(start + maxLen, text.length)
    if (end < text.length) {
      const nl = text.lastIndexOf("\n", end)
      if (nl > start + maxLen * 0.5) end = nl + 1
    }
    chunks.push(text.slice(start, end))
    start = end
  }
  return chunks
}

function extractDate(filename: string): string | undefined {
  return filename.match(/^(\d{4}-\d{2}-\d{2})/)?.[1]
}

async function ingestFile(filepath: string, sessionBase: string, date?: string) {
  const content = await readFile(filepath, "utf-8")
  if (!content.trim()) return

  const chunks = chunkText(content, MAX_CHUNK)
  const tasks: Promise<void>[] = []

  for (let i = 0; i < chunks.length; i++) {
    const sessionId = chunks.length === 1 ? sessionBase : `${sessionBase}_chunk${i}`
    const chunk = chunks[i]
    total++

    const ragBody: Record<string, unknown> = {
      containerTag: CONTAINER_TAG,
      sessionId: `gh_${sessionId}`,
      messages: [{ role: "user", content: chunk }],
    }
    if (date) ragBody.date = date

    tasks.push(postRAG(ragBody))
    tasks.push(postMem0({ text: chunk, user_id: USER_ID, metadata: { source: "github_migration", file: basename(filepath) } }))
  }

  await Promise.all(tasks)
  if (total % 20 === 0) process.stdout.write(`  ${total} chunks processed\r`)
}

async function collectFiles(dir: string, ext: string[]): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true })
  const files: string[] = []
  for (const e of entries) {
    if (e.isFile() && ext.includes(extname(e.name).toLowerCase())) {
      files.push(join(dir, e.name))
    }
  }
  return files.sort()
}

async function main() {
  const [rh, mh] = await Promise.all([
    fetch(`${RAG_URL}/health`).catch(() => null),
    fetch(`${MEM0_URL}/health`).catch(() => null),
  ])
  if (!rh?.ok) { console.error(`RAG unreachable`); process.exit(1) }
  if (!mh?.ok) { console.error(`mem0 unreachable`); process.exit(1) }

  console.log(`RAG: ${RAG_URL} | mem0: ${MEM0_URL}`)
  console.log(`container: ${CONTAINER_TAG}`)
  console.log(`RAG concurrency: ${RAG_CONCURRENCY} | mem0 concurrency: ${MEM0_CONCURRENCY}`)
  console.log(`chunk size: ${MAX_CHUNK} chars\n`)

  // identity files from workspace root
  const identityFiles = ["MEMORY.md", "IDENTITY.md", "SOUL.md", "USER.md", "AGENTS.md", "HEARTBEAT.md", "HORMOZI.md", "TOOLS.md"]
  const identityTasks: Promise<void>[] = []
  for (const f of identityFiles) {
    const fp = join(REPO_WORKSPACE, f)
    try {
      await stat(fp)
      identityTasks.push(ingestFile(fp, `identity_${basename(f, extname(f))}`, "2026-02-23"))
    } catch {}
  }
  console.log(`[1/2] ${identityTasks.length} identity files queued`)

  // all memory files
  const memoryFiles = await collectFiles(REPO_MEMORY, [".md", ".txt"])
  const memoryTasks = memoryFiles.map(f => {
    const name = basename(f, extname(f))
    const date = extractDate(basename(f))
    return ingestFile(f, name, date)
  })
  console.log(`[2/2] ${memoryFiles.length} memory files queued`)

  const allTasks = [...identityTasks, ...memoryTasks]
  console.log(`\nTotal files: ${allTasks.length}, firing all async...\n`)

  const t0 = performance.now()
  await Promise.all(allTasks)
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1)

  console.log(`\n\nDone in ${elapsed}s`)
  console.log(`  Total chunks: ${total}`)
  console.log(`  RAG:  ${ragOk} ok, ${ragFail} failed`)
  console.log(`  mem0: ${mem0Ok} ok, ${mem0Fail} failed`)
}

main()
