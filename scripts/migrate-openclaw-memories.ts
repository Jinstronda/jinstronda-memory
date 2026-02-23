// Migrate all OpenClaw memories into the dual-layer system (RAG + mem0 + graph).
// Reads from: workspace markdown files + Chroma DB (openclaw-source facts only).
// Skips history.db (polluted with MemoryBench test data).
// Fully async, concurrent sends to both RAG and mem0.
//
// Run: bun run scripts/migrate-openclaw-memories.ts

import { readdir, readFile } from "fs/promises"
import { join, basename } from "path"
import { Database } from "bun:sqlite"

const RAG_URL = process.env.RAG_URL || "http://localhost:3847"
const MEM0_URL = process.env.MEM0_URL || "http://localhost:3848"
const CONTAINER_TAG = process.env.CONTAINER_TAG || "openclaw_Joaos_MacBook_Pro_local"
const USER_ID = process.env.USER_ID || "openclaw_Joaos_MacBook_Pro_local"
const HOME = process.env.HOME!
const MEMORY_DIR = join(HOME, ".openclaw/workspace/memory")
const WORKSPACE_DIR = join(HOME, ".openclaw/workspace")
const CHROMA_DB = join(HOME, ".openclaw/mem0/data/chroma/chroma.sqlite3")
const MAX_CONCURRENT = 100

let ragOk = 0, ragFail = 0, mem0Ok = 0, mem0Fail = 0, total = 0

class Semaphore {
  private count = 0
  private queue: (() => void)[] = []
  constructor(private max: number) {}
  async acquire() {
    if (this.count >= this.max)
      await new Promise<void>(r => this.queue.push(r))
    this.count++
  }
  release() {
    this.count--
    this.queue.shift()?.()
  }
}

const sem = new Semaphore(MAX_CONCURRENT)

async function postRAG(path: string, body: unknown) {
  const res = await fetch(`${RAG_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`RAG ${path}: ${res.status}`)
  return res.json()
}

async function postMem0(path: string, body: unknown) {
  const res = await fetch(`${MEM0_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`mem0 ${path}: ${res.status}`)
  return res.json()
}

async function ingestDual(sessionId: string, content: string, date?: string) {
  if (!content.trim()) return
  await sem.acquire()
  try {
    total++
    const ragBody: Record<string, unknown> = {
      containerTag: CONTAINER_TAG,
      sessionId,
      messages: [{ role: "user", content }],
    }
    if (date) ragBody.date = date

    const [r1, r2] = await Promise.allSettled([
      postRAG("/ingest", ragBody),
      postMem0("/add", { text: content, user_id: USER_ID, metadata: { source: "migration", session: sessionId } }),
    ])

    r1.status === "fulfilled" ? ragOk++ : (ragFail++, console.error(`  RAG fail [${sessionId}]`))
    r2.status === "fulfilled" ? mem0Ok++ : (mem0Fail++, console.error(`  mem0 fail [${sessionId}]`))

    if (total % 10 === 0) process.stdout.write(`  ${total} processed\r`)
  } finally {
    sem.release()
  }
}

function extractChromaOpenclawFacts(): string[] {
  try {
    const db = new Database(CHROMA_DB, { readonly: true })
    const rows = db.query<{ data: string }, []>(`
      SELECT em_data.string_value as data
      FROM embedding_metadata em_uid
      JOIN embedding_metadata em_data ON em_uid.id = em_data.id AND em_data.key = 'data'
      JOIN embedding_metadata em_src ON em_uid.id = em_src.id AND em_src.key = 'source'
      WHERE em_uid.key = 'user_id' AND em_uid.string_value = 'jinstronda'
      AND em_src.string_value = 'openclaw'
    `).all()
    db.close()
    return rows.map(r => r.data).filter(Boolean)
  } catch (e) {
    console.error(`Chroma read failed: ${e}`)
    return []
  }
}

function extractDate(filename: string): string | undefined {
  return filename.match(/^(\d{4}-\d{2}-\d{2})/)?.[1]
}

async function main() {
  const [rh, mh] = await Promise.all([
    fetch(`${RAG_URL}/health`).catch(() => null),
    fetch(`${MEM0_URL}/health`).catch(() => null),
  ])
  if (!rh?.ok) { console.error(`RAG unreachable at ${RAG_URL}`); process.exit(1) }
  if (!mh?.ok) { console.error(`mem0 unreachable at ${MEM0_URL}`); process.exit(1) }

  console.log(`RAG: ${RAG_URL} | mem0: ${MEM0_URL}`)
  console.log(`container: ${CONTAINER_TAG} | concurrency: ${MAX_CONCURRENT}`)

  const seen = new Set<string>()
  const hash = (t: string) => Bun.hash(t.trim().toLowerCase()).toString(36)
  const tasks: Promise<void>[] = []

  // 1. identity files
  const identityFiles = ["MEMORY.md", "IDENTITY.md", "SOUL.md", "USER.md", "AGENTS.md"]
  let identityCount = 0
  for (const file of identityFiles) {
    try {
      const content = await readFile(join(WORKSPACE_DIR, file), "utf-8")
      const h = hash(content)
      if (seen.has(h)) continue
      seen.add(h)
      identityCount++
      tasks.push(ingestDual(`identity_${basename(file, ".md")}`, content, "2026-02-23"))
    } catch {}
  }
  console.log(`\n[1/3] ${identityCount} identity files queued`)

  // 2. memory markdown files
  const memFiles = (await readdir(MEMORY_DIR)).filter(f => f.endsWith(".md")).sort()
  let memCount = 0
  for (const file of memFiles) {
    try {
      const content = await readFile(join(MEMORY_DIR, file), "utf-8")
      const h = hash(content)
      if (seen.has(h)) continue
      seen.add(h)
      memCount++
      tasks.push(ingestDual(`memory_${basename(file, ".md")}`, content, extractDate(file)))
    } catch {}
  }
  console.log(`[2/3] ${memCount} memory files queued`)

  // 3. chroma openclaw facts (not migration, those are covered by files)
  const facts = extractChromaOpenclawFacts()
  let factCount = 0
  for (let i = 0; i < facts.length; i++) {
    const h = hash(facts[i])
    if (seen.has(h)) continue
    seen.add(h)
    factCount++
    tasks.push(ingestDual(`chroma_${i}`, facts[i]))
  }
  console.log(`[3/3] ${factCount} unique Chroma openclaw facts queued (${facts.length - factCount} deduped)`)

  console.log(`\nTotal: ${tasks.length} items, firing all async...\n`)
  const t0 = performance.now()

  await Promise.all(tasks)

  const elapsed = ((performance.now() - t0) / 1000).toFixed(1)
  console.log(`\nDone in ${elapsed}s`)
  console.log(`  RAG:  ${ragOk} ok, ${ragFail} failed`)
  console.log(`  mem0: ${mem0Ok} ok, ${mem0Fail} failed`)
  console.log(`  Unique items: ${seen.size}`)
}

main()
