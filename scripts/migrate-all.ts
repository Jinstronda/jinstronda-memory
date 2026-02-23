// Full migration: local workspace + GitHub repo -> RAG + mem0
// RAG gets chunks (6K), mem0 gets whole files (LLM extraction handles it)
// bun run scripts/migrate-all.ts

import { readdir, readFile } from "fs/promises"
import { join, basename, extname } from "path"
import { Database } from "bun:sqlite"

const RAG = "http://localhost:3847"
const MEM0 = "http://localhost:3848"
const TAG = "openclaw_Joaos_MacBook_Pro_local"
const HOME = process.env.HOME!
const LOCAL_MEM = join(HOME, ".openclaw/workspace/memory")
const LOCAL_WS = join(HOME, ".openclaw/workspace")
const GITHUB_MEM = "/tmp/my-clawdbot/workspace/memory"
const GITHUB_WS = "/tmp/my-clawdbot/workspace"
const CHROMA = join(HOME, ".openclaw/mem0/data/chroma/chroma.sqlite3")
const CHUNK_SIZE = 6000
const MEM0_MAX_TEXT = 50000  // mem0 can handle larger texts via LLM
const RAG_WORKERS = 5
const MEM0_WORKERS = 10
const MAX_RETRIES = 3

type Item = { sid: string; text: string; date?: string; label?: string }
type Chunk = { sid: string; text: string; date?: string; label?: string }

const stats = { ragOk: 0, ragFail: 0, mem0Ok: 0, mem0Fail: 0 }

async function retry<T>(fn: () => Promise<T>): Promise<T> {
  for (let i = 0; i < MAX_RETRIES; i++) {
    try { return await fn() }
    catch (e) {
      if (i === MAX_RETRIES - 1) throw e
      await new Promise(r => setTimeout(r, 2000 * (i + 1)))
    }
  }
  throw new Error("unreachable")
}

async function post(url: string, body: unknown): Promise<void> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 600_000)
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!res.ok) {
      const t = await res.text().catch(() => "")
      throw new Error(`${res.status}: ${t.slice(0, 200)}`)
    }
  } finally {
    clearTimeout(timer)
  }
}

function chunkForRag(text: string): string[] {
  if (text.length <= CHUNK_SIZE) return [text]
  const out: string[] = []
  let i = 0
  while (i < text.length) {
    let end = Math.min(i + CHUNK_SIZE, text.length)
    if (end < text.length) {
      const nl = text.lastIndexOf("\n", end)
      if (nl > i + CHUNK_SIZE * 0.4) end = nl + 1
    }
    out.push(text.slice(i, end))
    i = end
  }
  return out
}

// for mem0, just truncate if too long
function trimForMem0(text: string): string {
  return text.length <= MEM0_MAX_TEXT ? text : text.slice(0, MEM0_MAX_TEXT)
}

function getDate(f: string): string | undefined {
  return f.match(/^(\d{4}-\d{2}-\d{2})/)?.[1]
}

async function loadDir(dir: string, exts: string[]): Promise<{ path: string; name: string }[]> {
  try {
    const entries = await readdir(dir)
    return entries
      .filter(f => exts.includes(extname(f).toLowerCase()))
      .sort()
      .map(f => ({ path: join(dir, f), name: basename(f, extname(f)) }))
  } catch { return [] }
}

async function loadChromaFacts(): Promise<string[]> {
  try {
    const db = new Database(CHROMA, { readonly: true })
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
  } catch { return [] }
}

// RAG worker: processes chunks from shared index
async function ragWorker(chunks: Chunk[], idx: { v: number }) {
  while (true) {
    const i = idx.v++
    if (i >= chunks.length) break
    const c = chunks[i]
    try {
      await retry(() => post(`${RAG}/ingest`, {
        containerTag: TAG,
        sessionId: c.sid,
        messages: [{ role: "user", content: c.text }],
        ...(c.date ? { date: c.date } : {}),
      }))
      stats.ragOk++
    } catch (e) {
      stats.ragFail++
      console.log(`  FAIL rag [${c.label || c.sid}] ${e}`)
    }
  }
}

// mem0 worker: processes whole items from shared index
async function mem0Worker(items: Item[], idx: { v: number }) {
  while (true) {
    const i = idx.v++
    if (i >= items.length) break
    const item = items[i]
    try {
      const text = trimForMem0(item.text)
      await retry(() => post(`${MEM0}/add`, { text, user_id: TAG }))
      stats.mem0Ok++
    } catch (e) {
      stats.mem0Fail++
      console.log(`  FAIL mem0 [${item.label || item.sid}] ${e}`)
    }
  }
}

const MEM0_ONLY = process.argv.includes("--mem0-only")
const RAG_ONLY = process.argv.includes("--rag-only")

async function main() {
  if (!MEM0_ONLY) {
    const rh = await fetch(`${RAG}/health`).catch(() => null)
    if (!rh?.ok) { console.log("RAG not ready"); process.exit(1) }
  }
  if (!RAG_ONLY) {
    const mh = await fetch(`${MEM0}/health`).catch(() => null)
    if (!mh?.ok) { console.log("mem0 not ready"); process.exit(1) }
  }

  console.log(`RAG: ${RAG_WORKERS} workers, ${CHUNK_SIZE} chunk | mem0: ${MEM0_WORKERS} workers, whole files`)
  console.log(`RAG: ${RAG} | mem0: ${MEM0} | tag: ${TAG}\n`)

  const seen = new Set<string>()
  const h = (t: string) => Bun.hash(t.trim().toLowerCase()).toString(36)
  const items: Item[] = []
  let skipCount = 0

  // 1. Identity files
  const idFiles = ["MEMORY.md", "IDENTITY.md", "SOUL.md", "USER.md", "AGENTS.md", "HEARTBEAT.md", "HORMOZI.md", "TOOLS.md"]
  let idCount = 0
  for (const dir of [LOCAL_WS, GITHUB_WS]) {
    for (const f of idFiles) {
      try {
        const content = await readFile(join(dir, f), "utf-8")
        const key = h(content)
        if (seen.has(key)) continue
        seen.add(key)
        idCount++
        items.push({ sid: `id_${basename(f, ".md")}`, text: content, date: "2026-02-23", label: f })
      } catch {}
    }
  }
  console.log(`[1/4] ${idCount} identity files`)

  // 2. Local memory files
  const localFiles = await loadDir(LOCAL_MEM, [".md", ".txt"])
  let localCount = 0
  for (const f of localFiles) {
    try {
      const content = await readFile(f.path, "utf-8")
      const key = h(content)
      if (seen.has(key)) { skipCount++; continue }
      seen.add(key)
      localCount++
      items.push({ sid: `local_${f.name}`, text: content, date: getDate(basename(f.path)), label: f.name })
    } catch {}
  }
  console.log(`[2/4] ${localCount} local memory files`)

  // 3. GitHub repo memory files
  const ghFiles = await loadDir(GITHUB_MEM, [".md", ".txt"])
  let ghCount = 0
  for (const f of ghFiles) {
    try {
      const content = await readFile(f.path, "utf-8")
      const key = h(content)
      if (seen.has(key)) { skipCount++; continue }
      seen.add(key)
      ghCount++
      items.push({ sid: `gh_${f.name}`, text: content, date: getDate(basename(f.path)), label: `gh:${f.name}` })
    } catch {}
  }
  console.log(`[3/4] ${ghCount} GitHub memory files (${skipCount} dupes skipped)`)

  // 4. Chroma openclaw facts
  const facts = await loadChromaFacts()
  let factCount = 0
  for (let i = 0; i < facts.length; i++) {
    const key = h(facts[i])
    if (seen.has(key)) continue
    seen.add(key)
    factCount++
    items.push({ sid: `chroma_${i}`, text: facts[i], label: `chroma:${i}` })
  }
  console.log(`[4/4] ${factCount} Chroma facts`)

  // chunk items for RAG only
  const ragChunks: Chunk[] = []
  for (const item of items) {
    if (!item.text.trim()) continue
    const chunks = chunkForRag(item.text)
    for (let c = 0; c < chunks.length; c++) {
      ragChunks.push({
        sid: chunks.length === 1 ? item.sid : `${item.sid}_c${c}`,
        text: chunks[c],
        date: item.date,
        label: item.label,
      })
    }
  }

  const totalItems = idCount + localCount + ghCount + factCount
  console.log(`\nTotal: ${totalItems} items`)
  console.log(`  RAG: ${ragChunks.length} chunks (${RAG_WORKERS} workers)`)
  console.log(`  mem0: ${items.length} whole files (${MEM0_WORKERS} workers, serialized on server)\n`)

  const t0 = performance.now()

  const interval = setInterval(() => {
    const elapsed = ((performance.now() - t0) / 1000).toFixed(0)
    const ragPct = ((stats.ragOk / ragChunks.length) * 100).toFixed(1)
    const mem0Pct = ((stats.mem0Ok / items.length) * 100).toFixed(1)
    console.log(`  [${elapsed}s] rag: ${stats.ragOk}/${ragChunks.length} (${ragPct}%) ${stats.ragFail}f | mem0: ${stats.mem0Ok}/${items.length} (${mem0Pct}%) ${stats.mem0Fail}f`)
  }, 10_000)

  const ragIdx = { v: 0 }
  const mem0Idx = { v: 0 }

  const tasks: Promise<void[]>[] = []
  if (!MEM0_ONLY) tasks.push(Promise.all(Array.from({ length: RAG_WORKERS }, () => ragWorker(ragChunks, ragIdx))))
  if (!RAG_ONLY) tasks.push(Promise.all(Array.from({ length: MEM0_WORKERS }, () => mem0Worker(items, mem0Idx))))
  await Promise.all(tasks)

  clearInterval(interval)
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1)
  console.log(`\nDone in ${elapsed}s`)
  console.log(`  RAG:  ${stats.ragOk} ok, ${stats.ragFail} fail (${ragChunks.length} chunks)`)
  console.log(`  mem0: ${stats.mem0Ok} ok, ${stats.mem0Fail} fail (${items.length} items)`)
  console.log(`  Deduped: ${skipCount}`)
}

main()
