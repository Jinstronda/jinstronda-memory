// Migrate existing OpenClaw markdown memories into the RAG system.
// Run: bun run scripts/migrate-openclaw-memories.ts
// Requires RAG server running on localhost:3847.

import { readdir, readFile } from "fs/promises"
import { join, basename } from "path"

const RAG_URL = process.env.RAG_URL || "http://localhost:3847"
const CONTAINER_TAG = process.env.CONTAINER_TAG || "jinstronda"
const MEMORY_DIR = process.env.MEMORY_DIR || join(process.env.HOME!, ".openclaw/workspace/memory")
const MAIN_MEMORY = process.env.MAIN_MEMORY || join(process.env.HOME!, ".openclaw/workspace/MEMORY.md")

async function post(path: string, body: unknown): Promise<unknown> {
	const res = await fetch(`${RAG_URL}${path}`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	})
	if (!res.ok) {
		const text = await res.text()
		throw new Error(`${path} failed (${res.status}): ${text}`)
	}
	return res.json()
}

function extractDate(filename: string): string | undefined {
	const match = filename.match(/^(\d{4}-\d{2}-\d{2})/)
	return match ? match[1] : undefined
}

async function ingestFile(filepath: string, sessionId: string, date?: string): Promise<void> {
	const content = await readFile(filepath, "utf-8")
	if (!content.trim()) return

	const body: Record<string, unknown> = {
		containerTag: CONTAINER_TAG,
		sessionId: `migrate_${sessionId}`,
		messages: [{ role: "user", content }],
	}
	if (date) body.date = date

	await post("/ingest", body)
	console.log(`  ingested: ${sessionId}${date ? ` (${date})` : ""}`)
}

async function main() {
	const health = await fetch(`${RAG_URL}/health`).catch(() => null)
	if (!health?.ok) {
		console.error(`RAG server not reachable at ${RAG_URL}. Start it first.`)
		process.exit(1)
	}
	console.log(`RAG server OK at ${RAG_URL}`)
	console.log(`Container: ${CONTAINER_TAG}`)
	console.log()

	// ingest main MEMORY.md
	console.log("Ingesting MEMORY.md...")
	await ingestFile(MAIN_MEMORY, "MEMORY", "2026-02-19")

	// ingest all files in memory dir
	const files = await readdir(MEMORY_DIR)
	const mdFiles = files.filter((f) => f.endsWith(".md")).sort()

	console.log(`\nIngesting ${mdFiles.length} memory files from ${MEMORY_DIR} (concurrency: 5)...`)

	let ok = 0
	let failed = 0
	const CONCURRENCY = 5

	for (let i = 0; i < mdFiles.length; i += CONCURRENCY) {
		const batch = mdFiles.slice(i, i + CONCURRENCY)
		const results = await Promise.allSettled(
			batch.map(async (file) => {
				const name = basename(file, ".md")
				const date = extractDate(file)
				await ingestFile(join(MEMORY_DIR, file), name, date)
				return name
			})
		)
		for (const r of results) {
			if (r.status === "fulfilled") ok++
			else {
				console.error(`  FAILED: ${r.reason}`)
				failed++
			}
		}
		console.log(`  progress: ${Math.min(i + CONCURRENCY, mdFiles.length)}/${mdFiles.length}`)
	}

	console.log(`\nDone. ${ok} ingested, ${failed} failed.`)
}

main()
