import { log } from "./logger.ts"
import { sanitizeContent, sanitizeMetadata } from "./sanitize.ts"

const RELEVANCE_WEIGHT = 0.4
const RECENCY_WEIGHT = 0.35
const IMPORTANCE_WEIGHT = 0.25
const RECENCY_HALF_LIFE_MS = 7 * 86400000 // 7 days

const IMPORTANCE_BY_TYPE: Record<string, number> = {
	preference: 0.9,
	decision: 0.85,
	entity: 0.7,
	fact: 0.6,
	other: 0.4,
}

export type SearchResult = {
	id: string
	content: string
	memory?: string
	similarity?: number
	score?: number
	metadata?: Record<string, unknown>
}

export type ProfileSearchResult = {
	memory?: string
	updatedAt?: string
	similarity?: number
	[key: string]: unknown
}

export type ProfileResult = {
	static: string[]
	dynamic: string[]
	searchResults: ProfileSearchResult[]
}

function limitText(text: string, max: number): string {
	return text.length > max ? `${text.slice(0, max)}...` : text
}

function recencyDecay(ageMs: number): number {
	return Math.exp((-Math.LN2 * ageMs) / RECENCY_HALF_LIFE_MS)
}

function importanceScore(metadata?: Record<string, unknown>): number {
	const memType = (metadata?.type as string) ?? "other"
	return IMPORTANCE_BY_TYPE[memType] ?? 0.4
}

export function scoreMemory(result: SearchResult, now = Date.now()): number {
	const relevance = result.similarity ?? 0.5
	const ts =
		(result.metadata?.updated_at as string) ??
		(result.metadata?.created_at as string)
	const ageMs = ts ? now - new Date(ts).getTime() : Number.POSITIVE_INFINITY
	const recency = Number.isFinite(ageMs) ? recencyDecay(ageMs) : 0
	const importance = importanceScore(result.metadata)
	return (
		RELEVANCE_WEIGHT * relevance +
		RECENCY_WEIGHT * recency +
		IMPORTANCE_WEIGHT * importance
	)
}

export class Mem0Client {
	private baseUrl: string
	private userId: string

	constructor(mem0Url: string, userId: string) {
		this.baseUrl = mem0Url.replace(/\/$/, "")
		this.userId = userId
		log.info(`initialized (url: ${this.baseUrl}, user: ${userId})`)
	}

	private async request(
		method: string,
		path: string,
		body?: unknown,
	): Promise<unknown> {
		const url = `${this.baseUrl}${path}`
		const opts: RequestInit = {
			method,
			headers: { "Content-Type": "application/json" },
		}
		if (body) opts.body = JSON.stringify(body)

		const res = await fetch(url, opts)
		if (!res.ok) {
			const text = await res.text().catch(() => "")
			throw new Error(`mem0 ${method} ${path} failed: ${res.status} ${text}`)
		}
		return res.json()
	}

	async addMemory(
		content: string,
		metadata?: Record<string, string | number | boolean>,
		_customId?: string,
		userId?: string,
	): Promise<{ id: string }> {
		const uid = userId ?? this.userId
		const cleaned = sanitizeContent(content)
		const cleanMeta = metadata ? sanitizeMetadata(metadata) : undefined

		log.debugRequest("add", {
			contentLength: cleaned.length,
			metadata: cleanMeta,
			userId: uid,
		})

		const result = (await this.request("POST", "/v1/memories/", {
			messages: cleaned,
			user_id: uid,
			metadata: cleanMeta,
			infer: true,
		})) as { ids?: string[]; results?: Array<{ id?: string }> }

		const id = result.ids?.[0] ?? result.results?.[0]?.id ?? ""
		log.debugResponse("add", { id })
		return { id }
	}

	async search(
		query: string,
		limit = 5,
		userId?: string,
	): Promise<SearchResult[]> {
		const uid = userId ?? this.userId

		log.debugRequest("search", { query, limit, userId: uid })

		const response = (await this.request("POST", "/v1/memories/search/", {
			query,
			user_id: uid,
			top_k: limit,
		})) as { results?: Array<Record<string, unknown>> }

		const raw = response.results ?? []
		const results: SearchResult[] = raw.map((r) => ({
			id: (r.id as string) ?? "",
			content: (r.memory as string) ?? "",
			memory: r.memory as string | undefined,
			similarity: r.score as number | undefined,
			metadata: r.metadata as Record<string, unknown> | undefined,
		}))

		log.debugResponse("search", { count: results.length })
		return results
	}

	async searchMultiple(
		query: string,
		limit: number,
		namespaces: string[],
	): Promise<SearchResult[]> {
		const unique = [...new Set(namespaces)]
		const all = await Promise.all(
			unique.map((ns) => this.search(query, limit, ns)),
		)

		const seen = new Set<string>()
		const merged: SearchResult[] = []
		const now = Date.now()

		for (const batch of all) {
			for (const r of batch) {
				const key = r.id || r.content
				if (seen.has(key)) continue
				seen.add(key)
				r.score = scoreMemory(r, now)
				merged.push(r)
			}
		}

		merged.sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
		return merged.slice(0, limit)
	}

	async getProfile(query?: string, userId?: string): Promise<ProfileResult> {
		const uid = userId ?? this.userId

		log.debugRequest("profile", { userId: uid, query })

		// mem0 has no profile endpoint; build from search + list
		const [searchResults, allMemories] = await Promise.all([
			query
				? this.search(query, 15, uid)
				: Promise.resolve([] as SearchResult[]),
			this.listAll(uid, 50),
		])

		const now = Date.now()
		const staticFacts: string[] = []
		const dynamicFacts: string[] = []

		// score and sort all memories by weighted formula
		const scored = allMemories
			.filter((mem) => mem.memory ?? mem.content)
			.map((mem) => {
				const result: SearchResult = {
					id: mem.id,
					content: mem.content,
					memory: mem.memory,
					similarity: 0.5, // neutral relevance for list results
					metadata: mem.metadata,
				}
				return {
					text: mem.memory ?? mem.content ?? "",
					score: scoreMemory(result, now),
					metadata: mem.metadata,
				}
			})
			.sort((a, b) => b.score - a.score)

		for (const mem of scored) {
			const ts =
				(mem.metadata?.updated_at as string) ??
				(mem.metadata?.created_at as string)
			const age = ts ? now - new Date(ts).getTime() : Number.POSITIVE_INFINITY
			// high recency (< 7 days) = dynamic, else static
			if (Number.isFinite(age) && age < RECENCY_HALF_LIFE_MS) {
				dynamicFacts.push(mem.text)
			} else {
				staticFacts.push(mem.text)
			}
		}

		const profileSearch: ProfileSearchResult[] = searchResults.map((r) => ({
			memory: r.memory ?? r.content,
			updatedAt: r.metadata?.updated_at as string | undefined,
			similarity: r.similarity,
		}))

		log.debugResponse("profile", {
			staticCount: staticFacts.length,
			dynamicCount: dynamicFacts.length,
			searchCount: profileSearch.length,
		})

		return {
			static: staticFacts,
			dynamic: dynamicFacts,
			searchResults: profileSearch,
		}
	}

	async deleteMemory(
		id: string,
		userId?: string,
	): Promise<{ id: string; forgotten: boolean }> {
		const uid = userId ?? this.userId
		log.debugRequest("delete", { id, userId: uid })
		await this.request("DELETE", `/v1/memories/${id}`)
		log.debugResponse("delete", { id })
		return { id, forgotten: true }
	}

	async forgetByQuery(
		query: string,
		userId?: string,
	): Promise<{ success: boolean; message: string }> {
		log.debugRequest("forgetByQuery", { query, userId })

		const results = await this.search(query, 5, userId)
		if (results.length === 0) {
			return { success: false, message: "No matching memory found to forget." }
		}

		const target = results[0]
		await this.deleteMemory(target.id)

		const preview = limitText(target.content || target.memory || "", 100)
		return { success: true, message: `Forgot: "${preview}"` }
	}

	async wipeAllMemories(): Promise<{ deletedCount: number }> {
		log.debugRequest("wipe", { userId: this.userId })

		// count before reset
		const all = await this.listAll(this.userId, 1000)
		const count = all.length

		await this.request(
			"DELETE",
			`/v1/memories/?user_id=${encodeURIComponent(this.userId)}`,
		)

		log.debugResponse("wipe", { deletedCount: count })
		return { deletedCount: count }
	}

	getUserId(): string {
		return this.userId
	}

	// kept for interface compat with tools/commands that call getContainerTag
	getContainerTag(): string {
		return this.userId
	}

	private async listAll(
		userId: string,
		limit: number,
	): Promise<
		Array<{
			id: string
			content: string
			memory?: string
			metadata?: Record<string, unknown>
		}>
	> {
		const response = (await this.request(
			"GET",
			`/v1/memories/?user_id=${encodeURIComponent(userId)}&limit=${limit}`,
		)) as { results?: Array<Record<string, unknown>> }

		return (response.results ?? []).map((r) => ({
			id: (r.id as string) ?? "",
			content: (r.memory as string) ?? "",
			memory: r.memory as string | undefined,
			metadata: r.metadata as Record<string, unknown> | undefined,
		}))
	}
}
