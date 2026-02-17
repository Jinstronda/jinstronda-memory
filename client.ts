import { log } from "./logger.ts"

export type SearchResult = {
	id: string
	content: string
	memory?: string
	similarity?: number
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

		log.debugRequest("add", {
			contentLength: content.length,
			metadata,
			userId: uid,
		})

		const result = (await this.request("POST", "/v1/memories/", {
			messages: content,
			user_id: uid,
			metadata: metadata ?? undefined,
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

	async getProfile(
		query?: string,
		userId?: string,
	): Promise<ProfileResult> {
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
		const DAY_MS = 86400000
		const staticFacts: string[] = []
		const dynamicFacts: string[] = []

		for (const mem of allMemories) {
			const text = mem.memory ?? mem.content ?? ""
			if (!text) continue

			const updatedAt = mem.metadata?.updated_at as string | undefined
			const createdAt = mem.metadata?.created_at as string | undefined
			const ts = updatedAt ?? createdAt
			const age = ts ? now - new Date(ts).getTime() : Infinity

			// recent (last 7 days) = dynamic, older = static
			if (age < 7 * DAY_MS) {
				dynamicFacts.push(text)
			} else {
				staticFacts.push(text)
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
		_userId?: string,
	): Promise<{ id: string; forgotten: boolean }> {
		log.debugRequest("delete", { id })
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

		await this.request("DELETE", `/v1/memories/?user_id=${encodeURIComponent(this.userId)}`)

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
	): Promise<Array<{ id: string; content: string; memory?: string; metadata?: Record<string, unknown> }>> {
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
