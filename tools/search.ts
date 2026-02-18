import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import { stringEnum } from "openclaw/plugin-sdk"
import type { Mem0Client, SearchResult } from "../client.ts"
import type { Mem0Config } from "../config.ts"
import { log } from "../logger.ts"

export function registerSearchTool(
	api: OpenClawPluginApi,
	client: Mem0Client,
	cfg: Mem0Config,
): void {
	api.registerTool(
		{
			name: "mem0_search",
			label: "Memory Search",
			description:
				"Search through long-term memories for relevant information.",
			parameters: Type.Object({
				query: Type.String({ description: "Search query" }),
				limit: Type.Optional(
					Type.Number({ description: "Max results (default: 5)" }),
				),
				scope: Type.Optional(stringEnum(["own", "shared", "all"] as const)),
				userId: Type.Optional(
					Type.String({
						description:
							"Explicit user ID override (takes precedence over scope)",
					}),
				),
			}),
			async execute(
				_toolCallId: string,
				params: {
					query: string
					limit?: number
					scope?: "own" | "shared" | "all"
					userId?: string
				},
			) {
				const limit = params.limit ?? 5
				const scope = params.scope ?? "all"
				log.debug(
					`search tool: query="${params.query}" limit=${limit} scope="${scope}" userId="${params.userId ?? "default"}"`,
				)

				let results: SearchResult[]
				if (params.userId) {
					results = await client.search(params.query, limit, params.userId)
				} else if (
					scope === "all" &&
					cfg.inheritSharedMemory &&
					cfg.sharedUserId !== cfg.userId
				) {
					results = await client.searchMultiple(params.query, limit, [
						cfg.userId,
						cfg.sharedUserId,
					])
				} else if (scope === "shared") {
					results = await client.search(params.query, limit, cfg.sharedUserId)
				} else {
					results = await client.search(params.query, limit)
				}

				if (results.length === 0) {
					return {
						content: [
							{ type: "text" as const, text: "No relevant memories found." },
						],
					}
				}

				const text = results
					.map((r, i) => {
						const score = r.similarity
							? ` (${(r.similarity * 100).toFixed(0)}%)`
							: ""
						return `${i + 1}. ${r.content || r.memory || ""}${score}`
					})
					.join("\n")

				return {
					content: [
						{
							type: "text" as const,
							text: `Found ${results.length} memories:\n\n${text}`,
						},
					],
					details: {
						count: results.length,
						memories: results.map((r) => ({
							id: r.id,
							content: r.content,
							similarity: r.similarity,
						})),
					},
				}
			},
		},
		{ name: "mem0_search" },
	)
}
