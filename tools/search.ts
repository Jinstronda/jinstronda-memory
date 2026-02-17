import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { Mem0Client } from "../client.ts"
import type { Mem0Config } from "../config.ts"
import { log } from "../logger.ts"

export function registerSearchTool(
	api: OpenClawPluginApi,
	client: Mem0Client,
	_cfg: Mem0Config,
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
				userId: Type.Optional(
					Type.String({
						description:
							"Optional user ID to search under a specific scope",
					}),
				),
			}),
			async execute(
				_toolCallId: string,
				params: { query: string; limit?: number; userId?: string },
			) {
				const limit = params.limit ?? 5
				log.debug(
					`search tool: query="${params.query}" limit=${limit} userId="${params.userId ?? "default"}"`,
				)

				const results = await client.search(
					params.query,
					limit,
					params.userId,
				)

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
