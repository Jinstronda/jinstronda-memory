import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { Mem0Client } from "../client.ts"
import type { Mem0Config } from "../config.ts"
import { log } from "../logger.ts"

export function registerForgetTool(
	api: OpenClawPluginApi,
	client: Mem0Client,
	_cfg: Mem0Config,
): void {
	api.registerTool(
		{
			name: "mem0_forget",
			label: "Memory Forget",
			description:
				"Forget/delete a specific memory. Searches for the closest match and removes it.",
			parameters: Type.Object({
				query: Type.Optional(
					Type.String({ description: "Describe the memory to forget" }),
				),
				memoryId: Type.Optional(
					Type.String({ description: "Direct memory ID to delete" }),
				),
				userId: Type.Optional(
					Type.String({
						description: "Optional user ID to delete from a specific scope",
					}),
				),
			}),
			async execute(
				_toolCallId: string,
				params: { query?: string; memoryId?: string; userId?: string },
			) {
				if (params.memoryId) {
					log.debug(`forget tool: direct delete id="${params.memoryId}"`)
					await client.deleteMemory(params.memoryId)
					return {
						content: [{ type: "text" as const, text: "Memory forgotten." }],
					}
				}

				if (params.query) {
					log.debug(
						`forget tool: search-then-delete query="${params.query}" userId="${params.userId ?? "default"}"`,
					)
					const result = await client.forgetByQuery(params.query, params.userId)
					return {
						content: [{ type: "text" as const, text: result.message }],
					}
				}

				return {
					content: [
						{
							type: "text" as const,
							text: "Provide a query or memoryId to forget.",
						},
					],
				}
			},
		},
		{ name: "mem0_forget" },
	)
}
