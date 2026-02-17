import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import { stringEnum } from "openclaw/plugin-sdk"
import type { Mem0Client } from "../client.ts"
import type { Mem0Config } from "../config.ts"
import { log } from "../logger.ts"
import {
	buildDocumentId,
	detectCategory,
	MEMORY_CATEGORIES,
} from "../memory.ts"

export function registerStoreTool(
	api: OpenClawPluginApi,
	client: Mem0Client,
	_cfg: Mem0Config,
	getSessionKey: () => string | undefined,
): void {
	api.registerTool(
		{
			name: "mem0_store",
			label: "Memory Store",
			description: "Save important information to long-term memory.",
			parameters: Type.Object({
				text: Type.String({ description: "Information to remember" }),
				category: Type.Optional(stringEnum(MEMORY_CATEGORIES)),
				userId: Type.Optional(
					Type.String({
						description:
							"Optional user ID to store the memory under a specific scope",
					}),
				),
			}),
			async execute(
				_toolCallId: string,
				params: { text: string; category?: string; userId?: string },
			) {
				const category = params.category ?? detectCategory(params.text)
				const sk = getSessionKey()
				const customId = sk ? buildDocumentId(sk) : undefined

				log.debug(
					`store tool: category="${category}" customId="${customId}" userId="${params.userId ?? "default"}"`,
				)

				await client.addMemory(
					params.text,
					{ type: category, source: "openclaw_tool" },
					customId,
					params.userId,
				)

				const preview =
					params.text.length > 80 ? `${params.text.slice(0, 80)}...` : params.text

				return {
					content: [{ type: "text" as const, text: `Stored: "${preview}"` }],
				}
			},
		},
		{ name: "mem0_store" },
	)
}
