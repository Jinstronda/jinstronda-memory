import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { Mem0Client } from "../client.ts"
import type { Mem0Config } from "../config.ts"
import { log } from "../logger.ts"

export function registerProfileTool(
	api: OpenClawPluginApi,
	client: Mem0Client,
	_cfg: Mem0Config,
): void {
	api.registerTool(
		{
			name: "mem0_profile",
			label: "User Profile",
			description:
				"Get a summary of what is known about the user, stable preferences and recent context.",
			parameters: Type.Object({
				query: Type.Optional(
					Type.String({
						description: "Optional query to focus the profile",
					}),
				),
				userId: Type.Optional(
					Type.String({
						description:
							"Optional user ID to get profile from a specific scope",
					}),
				),
			}),
			async execute(
				_toolCallId: string,
				params: { query?: string; userId?: string },
			) {
				log.debug(
					`profile tool: query="${params.query ?? "(none)"}" userId="${params.userId ?? "default"}"`,
				)

				const profile = await client.getProfile(
					params.query,
					params.userId,
				)

				if (profile.static.length === 0 && profile.dynamic.length === 0) {
					return {
						content: [
							{
								type: "text" as const,
								text: "No profile information available yet.",
							},
						],
					}
				}

				const sections: string[] = []

				if (profile.static.length > 0) {
					sections.push(
						"## User Profile (Persistent)\n" +
							profile.static.map((f) => `- ${f}`).join("\n"),
					)
				}

				if (profile.dynamic.length > 0) {
					sections.push(
						"## Recent Context\n" +
							profile.dynamic.map((f) => `- ${f}`).join("\n"),
					)
				}

				return {
					content: [{ type: "text" as const, text: sections.join("\n\n") }],
					details: {
						staticCount: profile.static.length,
						dynamicCount: profile.dynamic.length,
					},
				}
			},
		},
		{ name: "mem0_profile" },
	)
}
