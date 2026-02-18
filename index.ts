import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import { Mem0Client } from "./client.ts"
import { registerCli, registerCliSetup } from "./commands/cli.ts"
import { registerCommands } from "./commands/slash.ts"
import { mem0ConfigSchema, parseConfig } from "./config.ts"
import { buildCaptureHandler } from "./hooks/capture.ts"
import { buildRecallHandler } from "./hooks/recall.ts"
import { initLogger } from "./logger.ts"
import { registerForgetTool } from "./tools/forget.ts"
import { registerProfileTool } from "./tools/profile.ts"
import { registerSearchTool } from "./tools/search.ts"
import { registerStoreTool } from "./tools/store.ts"

export default {
	id: "openclaw-mem0-memory",
	name: "Mem0 Memory",
	description: "OpenClaw long-term memory powered by Mem0",
	kind: "memory" as const,
	configSchema: mem0ConfigSchema,

	register(api: OpenClawPluginApi) {
		const cfg = parseConfig(api.pluginConfig)

		initLogger(api.logger, cfg.debug)

		registerCliSetup(api)

		const client = new Mem0Client(cfg.mem0Url, cfg.userId)

		let sessionKey: string | undefined
		const getSessionKey = () => sessionKey

		registerSearchTool(api, client, cfg)
		registerStoreTool(api, client, cfg, getSessionKey)
		registerForgetTool(api, client, cfg)
		registerProfileTool(api, client, cfg)

		if (cfg.autoRecall) {
			const recallHandler = buildRecallHandler(client, cfg)
			api.on(
				"before_agent_start",
				(event: Record<string, unknown>, ctx: Record<string, unknown>) => {
					if (ctx.sessionKey) sessionKey = ctx.sessionKey as string
					return recallHandler(event, ctx)
				},
			)
		}

		if (cfg.autoCapture) {
			api.on("agent_end", buildCaptureHandler(client, cfg, getSessionKey))
		}

		registerCommands(api, client, cfg, getSessionKey)
		registerCli(api, client, cfg)

		api.registerService({
			id: "openclaw-mem0-memory",
			start: async () => {
				try {
					const res = await fetch(cfg.mem0Url + "/health")
					if (res.ok) {
						api.logger.info("mem0: connected to " + cfg.mem0Url)
					} else {
						api.logger.warn(
							"mem0: server returned " + res.status + ", memory may not work",
						)
					}
				} catch {
					api.logger.warn(
						"mem0: cannot reach " +
							cfg.mem0Url +
							", memory will not work until server is running",
					)
				}
			},
			stop: () => {
				api.logger.info("mem0: stopped")
			},
		})
	},
}
