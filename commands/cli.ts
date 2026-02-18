import * as os from "node:os"
import * as path from "node:path"
import * as readline from "node:readline"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { Mem0Client } from "../client.ts"
import type { Mem0Config } from "../config.ts"
import { log } from "../logger.ts"

export function registerCliSetup(api: OpenClawPluginApi): void {
	api.registerCli(
		// biome-ignore lint/suspicious/noExplicitAny: openclaw SDK does not ship types
		({ program }: { program: any }) => {
			const cmd = program
				.command("mem0")
				.description("Mem0 long-term memory commands")

			cmd
				.command("status")
				.description("Check Mem0 configuration status")
				.action(async () => {
					const defaultUserId = `openclaw_${os.hostname().replace(/[^a-zA-Z0-9_]/g, "_")}`

					console.log("\nMem0 Memory Status\n")

					const configPath = path.join(
						os.homedir(),
						".openclaw",
						"openclaw.json",
					)
					let pluginConfig: Record<string, unknown> = {}
					let enabled = true

					const fs = await import("node:fs")
					if (fs.existsSync(configPath)) {
						try {
							const config = JSON.parse(fs.readFileSync(configPath, "utf-8"))
							const entry = config?.plugins?.entries?.["openclaw-mem0-memory"]
							if (entry) {
								enabled = entry.enabled ?? true
								pluginConfig = entry.config ?? {}
							}
						} catch {
							console.log("Could not read config file\n")
							return
						}
					}

					const customContainers = Array.isArray(pluginConfig.customContainers)
						? pluginConfig.customContainers
						: []

					console.log(`  Enabled:          ${enabled}`)
					console.log(
						"  Mem0 URL:         " +
							(pluginConfig.mem0Url ?? "http://localhost:8080"),
					)
					console.log(
						`  User ID:          ${pluginConfig.userId ?? defaultUserId}`,
					)
					console.log(`  Auto-recall:      ${pluginConfig.autoRecall ?? true}`)
					console.log(`  Auto-capture:     ${pluginConfig.autoCapture ?? true}`)
					console.log(
						`  Max results:      ${pluginConfig.maxRecallResults ?? 10}`,
					)
					console.log(
						`  Profile freq:     ${pluginConfig.profileFrequency ?? 50}`,
					)
					console.log(
						`  Capture mode:     ${pluginConfig.captureMode ?? "all"}`,
					)
					console.log(
						"  Shared User ID:   " +
							(pluginConfig.sharedUserId ?? "(same as userId)"),
					)
					console.log(
						`  Inherit shared:   ${pluginConfig.inheritSharedMemory ?? true}`,
					)
					console.log(`  Custom containers: ${customContainers.length}`)
					console.log("")
				})
		},
		{ commands: ["mem0"] },
	)
}

export function registerCli(
	api: OpenClawPluginApi,
	client: Mem0Client,
	_cfg: Mem0Config,
): void {
	api.registerCli(
		// biome-ignore lint/suspicious/noExplicitAny: openclaw SDK does not ship types
		({ program }: { program: any }) => {
			const cmd = program.commands.find(
				// biome-ignore lint/suspicious/noExplicitAny: openclaw SDK does not ship types
				(c: any) => c.name() === "mem0",
			)
			if (!cmd) return

			cmd
				.command("search")
				.argument("<query>", "Search query")
				.option("--limit <n>", "Max results", "5")
				.action(async (query: string, opts: { limit: string }) => {
					const limit = Number.parseInt(opts.limit, 10) || 5
					log.debug(`cli search: query="${query}" limit=${limit}`)

					const results = await client.search(query, limit)

					if (results.length === 0) {
						console.log("No memories found.")
						return
					}

					for (const r of results) {
						const score = r.similarity
							? ` (${(r.similarity * 100).toFixed(0)}%)`
							: ""
						console.log(`- ${r.content || r.memory || ""}${score}`)
					}
				})

			cmd
				.command("profile")
				.option("--query <q>", "Optional query to focus the profile")
				.action(async (opts: { query?: string }) => {
					log.debug(`cli profile: query="${opts.query ?? "(none)"}"`)

					const profile = await client.getProfile(opts.query)

					if (profile.static.length === 0 && profile.dynamic.length === 0) {
						console.log("No profile information available yet.")
						return
					}

					if (profile.static.length > 0) {
						console.log("Stable Preferences:")
						for (const f of profile.static) {
							console.log(`  - ${f}`)
						}
					}

					if (profile.dynamic.length > 0) {
						console.log("Recent Context:")
						for (const f of profile.dynamic) {
							console.log(`  - ${f}`)
						}
					}
				})

			cmd
				.command("wipe")
				.description("Delete ALL memories for this user")
				.action(async () => {
					const userId = client.getUserId()
					const rl = readline.createInterface({
						input: process.stdin,
						output: process.stdout,
					})

					const answer = await new Promise<string>((resolve) => {
						rl.question(
							'This will permanently delete all memories for "' +
								userId +
								'". Type "yes" to confirm: ',
							resolve,
						)
					})
					rl.close()

					if (answer.trim().toLowerCase() !== "yes") {
						console.log("Aborted.")
						return
					}

					log.debug(`cli wipe: userId="${userId}"`)
					const result = await client.wipeAllMemories()
					console.log(`Wiped ${result.deletedCount} memories for "${userId}".`)
				})
		},
		{ commands: ["mem0"] },
	)
}
