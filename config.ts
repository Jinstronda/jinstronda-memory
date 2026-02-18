import { hostname } from "node:os"

export type CaptureMode = "everything" | "all"

export type CustomContainer = {
	tag: string
	description: string
}

export type Mem0Config = {
	mem0Url: string
	userId: string
	sharedUserId: string
	inheritSharedMemory: boolean
	autoRecall: boolean
	autoCapture: boolean
	maxRecallResults: number
	profileFrequency: number
	captureMode: CaptureMode
	debug: boolean
	enableCustomContainerTags: boolean
	customContainers: CustomContainer[]
	customContainerInstructions: string
}

const ALLOWED_KEYS = [
	"mem0Url",
	"userId",
	"sharedUserId",
	"inheritSharedMemory",
	"autoRecall",
	"autoCapture",
	"maxRecallResults",
	"profileFrequency",
	"captureMode",
	"debug",
	"enableCustomContainerTags",
	"customContainers",
	"customContainerInstructions",
]

function assertAllowedKeys(
	value: Record<string, unknown>,
	allowed: string[],
	label: string,
): void {
	const unknown = Object.keys(value).filter((k) => !allowed.includes(k))
	if (unknown.length > 0) {
		throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`)
	}
}

function sanitizeTag(raw: string): string {
	return raw
		.replace(/[^a-zA-Z0-9_]/g, "_")
		.replace(/_+/g, "_")
		.replace(/^_|_$/g, "")
}

function defaultUserId(): string {
	return sanitizeTag(`openclaw_${hostname()}`)
}

export function parseConfig(raw: unknown): Mem0Config {
	const cfg =
		raw && typeof raw === "object" && !Array.isArray(raw)
			? (raw as Record<string, unknown>)
			: {}

	if (Object.keys(cfg).length > 0) {
		assertAllowedKeys(cfg, ALLOWED_KEYS, "mem0 config")
	}

	const customContainers: CustomContainer[] = []
	if (Array.isArray(cfg.customContainers)) {
		for (const c of cfg.customContainers) {
			if (
				c &&
				typeof c === "object" &&
				typeof (c as Record<string, unknown>).tag === "string" &&
				typeof (c as Record<string, unknown>).description === "string"
			) {
				customContainers.push({
					tag: sanitizeTag((c as Record<string, unknown>).tag as string),
					description: (c as Record<string, unknown>).description as string,
				})
			}
		}
	}

	const userId = cfg.userId
		? sanitizeTag(cfg.userId as string)
		: defaultUserId()

	return {
		mem0Url:
			typeof cfg.mem0Url === "string" && cfg.mem0Url.length > 0
				? cfg.mem0Url
				: "http://localhost:8080",
		userId,
		sharedUserId: cfg.sharedUserId
			? sanitizeTag(cfg.sharedUserId as string)
			: userId,
		inheritSharedMemory: (cfg.inheritSharedMemory as boolean) ?? true,
		autoRecall: (cfg.autoRecall as boolean) ?? true,
		autoCapture: (cfg.autoCapture as boolean) ?? true,
		maxRecallResults: (cfg.maxRecallResults as number) ?? 10,
		profileFrequency: (cfg.profileFrequency as number) ?? 50,
		captureMode:
			cfg.captureMode === "everything"
				? ("everything" as const)
				: ("all" as const),
		debug: (cfg.debug as boolean) ?? false,
		enableCustomContainerTags:
			(cfg.enableCustomContainerTags as boolean) ?? false,
		customContainers,
		customContainerInstructions:
			typeof cfg.customContainerInstructions === "string"
				? cfg.customContainerInstructions
				: "",
	}
}

export const mem0ConfigSchema = {
	jsonSchema: {
		type: "object",
		additionalProperties: false,
		properties: {
			mem0Url: { type: "string" },
			userId: { type: "string" },
			sharedUserId: { type: "string" },
			inheritSharedMemory: { type: "boolean" },
			autoRecall: { type: "boolean" },
			autoCapture: { type: "boolean" },
			maxRecallResults: { type: "number" },
			profileFrequency: { type: "number" },
			captureMode: { type: "string", enum: ["all", "everything"] },
			debug: { type: "boolean" },
			enableCustomContainerTags: { type: "boolean" },
			customContainers: {
				type: "array",
				items: {
					type: "object",
					properties: {
						tag: { type: "string" },
						description: { type: "string" },
					},
					required: ["tag", "description"],
				},
			},
			customContainerInstructions: { type: "string" },
		},
	},
	parse: parseConfig,
}
