// biome-ignore lint/suspicious/noControlCharactersInRegex: intentional control char stripping
const CONTROL_CHARS = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g
const BOM_AND_SPECIALS = /[\uFEFF\uFFF0-\uFFFF]/g
const MAX_CONTENT_LENGTH = 500_000
const MAX_METADATA_KEYS = 50
const MAX_KEY_LENGTH = 128
const MAX_VALUE_LENGTH = 4096
const VALID_KEY = /^[\w.-]+$/

export function sanitizeContent(
	content: string,
	maxLength = MAX_CONTENT_LENGTH,
): string {
	return content
		.replace(CONTROL_CHARS, "")
		.replace(BOM_AND_SPECIALS, "")
		.slice(0, maxLength)
}

export function sanitizeMetadata(
	meta: Record<string, unknown>,
): Record<string, string | number | boolean> {
	const clean: Record<string, string | number | boolean> = {}
	let count = 0

	for (const [key, val] of Object.entries(meta)) {
		if (count >= MAX_METADATA_KEYS) break
		if (key.length > MAX_KEY_LENGTH || !VALID_KEY.test(key)) continue

		if (typeof val === "boolean" || typeof val === "number") {
			clean[key] = val
		} else if (typeof val === "string") {
			clean[key] = val.slice(0, MAX_VALUE_LENGTH)
		}
		count++
	}

	return clean
}
