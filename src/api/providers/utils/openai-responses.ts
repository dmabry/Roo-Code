import type { ModelInfo, ReasoningEffortWithMinimal, ServiceTier } from "@roo-code/types"
import type {
	ApiStreamChunk,
	ApiStreamTextChunk,
	ApiStreamUsageChunk,
	ApiStreamReasoningChunk,
} from "../../transform/stream"

type ConversationMessage = {
	role: "user" | "assistant"
	content: any[]
}

/**
 * Format Anthropic-style messages into the structure expected by the OpenAI
 * Responses API. System prompts are handled separately via the `instructions`
 * field, but we allow the caller to pass the prompt to preserve parity with
 * existing call sites.
 */
export function formatFullConversation(_systemPrompt: string | undefined, messages: any[]): ConversationMessage[] {
	const formatted: ConversationMessage[] = []

	for (const message of messages || []) {
		if (!message) continue

		const role: ConversationMessage["role"] = message.role === "user" ? "user" : "assistant"
		const contentBlocks: any[] = []

		const rawContent = message.content

		if (typeof rawContent === "string") {
			contentBlocks.push({
				type: role === "user" ? "input_text" : "output_text",
				text: rawContent,
			})
		} else if (Array.isArray(rawContent)) {
			for (const block of rawContent) {
				if (!block) continue

				if (block.type === "text" && typeof block.text === "string") {
					contentBlocks.push({
						type: role === "user" ? "input_text" : "output_text",
						text: block.text,
					})
				} else if (typeof block.type === "string") {
					// Preserve any structured block (images, tool calls, etc.)
					contentBlocks.push({ ...block })
				}
			}
		}

		if (contentBlocks.length > 0) {
			formatted.push({ role, content: contentBlocks })
		}
	}

	return formatted
}

type BuildOptions = {
	enableGpt5ReasoningSummary?: boolean
	openAiNativeServiceTier?: ServiceTier | undefined
	modelTemperature?: number | undefined
}

/**
 * Build a Responses API request body compatible with Roo's expectations.
 * This is a pared-down, reusable extraction of the logic previously in
 * openai-native.ts:buildRequestBody(...)
 */
export function buildResponsesRequestBody(
	model: { id: string; info: ModelInfo },
	formattedInput: any,
	requestPreviousResponseId?: string,
	systemPrompt?: string,
	verbosity?: any,
	reasoningEffort?: ReasoningEffortWithMinimal | undefined,
	metadata?: any,
	opts: BuildOptions = {},
): any {
	// Basic body
	const body: any = {
		model: model.id,
		input: formattedInput,
		stream: true,
		store: metadata?.store !== false,
		// Instructions carry system prompt semantics for Responses API
		instructions: systemPrompt,
	}

	// Reasoning options
	if (reasoningEffort) {
		body.reasoning = {
			effort: reasoningEffort,
			...(opts.enableGpt5ReasoningSummary ? { summary: "auto" } : {}),
		}
	}

	// Verbosity (only include when model explicitly supports it)
	if (model.info?.supportsVerbosity === true) {
		body.text = { verbosity: (verbosity || "medium") as any }
	}

	// Temperature when supported by model
	if (model.info?.supportsTemperature !== false) {
		if (typeof opts.modelTemperature === "number") {
			body.temperature = opts.modelTemperature
		}
	}

	// Explicit max output tokens when model specifies it
	const modelMaxTokens = (model as any).maxTokens ?? model.info?.maxTokens
	if (modelMaxTokens) {
		body.max_output_tokens = modelMaxTokens
	}

	// previous_response_id for continuity when provided
	if (requestPreviousResponseId) {
		body.previous_response_id = requestPreviousResponseId
	}

	// Service tier when requested and supported by model
	const requestedTier = opts.openAiNativeServiceTier
	const allowedTierNames = new Set(model.info?.tiers?.map((t: any) => t.name).filter(Boolean) || [])
	if (requestedTier && (requestedTier === "default" || allowedTierNames.has(requestedTier))) {
		body.service_tier = requestedTier
	}

	return body
}

/**
 * Try to call the OpenAI SDK's responses.create. The SDK may return:
 *  - an AsyncIterable of events (preferred), or
 *  - a single event object (non-iterable), or
 *  - throw (in which case callers can fallback to SSE fetch)
 */
export async function responsesSdkStream(client: any, requestBody: any) {
	if (!client || !client.responses || typeof client.responses.create !== "function") {
		throw new Error("OpenAI SDK client does not support responses.create")
	}
	// Delegate to SDK; caller will handle iterable vs non-iterable shapes
	return client.responses.create(requestBody)
}

/**
 * SSE fallback: POST to /v1/responses and parse text/event-stream.
 * Yields parsed JSON events (same shape as SDK events where possible).
 */
export async function* responsesSseFetch(baseUrl: string, apiKey: string, requestBody: any) {
	// Normalize the provided baseUrl to avoid duplicating `/v1` when callers
	// already include it (e.g. "https://api.openai.com/v1").
	let normalized = (baseUrl || "").replace(/\/+$/, "") // remove trailing slashes
	let url: string
	// If caller already provided a full /responses path, use it as-is.
	if (normalized.endsWith("/v1/responses") || normalized.endsWith("/responses")) {
		url = normalized
	} else if (normalized.endsWith("/v1")) {
		// e.g. "https://api.openai.com/v1" -> "https://api.openai.com/v1/responses"
		url = `${normalized}/responses`
	} else {
		// e.g. "https://api.example.com" -> "https://api.example.com/v1/responses"
		url = `${normalized}/v1/responses`
	}

	// Ensure streaming
	requestBody = { ...requestBody, stream: true }

	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Accept: "text/event-stream",
			Authorization: `Bearer ${apiKey}`,
		},
		body: JSON.stringify(requestBody),
	})

	if (!res.ok) {
		const text = await (typeof res.text === "function" ? res.text() : Promise.resolve("<no body>"))
		throw new Error(`Responses SSE request failed: ${res.status} ${res.statusText} - ${text}`)
	}

	const body = res.body
	if (!body) return

	const reader = body.getReader()
	const decoder = new TextDecoder()
	let buffer = ""

	try {
		while (true) {
			const { done, value } = await reader.read()
			if (done) break
			buffer += decoder.decode(value, { stream: true })

			// SSE events are separated by blank lines
			const parts = buffer.split("\n\n")
			buffer = parts.pop() || ""

			for (const part of parts) {
				// Each part may contain multiple lines (we care about "data: " lines)
				const lines = part.split("\n").map((l) => l.trim())
				for (const line of lines) {
					if (!line) continue
					if (line.startsWith("data: ")) {
						const data = line.slice(6).trim()
						if (data === "[DONE]") {
							return
						}
						try {
							const parsed = JSON.parse(data)
							yield parsed as unknown
						} catch {}
					}
				}
			}
		}
	} finally {
		reader.releaseLock()
	}
}

export async function* processResponsesEventStream(
	streamOrEvent: any,
	model: { id: string; info: ModelInfo },
): AsyncGenerator<ApiStreamChunk> {
	if (!streamOrEvent) {
		return
	}

	if (typeof streamOrEvent[Symbol.asyncIterator] === "function") {
		for await (const event of streamOrEvent) {
			if (event && typeof event === "object" && typeof event.body?.getReader === "function") {
				for await (const parsedEvent of parseReadableStreamAsSse(event.body)) {
					yield* processSingleEvent(parsedEvent, model)
				}
			} else {
				yield* processSingleEvent(event, model)
			}
		}
		return
	}

	if (typeof streamOrEvent?.getReader === "function") {
		for await (const parsedEvent of parseReadableStreamAsSse(streamOrEvent)) {
			yield* processSingleEvent(parsedEvent, model)
		}
		return
	}

	yield* processSingleEvent(streamOrEvent, model)
}

/**
 * Convert a ReadableStream into SSE events.
 */
async function* parseReadableStreamAsSse(bodyStream: ReadableStream<Uint8Array>) {
	const reader = bodyStream.getReader()
	const decoder = new TextDecoder()
	let buffer = ""
	try {
		while (true) {
			const { done, value } = await reader.read()
			if (done) break
			buffer += decoder.decode(value, { stream: true })
			const parts = buffer.split("\n\n")
			buffer = parts.pop() || ""
			for (const part of parts) {
				const lines = part.split("\n").map((l) => l.trim())
				for (const line of lines) {
					if (!line) continue
					if (line.startsWith("data: ")) {
						const data = line.slice(6).trim()
						if (data === "[DONE]") {
							return
						}
						try {
							const parsed = JSON.parse(data)
							yield parsed as unknown
						} catch {}
					}
				}
			}
		}
	} finally {
		reader.releaseLock()
	}
}

/**
 * Process a single event into ApiStream chunks.
 */
async function* processSingleEvent(event: any, model: { id: string; info: ModelInfo }) {
	if (!event) return
	const type = event.type ?? event.event

	// Text delta or output text delta
	if (type === "response.text.delta" || type === "response.output_text.delta") {
		if (typeof event?.delta === "string" && event.delta.length > 0) {
			yield { type: "text", text: event.delta } as ApiStreamTextChunk
		}
		return
	}

	// Reasoning delta, reasoning_text_delta, summary etc.
	if (
		type === "response.reasoning.delta" ||
		type === "response.reasoning_text.delta" ||
		type === "response.reasoning_summary.delta" ||
		type === "response.reasoning_summary_text.delta"
	) {
		if (typeof event?.delta === "string" && event.delta.length > 0) {
			yield { type: "reasoning", text: event.delta } as ApiStreamReasoningChunk
		}
		return
	}

	// Usage or usage events
	if (type === "response.usage" || event?.usage || event?.response?.usage) {
		const usage = event?.usage ?? event?.response?.usage ?? {}
		yield {
			type: "usage",
			inputTokens: usage.input_tokens ?? usage.prompt_tokens ?? 0,
			outputTokens: usage.output_tokens ?? usage.completion_tokens ?? 0,
			cacheWriteTokens: usage.cache_write_tokens ?? 0,
			cacheReadTokens: usage.cache_read_tokens ?? 0,
			totalCost: usage.total_cost ?? 0,
		} as ApiStreamUsageChunk
		return
	}

	// Completion or done events
	if (type === "response.done" || type === "response.completed") {
		return
	}

	if (event?.response && Array.isArray(event.response.output)) {
		for (const output of event.response.output) {
			if (!output) continue

			if (output.type === "text") {
				if (typeof output.text === "string") {
					yield { type: "text", text: output.text } as ApiStreamTextChunk
				}
				if (Array.isArray(output.content)) {
					for (const content of output.content) {
						if (content?.type === "text" && typeof content.text === "string") {
							yield { type: "text", text: content.text } as ApiStreamTextChunk
						}
					}
				}
			} else if (output.type === "reasoning" && Array.isArray(output.summary)) {
				for (const summary of output.summary) {
					if (summary?.type === "summary_text" && typeof summary.text === "string") {
						yield { type: "reasoning", text: summary.text } as ApiStreamReasoningChunk
					}
				}
			}
		}
		return
	}

	if (typeof event?.delta === "string" && event.delta.length > 0) {
		yield { type: "text", text: event.delta } as ApiStreamTextChunk
	}
}
