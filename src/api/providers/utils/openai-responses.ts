import type { ModelInfo, ReasoningEffortWithMinimal, ServiceTier } from "@roo-code/types"
import type { ApiStreamChunk } from "../../transform/stream"

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
	const url = `${baseUrl.replace(/\/$/, "")}/v1/responses`
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
		const text = await res.text().catch(() => "<no body>")
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
							yield parsed
						} catch {
							// Non-JSON data — ignore
						}
					}
				}
			}
		}
	} finally {
		reader.releaseLock()
	}
}

/**
 * Process either:
 *  - an AsyncIterable of events (SDK streaming), or
 *  - a ReadableStream (body), or
 *  - a single event object
 *
 * The function yields Roo ApiStream chunks.
 */
export async function* processResponsesEventStream(
	streamOrEvent: any,
	model: { id: string; info: ModelInfo },
): AsyncGenerator<ApiStreamChunk> {
	// If it's an async iterable (SDK streaming)
	if (streamOrEvent && typeof streamOrEvent[Symbol.asyncIterator] === "function") {
		for await (const event of streamOrEvent) {
			// SDK sometimes yields a ReadableStream body; handle that case
			if (event && typeof event === "object" && typeof event.body?.getReader === "function") {
				// Convert body -> SSE events then process
				for await (const parsedEvent of parseReadableStreamAsSse(event.body)) {
					for await (const chunk of processSingleEvent(parsedEvent, model)) {
						yield chunk
					}
				}
			} else {
				for await (const chunk of processSingleEvent(event, model)) {
					yield chunk
				}
			}
		}
		return
	}

	// If it's a ReadableStream directly (SSE)
	if (streamOrEvent && typeof streamOrEvent.getReader === "function") {
		for await (const parsedEvent of parseReadableStreamAsSse(streamOrEvent)) {
			for await (const chunk of processSingleEvent(parsedEvent, model)) {
				yield chunk
			}
		}
		return
	}

	// Otherwise assume a single event object
	for await (const chunk of processSingleEvent(streamOrEvent, model)) {
		yield chunk
	}
}

/** Helper: parse a ReadableStream (Uint8Array) into parsed SSE JSON events */
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
							yield parsed
						} catch {
							// ignore parse errors
						}
					}
				}
			}
		}
	} finally {
		reader.releaseLock()
	}
}

/** Map a single Responses event to Roo ApiStream chunks */
async function* processSingleEvent(event: any, model: { id: string; info: ModelInfo }): AsyncGenerator<ApiStreamChunk> {
	if (!event) return

	// Prefer explicit event.type, fallback to event.event
	const type = event.type || event.event

	// Keep response shape handling similar to existing implementation:
	// - text deltas
	if (type === "response.text.delta" || type === "response.output_text.delta") {
		if (event.delta) {
			yield { type: "text", text: event.delta }
		}
		return
	}

	// - reasoning deltas and summaries
	if (
		type === "response.reasoning.delta" ||
		type === "response.reasoning_text.delta" ||
		type === "response.reasoning_summary.delta" ||
		type === "response.reasoning_summary_text.delta"
	) {
		if (event.delta) {
			yield { type: "reasoning", text: event.delta }
		}
		return
	}

	// - usage events (best-effort; precise cost calculation happens elsewhere)
	if (type === "response.usage" || event?.usage || event?.response?.usage) {
		const usage = event.usage ?? event.response?.usage ?? {}
		yield {
			type: "usage",
			inputTokens: usage.input_tokens ?? usage.prompt_tokens ?? 0,
			outputTokens: usage.output_tokens ?? usage.completion_tokens ?? 0,
			cacheReadTokens: usage.cache_read_tokens ?? 0,
			cacheWriteTokens: usage.cache_write_tokens ?? 0,
			totalCost: 0,
		} as any
		return
	}

	// - completed / done
	if (type === "response.done" || type === "response.completed") {
		yield { type: "done" } as any
		return
	}

	// - Non-streaming full response: event.response.output array
	if (event.response && Array.isArray(event.response.output)) {
		for (const outputItem of event.response.output) {
			if (outputItem.type === "text" && Array.isArray(outputItem.content)) {
				for (const content of outputItem.content) {
					if (content?.type === "text" && typeof content.text === "string") {
						yield { type: "text", text: content.text }
					}
				}
			}
			// Handle reasoning summaries included in non-streaming outputs
			if (outputItem.type === "reasoning" && Array.isArray(outputItem.summary)) {
				for (const summary of outputItem.summary) {
					if (summary?.type === "summary_text" && typeof summary.text === "string") {
						yield { type: "reasoning", text: summary.text }
					}
				}
			}
		}
		return
	}

	// Fallback: if event.delta is present as a string, yield as text
	if (typeof event.delta === "string") {
		yield { type: "text", text: event.delta }
	}
}

/**
 * Minimal recreation of the conversation formatting used by the Responses API.
 * Copied/adapted from openai-native.formatFullConversation(...) for reuse by providers.
 */
export function formatFullConversation(systemPrompt: string, messages: any[]): any[] {
	const formattedMessages: any[] = []

	for (const message of messages) {
		const role = message.role === "user" ? "user" : "assistant"
		const content: any[] = []

		if (typeof message.content === "string") {
			if (role === "user") {
				content.push({ type: "input_text", text: message.content })
			} else {
				content.push({ type: "output_text", text: message.content })
			}
		} else if (Array.isArray(message.content)) {
			for (const block of message.content) {
				if (block.type === "text") {
					content.push({ type: role === "user" ? "input_text" : "output_text", text: block.text })
				} else {
					// Preserve unknown block shapes (images, attachments, etc.)
					content.push(block)
				}
			}
		}

		if (content.length > 0) {
			formattedMessages.push({ role, content })
		}
	}

	return formattedMessages
}
