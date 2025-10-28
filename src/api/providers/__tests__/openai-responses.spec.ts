import { describe, it, expect, vitest, beforeEach } from "vitest"
import {
	formatFullConversation,
	buildResponsesRequestBody,
	responsesSdkStream,
	responsesSseFetch,
	processResponsesEventStream,
} from "../utils/openai-responses"

const encoder = new TextEncoder()

function createReadableStreamFromString(payload: string, chunkSize = payload.length): ReadableStream<Uint8Array> {
	const chunks: Uint8Array[] = []
	for (let i = 0; i < payload.length; i += chunkSize) {
		chunks.push(encoder.encode(payload.slice(i, i + chunkSize)))
	}
	let index = 0
	const reader = {
		async read() {
			if (index >= chunks.length) {
				return { done: true, value: new Uint8Array() }
			}
			return { done: false, value: chunks[index++] }
		},
		releaseLock() {},
	}
	return { getReader: () => reader } as unknown as ReadableStream<Uint8Array>
}

const baseModel = { id: "test-model", info: {} as any }

describe("formatFullConversation", () => {
	it("converts anthropic-style messages to Responses format", () => {
		const attachment = { type: "image_block", url: "https://example.com/cat.png" }
		const toolCall = { type: "tool_call", name: "search", payload: { query: "docs" } }

		const result = formatFullConversation("system prompt", [
			{ role: "user", content: "hello" },
			{ role: "assistant", content: [{ type: "text", text: "hi there" }, attachment] },
			{ role: "user", content: [{ type: "text", text: "another" }, toolCall] },
		])

		expect(result).toHaveLength(3)
		expect(result[0]).toEqual({
			role: "user",
			content: [{ type: "input_text", text: "hello" }],
		})
		expect(result[1].role).toBe("assistant")
		expect(result[1].content[0]).toEqual({ type: "output_text", text: "hi there" })
		expect(result[1].content[1]).toEqual(attachment)
		expect(result[2].role).toBe("user")
		expect(result[2].content[0]).toEqual({ type: "input_text", text: "another" })
		expect(result[2].content[1]).toEqual(toolCall)
	})
})

describe("openai-responses helpers", () => {
	beforeEach(() => {
		vitest.restoreAllMocks()
	})

	it("buildResponsesRequestBody builds expected fields", () => {
		const model: any = {
			id: "test-model",
			info: {
				supportsVerbosity: true,
				supportsTemperature: true,
				tiers: [{ name: "flex" }],
			},
			maxTokens: 256,
		}

		const formattedInput = [{ role: "user", content: [{ type: "input_text", text: "hi" }] }]

		const body = buildResponsesRequestBody(
			model,
			formattedInput,
			"prev-1",
			"system prompt",
			"high",
			{} as any,
			{ store: false },
			{ enableGpt5ReasoningSummary: true, openAiNativeServiceTier: "flex", modelTemperature: 0.7 },
		)

		expect(body.model).toBe("test-model")
		expect(body.instructions).toBe("system prompt")
		expect(body.input).toBe(formattedInput)
		expect(body.previous_response_id).toBe("prev-1")
		expect(body.max_output_tokens).toBe(256)
		expect(body.text?.verbosity).toBe("high")
		expect(body.temperature).toBe(0.7)
		expect(body.service_tier).toBe("flex")
		expect(body.store).toBe(false)
		expect(body.stream).toBe(true)
	})

	it("responsesSdkStream throws if client missing responses.create, otherwise returns SDK value", async () => {
		await expect(async () => {
			await responsesSdkStream({}, {})
		}).rejects.toThrow()

		const mockCreate = vitest.fn().mockResolvedValue("ok")
		const client: any = { responses: { create: mockCreate } }
		const res = await responsesSdkStream(client, { model: "m" })
		expect(res).toBe("ok")
		expect(mockCreate).toHaveBeenCalled()
	})

	it("responsesSseFetch normalizes base URLs and preserves streaming flags", async () => {
		const sampleEvent = { type: "response.text.delta", delta: "sse-hello" }
		const ssePayload = `data: ${JSON.stringify(sampleEvent)}\n\ndata: [DONE]\n\n`

		const fetchMock = vitest.fn().mockResolvedValue({
			ok: true,
			status: 200,
			statusText: "OK",
			body: createReadableStreamFromString(ssePayload, 7),
		} as any)
		vitest.stubGlobal("fetch", fetchMock)

		const requestBody = { model: "m", store: false }
		const events: any[] = []
		for await (const event of responsesSseFetch("https://api.openai.com/", "sk-test", requestBody)) {
			events.push(event)
		}

		expect(fetchMock).toHaveBeenCalledTimes(1)
		const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit]
		expect(url).toBe("https://api.openai.com/v1/responses")
		const parsed = JSON.parse(String(options?.body))
		expect(parsed.stream).toBe(true)
		expect(parsed.store).toBe(false)
		expect(parsed.model).toBe("m")

		expect(events).toHaveLength(1)
		expect(events[0]).toEqual(sampleEvent)
	})

	it("responsesSseFetch throws with detailed error when fetch fails", async () => {
		const fetchMock = vitest.fn().mockResolvedValue({
			ok: false,
			status: 503,
			statusText: "Service Unavailable",
			text: async () => "unavailable",
		})
		vitest.stubGlobal("fetch", fetchMock)

		await expect(async () => {
			for await (const _ of responsesSseFetch("https://api.openai.com/v1", "sk-test", { model: "m" })) {
				// no-op
			}
		}).rejects.toThrow(/503/)
	})

	it("processResponsesEventStream yields chunks from a single event", async () => {
		const single = { type: "response.text.delta", delta: "one" }
		const chunks: any[] = []
		for await (const chunk of processResponsesEventStream(single, baseModel as any)) {
			chunks.push(chunk)
		}
		expect(chunks).toHaveLength(1)
		expect(chunks[0]).toEqual({ type: "text", text: "one" })
	})

	it("processResponsesEventStream handles a ReadableStream input", async () => {
		const sampleEvent = { type: "response.text.delta", delta: "streamed" }
		const ssePayload = `data: ${JSON.stringify(sampleEvent)}\n\ndata: [DONE]\n\n`
		const stream = createReadableStreamFromString(ssePayload, 5)

		const chunks: any[] = []
		for await (const chunk of processResponsesEventStream(stream, baseModel as any)) {
			chunks.push(chunk)
		}

		expect(chunks).toHaveLength(1)
		expect(chunks[0]).toEqual({ type: "text", text: "streamed" })
	})

	it("processResponsesEventStream handles async iterable events with embedded stream and usage", async () => {
		const sampleEvent = { type: "response.reasoning.delta", delta: "thinking" }
		const ssePayload = `data: ${JSON.stringify(sampleEvent)}\n\ndata: [DONE]\n\n`

		const iterable = {
			async *[Symbol.asyncIterator]() {
				yield { body: createReadableStreamFromString(ssePayload, 4) }
				yield {
					type: "response.usage",
					usage: { input_tokens: 2, output_tokens: 3, cache_read_tokens: 1 },
				}
			},
		}

		const chunks: any[] = []
		for await (const chunk of processResponsesEventStream(iterable, baseModel as any)) {
			chunks.push(chunk)
		}

		expect(chunks).toHaveLength(2)
		expect(chunks[0]).toEqual({ type: "reasoning", text: "thinking" })
		expect(chunks[1]).toEqual({
			type: "usage",
			inputTokens: 2,
			outputTokens: 3,
			cacheWriteTokens: 0,
			cacheReadTokens: 1,
			totalCost: 0,
		})
	})

	it("processResponsesEventStream maps non-streaming response outputs", async () => {
		const responseEvent = {
			response: {
				output: [
					{
						type: "text",
						content: [{ type: "text", text: "final text" }],
					},
					{
						type: "reasoning",
						summary: [{ type: "summary_text", text: "summary" }],
					},
				],
			},
		}

		const chunks: any[] = []
		for await (const chunk of processResponsesEventStream(responseEvent, baseModel as any)) {
			chunks.push(chunk)
		}

		expect(chunks).toEqual([
			{ type: "text", text: "final text" },
			{ type: "reasoning", text: "summary" },
		])
	})

	it("processResponsesEventStream handles function call streaming and converts to XML", async () => {
		const events = [
			{
				type: "response.function_call_arguments.delta",
				call_id: "call-123",
				name: "search_files",
				delta: '{"path": "src", ',
			},
			{
				type: "response.function_call_arguments.delta",
				call_id: "call-123",
				delta: '"regex": "test"}',
			},
			{
				type: "response.function_call_arguments.done",
				call_id: "call-123",
				name: "search_files",
			},
		]

		// Create an async iterable from events array
		async function* eventStream() {
			for (const event of events) {
				yield event
			}
		}

		const chunks: any[] = []
		for await (const chunk of processResponsesEventStream(eventStream(), baseModel as any)) {
			chunks.push(chunk)
		}

		expect(chunks).toHaveLength(1)
		expect(chunks[0].type).toBe("text")
		expect(chunks[0].text).toContain("<search_files>")
		expect(chunks[0].text).toContain("<path>src</path>")
		expect(chunks[0].text).toContain("<regex>test</regex>")
		expect(chunks[0].text).toContain("</search_files>")
	})

	it("processResponsesEventStream handles function call with non-streaming done event", async () => {
		const event = {
			type: "response.function_call_arguments.done",
			call_id: "call-456",
			name: "read_file",
			arguments: '{"path": "src/test.ts"}',
		}

		const chunks: any[] = []
		for await (const chunk of processResponsesEventStream(event, baseModel as any)) {
			chunks.push(chunk)
		}

		expect(chunks).toHaveLength(1)
		expect(chunks[0].type).toBe("text")
		expect(chunks[0].text).toContain("<read_file>")
		expect(chunks[0].text).toContain("<path>src/test.ts</path>")
		expect(chunks[0].text).toContain("</read_file>")
	})

	it("processResponsesEventStream handles function call with empty arguments", async () => {
		const events = [
			{
				type: "response.function_call_arguments.delta",
				call_id: "call-789",
				name: "list_files",
				delta: "",
			},
			{
				type: "response.function_call_arguments.done",
				call_id: "call-789",
				name: "list_files",
			},
		]

		// Create an async iterable from events array
		async function* eventStream() {
			for (const event of events) {
				yield event
			}
		}

		const chunks: any[] = []
		for await (const chunk of processResponsesEventStream(eventStream(), baseModel as any)) {
			chunks.push(chunk)
		}

		expect(chunks).toHaveLength(1)
		expect(chunks[0].type).toBe("text")
		expect(chunks[0].text).toBe("<list_files></list_files>")
	})
})
