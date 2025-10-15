import { describe, it, expect, beforeEach, afterEach, vi, type Mock } from "vitest"
import { Anthropic } from "@anthropic-ai/sdk"
import { BaseOpenAiCompatibleProvider } from "../base-openai-compatible-provider"

// Mutable reference used by the mocked OpenAI constructor.
let currentResponsesCreate: Mock

vi.mock("openai", () => {
	class MockOpenAI {
		responses: { create: (...args: any[]) => Promise<any> }

		constructor() {
			this.responses = {
				create: (...args: any[]) => {
					if (!currentResponsesCreate) {
						throw new Error("mockResponsesCreate not initialised")
					}
					return currentResponsesCreate(...args)
				},
			}
		}
	}

	return { __esModule: true, default: MockOpenAI }
})

const mockModelInfo = {
	info: {
		contextWindow: 4096,
		supportsPromptCache: false,
		supportsVerbosity: false,
		supportsTemperature: true,
		tiers: [],
	},
	maxTokens: 128,
	id: "test-model",
}

class TestHandler extends BaseOpenAiCompatibleProvider<"test-model"> {
	constructor(options: any) {
		super({
			...options,
			providerName: "test-provider",
			baseURL: options.baseURL || "https://api.openai.com/v1",
			defaultProviderModelId: "test-model",
			providerModels: { "test-model": (mockModelInfo as any).info },
			defaultTemperature: 0.5,
		})
	}

	override getModel() {
		return { id: "test-model", info: mockModelInfo.info, maxTokens: mockModelInfo.maxTokens }
	}
}

function createSseReadable(payload: string, chunkSize = payload.length) {
	const encoder = new TextEncoder()
	const chunks: Uint8Array[] = []
	for (let i = 0; i < payload.length; i += chunkSize) {
		chunks.push(encoder.encode(payload.slice(i, i + chunkSize)))
	}

	let idx = 0
	const reader = {
		async read() {
			if (idx >= chunks.length) {
				return { done: true, value: new Uint8Array() }
			}
			return { done: false, value: chunks[idx++] }
		},
		releaseLock() {},
	}

	return {
		getReader: () => reader,
	} as ReadableStream<Uint8Array>
}

function stubFetchWithSse(events: unknown[], chunkSize = 8) {
	const payload = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n"

	const fetchMock = vi.fn().mockResolvedValue({
		ok: true,
		status: 200,
		statusText: "OK",
		body: createSseReadable(payload, chunkSize),
	} as any)

	vi.stubGlobal("fetch", fetchMock)
	return fetchMock
}

describe("BaseOpenAiCompatibleProvider Responses flow", () => {
	beforeEach(() => {
		currentResponsesCreate = vi.fn()
	})

	afterEach(() => {
		vi.restoreAllMocks()
	})

	it("uses SDK responses.create when openAiUseResponses is enabled", async () => {
		currentResponsesCreate.mockResolvedValue({
			async *[Symbol.asyncIterator]() {
				yield { type: "response.text.delta", delta: "sdk chunk" }
			},
		})

		const fetchMock = stubFetchWithSse([], 4)

		const handler = new TestHandler({
			apiKey: "sk-test",
			apiModelId: "test-model",
			openAiUseResponses: true,
		})

		const systemPrompt = "system prompt"
		const messages: Anthropic.Messages.MessageParam[] = [
			{ role: "user", content: [{ type: "text", text: "hello" }] } as any,
		]

		const results: any[] = []
		for await (const chunk of handler.createMessage(systemPrompt, messages)) {
			results.push(chunk)
		}

		expect(currentResponsesCreate).toHaveBeenCalled()
		const calledWith = currentResponsesCreate.mock.calls[0][0]
		expect(calledWith.model).toBe("test-model")
		expect(calledWith.instructions).toBe(systemPrompt)

		const textChunks = results.filter((r) => r.type === "text")
		expect(textChunks.length).toBeGreaterThan(0)
		expect(textChunks[0].text).toBe("sdk chunk")
		expect(fetchMock).not.toHaveBeenCalled()
	})

	it("falls back to SSE fetch when SDK throws", async () => {
		currentResponsesCreate.mockRejectedValue(new Error("SDK failure"))

		const fetchMock = stubFetchWithSse([{ type: "response.text.delta", delta: "sse chunk" }], 8)

		const handler = new TestHandler({
			apiKey: "sk-test",
			apiModelId: "test-model",
			openAiUseResponses: true,
			baseURL: "https://api.openai.com/v1",
		})

		const systemPrompt = "system prompt"
		const messages: Anthropic.Messages.MessageParam[] = [
			{ role: "user", content: [{ type: "text", text: "hello" }] } as any,
		]

		const results: any[] = []
		for await (const chunk of handler.createMessage(systemPrompt, messages)) {
			results.push(chunk)
		}

		expect(currentResponsesCreate).toHaveBeenCalled()
		expect(fetchMock).toHaveBeenCalled()

		const textChunks = results.filter((r) => r.type === "text")
		expect(textChunks.length).toBeGreaterThan(0)
		expect(textChunks[0].text).toBe("sse chunk")
	})

	it("falls back to SSE fetch when SDK resolves a non-iterable result", async () => {
		currentResponsesCreate.mockResolvedValue({})

		const fetchMock = stubFetchWithSse([{ type: "response.text.delta", delta: "fallback chunk" }], 6)

		const handler = new TestHandler({
			apiKey: "sk-test",
			apiModelId: "test-model",
			openAiUseResponses: true,
			baseURL: "https://api.openai.com/v1",
		})

		const results: any[] = []
		for await (const chunk of handler.createMessage("system prompt", [
			{ role: "user", content: [{ type: "text", text: "hello" }] } as any,
		])) {
			results.push(chunk)
		}

		expect(currentResponsesCreate).toHaveBeenCalled()
		expect(fetchMock).toHaveBeenCalled()

		const textChunks = results.filter((r) => r.type === "text")
		expect(textChunks).toHaveLength(1)
		expect(textChunks[0].text).toBe("fallback chunk")
	})
})
