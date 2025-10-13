import { describe, it, expect, vitest, beforeEach } from "vitest"
import OpenAI from "openai"
import { Anthropic } from "@anthropic-ai/sdk"
import { BaseOpenAiCompatibleProvider } from "../base-openai-compatible-provider"

// Minimal mock model info shape used in tests
const mockModelInfo = {
	info: {
		// Provide the minimal required ModelInfo fields expected by BaseOpenAiCompatibleProvider
		contextWindow: 4096,
		supportsPromptCache: false,
		// Preserve the fields used by these tests
		supportsVerbosity: false,
		supportsTemperature: true,
		tiers: [],
	},
	maxTokens: 128,
	id: "test-model",
}

// Create a small TestHandler subclass for exercising the base provider behavior
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
		// Return an object shaped like other handlers expect: { id, info, ...params }
		return { id: "test-model", info: mockModelInfo.info, maxTokens: mockModelInfo.maxTokens }
	}
}

describe("BaseOpenAiCompatibleProvider Responses flow", () => {
	let mockResponsesCreate: any

	beforeEach(() => {
		vitest.restoreAllMocks()
		mockResponsesCreate = vitest.fn()
		// Mock the OpenAI constructor to return an instance with responses.create
		vitest.mock("openai", () => {
			const mockConstructor = vitest.fn().mockImplementation(() => ({
				responses: {
					create: mockResponsesCreate,
				},
			}))
			return { __esModule: true, default: mockConstructor }
		})
	})

	it("uses SDK responses.create when openAiUseResponses is enabled", async () => {
		// Make responses.create return an async iterable of one event
		mockResponsesCreate.mockResolvedValue({
			[Symbol.asyncIterator]: async function* () {
				yield { type: "response.text.delta", delta: "sdk chunk" }
			},
		})

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

		expect(mockResponsesCreate).toHaveBeenCalled()
		const calledWith = mockResponsesCreate.mock.calls[0][0]
		expect(calledWith.model).toBe("test-model")
		expect(calledWith.instructions).toBe(systemPrompt)
		// Ensure we received the streamed text chunk
		const textChunks = results.filter((r) => r.type === "text")
		expect(textChunks.length).toBeGreaterThan(0)
		expect(textChunks[0].text).toBe("sdk chunk")
	})

	it("falls back to SSE fetch when SDK throws or returns non-iterable", async () => {
		// Make SDK throw
		mockResponsesCreate.mockRejectedValue(new Error("SDK failure"))

		// Prepare SSE payload reader similar to other tests
		const sampleEvent = { type: "response.text.delta", delta: "sse chunk" }
		const ssePayload = `data: ${JSON.stringify(sampleEvent)}\n\ndata: [DONE]\n\n`
		const encoder = new TextEncoder()
		const chunks: Uint8Array[] = []
		const chunkSize = 8
		for (let i = 0; i < ssePayload.length; i += chunkSize) {
			chunks.push(encoder.encode(ssePayload.slice(i, i + chunkSize)))
		}
		let idx = 0
		const reader = {
			async read() {
				if (idx >= chunks.length) return { done: true, value: new Uint8Array() }
				return { done: false, value: chunks[idx++] }
			},
			releaseLock() {},
		}

		// Mock global.fetch to return a Response-like object with .body.getReader()
		vitest.stubGlobal(
			"fetch",
			vitest.fn().mockResolvedValue({
				body: {
					getReader: () => reader,
				},
			} as any),
		)

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

		// Since SDK failed, fetch should have been used (our stub will return parsed event)
		// and the resulting streamed text should be present
		const textChunks = results.filter((r) => r.type === "text")
		expect(textChunks.length).toBeGreaterThan(0)
		expect(textChunks[0].text).toBe("sse chunk")
	})
})
