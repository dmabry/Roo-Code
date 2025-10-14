import { describe, it, expect, vi, beforeEach } from "vitest"
import { Anthropic } from "@anthropic-ai/sdk"
import { BaseOpenAiCompatibleProvider } from "../base-openai-compatible-provider"
import type { ApiHandlerOptions } from "../../../shared/api"
import type { ModelInfo } from "@roo-code/types"

// Create a concrete test implementation of the abstract base class
class TestOpenAiProvider extends BaseOpenAiCompatibleProvider<"test-model"> {
	constructor(options: ApiHandlerOptions) {
		super({
			...options,
			providerName: "Test Provider",
			baseURL: "https://api.test.com",
			defaultProviderModelId: "test-model",
			providerModels: {
				"test-model": {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsImages: false,
					supportsPromptCache: false,
				} as ModelInfo,
			},
		})
	}
}

describe("BaseOpenAiCompatibleProvider", () => {
	let handler: TestOpenAiProvider
	let mockOptions: ApiHandlerOptions

	beforeEach(() => {
		mockOptions = {
			apiKey: "test-api-key",
		}
		handler = new TestOpenAiProvider(mockOptions)
	})

	describe("reasoning support", () => {
		it("should handle delta.reasoning in streaming responses", async () => {
			const systemPrompt = "Test system prompt"
			const messages: Anthropic.Messages.MessageParam[] = [
				{
					role: "user",
					content: "Test message",
				},
			]

			// Mock the OpenAI client's streaming response
			const mockStream = async function* () {
				// First chunk with reasoning
				yield {
					id: "chatcmpl-test",
					object: "chat.completion.chunk",
					created: Date.now(),
					model: "test-model",
					choices: [
						{
							index: 0,
							delta: {
								reasoning: "Let me think about this...",
							},
							finish_reason: null,
						},
					],
				}
				// Second chunk with more reasoning
				yield {
					id: "chatcmpl-test",
					object: "chat.completion.chunk",
					created: Date.now(),
					model: "test-model",
					choices: [
						{
							index: 0,
							delta: {
								reasoning: " result.",
							},
							finish_reason: null,
						},
					],
				}
				// Third chunk with actual content
				yield {
					id: "chatcmpl-test",
					object: "chat.completion.chunk",
					created: Date.now(),
					model: "test-model",
					choices: [
						{
							index: 0,
							delta: {
								content: "Here is my response",
							},
							finish_reason: null,
						},
					],
				}
				// Final chunk with usage
				yield {
					id: "chatcmpl-test",
					object: "chat.completion.chunk",
					created: Date.now(),
					model: "test-model",
					choices: [
						{
							index: 0,
							delta: {},
							finish_reason: "stop",
						},
					],
					usage: {
						prompt_tokens: 10,
						completion_tokens: 20,
						total_tokens: 30,
					},
				}
			}

			// Mock the createStream method
			vi.spyOn(handler as any, "createStream").mockResolvedValue(mockStream())

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: any[] = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			expect(chunks).toHaveLength(4)

			// Check reasoning chunks
			expect(chunks[0]).toEqual({
				type: "reasoning",
				text: "Let me think about this...",
			})
			expect(chunks[1]).toEqual({
				type: "reasoning",
				text: " result.",
			})

			// Check text chunk
			expect(chunks[2]).toEqual({
				type: "text",
				text: "Here is my response",
			})

			// Check usage chunk
			expect(chunks[3]).toEqual({
				type: "usage",
				inputTokens: 10,
				outputTokens: 20,
			})
		})

		it("should handle both reasoning and content in the same chunk", async () => {
			const systemPrompt = "Test system prompt"
			const messages: Anthropic.Messages.MessageParam[] = [
				{
					role: "user",
					content: "Test message",
				},
			]

			const mockStream = async function* () {
				yield {
					id: "chatcmpl-test",
					object: "chat.completion.chunk",
					created: Date.now(),
					model: "test-model",
					choices: [
						{
							index: 0,
							delta: {
								reasoning: "thinking...",
								content: "response",
							},
							finish_reason: null,
						},
					],
				}
			}

			vi.spyOn(handler as any, "createStream").mockResolvedValue(mockStream())

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: any[] = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			expect(chunks).toHaveLength(2)
			expect(chunks[0]).toEqual({
				type: "reasoning",
				text: "thinking...",
			})
			expect(chunks[1]).toEqual({
				type: "text",
				text: "response",
			})
		})

		it("should handle responses with only content (no reasoning)", async () => {
			const systemPrompt = "Test system prompt"
			const messages: Anthropic.Messages.MessageParam[] = [
				{
					role: "user",
					content: "Test message",
				},
			]

			const mockStream = async function* () {
				yield {
					id: "chatcmpl-test",
					object: "chat.completion.chunk",
					created: Date.now(),
					model: "test-model",
					choices: [
						{
							index: 0,
							delta: {
								content: "Standard response",
							},
							finish_reason: null,
						},
					],
				}
			}

			vi.spyOn(handler as any, "createStream").mockResolvedValue(mockStream())

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: any[] = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			expect(chunks).toHaveLength(1)
			expect(chunks[0]).toEqual({
				type: "text",
				text: "Standard response",
			})
		})
	})
})
