import { describe, it, expect, vitest, beforeEach } from "vitest"
import {
	buildResponsesRequestBody,
	responsesSdkStream,
	responsesSseFetch,
	processResponsesEventStream,
} from "../utils/openai-responses"

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
	})

	it("responsesSdkStream throws if client missing responses.create, otherwise returns SDK value", async () => {
		await expect(async () => {
			// client without responses.create
			await responsesSdkStream({}, {})
		}).rejects.toThrow()

		const mockCreate = vitest.fn().mockResolvedValue("ok")
		const client: any = { responses: { create: mockCreate } }
		const res = await responsesSdkStream(client, { model: "m" })
		expect(res).toBe("ok")
		expect(mockCreate).toHaveBeenCalled()
	})

	it("responsesSseFetch yields parsed SSE events from fetch body", async () => {
		const sampleEvent = { type: "response.text.delta", delta: "sse-hello" }
		const ssePayload = `data: ${JSON.stringify(sampleEvent)}\n\ndata: [DONE]\n\n`
		const encoder = new TextEncoder()
		const chunks: Uint8Array[] = []
		const chunkSize = 10
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

		vitest.stubGlobal(
			"fetch",
			vitest.fn().mockResolvedValue({
				ok: true,
				status: 200,
				body: {
					getReader: () => reader,
				},
			} as any),
		)

		const gen = responsesSseFetch("https://api.openai.com/v1", "sk-test", { model: "m", input: [] })
		const events: any[] = []
		for await (const ev of gen) {
			events.push(ev)
		}

		expect(events.length).toBeGreaterThan(0)
		expect(events[0].type).toBe("response.text.delta")
		expect(events[0].delta).toBe("sse-hello")
	})

	it("processResponsesEventStream handles single event and async iterable", async () => {
		// single event object
		const single = { type: "response.text.delta", delta: "one" }
		const out1: any[] = []
		for await (const chunk of processResponsesEventStream(single, { id: "m" } as any)) {
			out1.push(chunk)
		}
		expect(out1.length).toBeGreaterThan(0)
		expect(out1[0].type).toBe("text")
		expect(out1[0].text).toBe("one")

		// async iterable of events
		async function* source() {
			yield { type: "response.text.delta", delta: "a" }
			yield { type: "response.reasoning.delta", delta: "reason" }
		}
		const out2: any[] = []
		for await (const chunk of processResponsesEventStream(source(), { id: "m" } as any)) {
			out2.push(chunk)
		}
		// should include both text and reasoning chunks
		expect(out2.some((c) => c.type === "text" && c.text === "a")).toBe(true)
		expect(out2.some((c) => c.type === "reasoning" && c.text === "reason")).toBe(true)
	})
})
