import OutputFormatAdapter from "../middleware/OutputFormatAdapter";
import { Providers } from "../types";

describe("OutputFormatAdapter", () => {
  describe("mapStopReasonToFinishReason", () => {
    it("maps Anthropic stop reasons to OpenAI finish reasons", () => {
      expect(OutputFormatAdapter.mapStopReasonToFinishReason("end_turn")).toBe(
        "stop"
      );
      expect(
        OutputFormatAdapter.mapStopReasonToFinishReason("stop_sequence")
      ).toBe("stop");
      expect(
        OutputFormatAdapter.mapStopReasonToFinishReason("max_tokens")
      ).toBe("length");
      expect(OutputFormatAdapter.mapStopReasonToFinishReason("tool_use")).toBe(
        "tool_calls"
      );
      expect(OutputFormatAdapter.mapStopReasonToFinishReason("refusal")).toBe(
        "content_filter"
      );
      expect(OutputFormatAdapter.mapStopReasonToFinishReason(undefined)).toBe(
        "stop"
      );
    });
  });

  describe("non-streaming Anthropic responses", () => {
    const baseResponse = {
      id: "msg_123",
      model: "claude-sonnet-4-6",
      usage: { input_tokens: 10, output_tokens: 20 }
    };

    it("reports a truncated response as length, not stop", async () => {
      const adapted = await OutputFormatAdapter.adaptResponse({
        response: {
          ...baseResponse,
          stop_reason: "max_tokens",
          content: [{ type: "text", text: "Let me grab that data" }]
        },
        provider: Providers.ANTHROPIC,
        isStream: false
      });

      expect(adapted.choices[0].finish_reason).toBe("length");
    });

    it("reports a completed response as stop", async () => {
      const adapted = await OutputFormatAdapter.adaptResponse({
        response: {
          ...baseResponse,
          stop_reason: "end_turn",
          content: [{ type: "text", text: "All done." }]
        },
        provider: Providers.ANTHROPIC,
        isStream: false
      });

      expect(adapted.choices[0].finish_reason).toBe("stop");
    });

    it("reports a tool call with finish_reason tool_calls and function_call payload", async () => {
      const adapted = await OutputFormatAdapter.adaptResponse({
        response: {
          ...baseResponse,
          stop_reason: "tool_use",
          content: [
            {
              type: "tool_use",
              id: "toolu_1",
              name: "get_weather",
              input: { city: "Paris" }
            }
          ]
        },
        provider: Providers.ANTHROPIC,
        isStream: false
      });

      expect(adapted.choices[0].finish_reason).toBe("tool_calls");
      expect(adapted.choices[0].message.function_call).toEqual({
        name: "get_weather",
        arguments: JSON.stringify({ city: "Paris" })
      });
    });
  });

  describe("streaming Anthropic chunks", () => {
    it("uses the observed stop_reason on the final chunk", async () => {
      const adapted = await OutputFormatAdapter.adaptResponse({
        response: { type: "message_stop" },
        provider: Providers.ANTHROPIC,
        isStream: true,
        stopReason: "max_tokens"
      });

      expect(adapted.choices[0].finish_reason).toBe("length");
    });

    it("defaults the final chunk to stop when no stop_reason is known", async () => {
      const adapted = await OutputFormatAdapter.adaptResponse({
        response: { type: "message_stop" },
        provider: Providers.ANTHROPIC,
        isStream: true
      });

      expect(adapted.choices[0].finish_reason).toBe("stop");
    });

    it("leaves finish_reason null on intermediate chunks", async () => {
      const adapted = await OutputFormatAdapter.adaptResponse({
        response: {
          type: "content_block_delta",
          delta: { type: "text_delta", text: "hello" }
        },
        provider: Providers.ANTHROPIC,
        isStream: true,
        stopReason: "end_turn"
      });

      expect(adapted.choices[0].finish_reason).toBeNull();
      expect(adapted.choices[0].delta.content).toBe("hello");
    });
  });
});
