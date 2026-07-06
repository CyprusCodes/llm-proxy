/**
 * End-to-end tests (with a mocked OpenAI SDK) for the future-promise
 * recovery flow: when the model says it will do something but ends its turn
 * without a tool call, the proxy re-runs the request with tool choice forced
 * and appends the resulting tool call to the stream.
 */
import { generateLLMStreamResponse } from "../index";

const mockCreate = jest.fn();

jest.mock("openai", () => ({
  __esModule: true,
  default: jest.fn().mockImplementation(() => ({
    chat: { completions: { create: mockCreate } }
  }))
}));

async function* chunkStream(chunks: any[]) {
  for (const chunk of chunks) {
    yield chunk;
  }
}

function textChunk(content: string) {
  return {
    choices: [{ index: 0, delta: { content }, finish_reason: null }]
  };
}

function finishChunk(finishReason: string) {
  return {
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
  };
}

function toolCallChunk() {
  return {
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              id: "call_abc123",
              type: "function",
              function: { name: "get_data", arguments: '{"id":1}' }
            }
          ]
        },
        finish_reason: null
      }
    ]
  };
}

const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_data",
      description: "Fetch data by id",
      parameters: { type: "object", properties: { id: { type: "number" } } }
    }
  }
];

const PARAMS = {
  model: "gpt-4o",
  messages: [{ role: "user" as const, content: "get my data" }],
  functions: TOOLS,
  credentials: { apiKey: "test-key" }
};

async function collect(stream: AsyncGenerator<any>) {
  const chunks: any[] = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return chunks;
}

describe("future-promise recovery (OpenAI streaming)", () => {
  beforeEach(() => {
    mockCreate.mockReset();
  });

  it("forces a tool call when the model promises an action and stops", async () => {
    mockCreate
      // First request: model promises but never calls the tool
      .mockImplementationOnce(() =>
        chunkStream([
          textChunk("Let me grab that data for you."),
          finishChunk("stop")
        ])
      )
      // Forced retry: model emits the tool call
      .mockImplementationOnce(() =>
        chunkStream([toolCallChunk(), finishChunk("tool_calls")])
      );

    const stream = await generateLLMStreamResponse(PARAMS);
    const chunks = await collect(stream);

    expect(mockCreate).toHaveBeenCalledTimes(2);
    expect(mockCreate.mock.calls[0][0].tool_choice).toBeUndefined();
    expect(mockCreate.mock.calls[1][0].tool_choice).toBe("required");

    // Original text still streamed through
    const texts = chunks
      .map(c => c.choices?.[0]?.delta?.content)
      .filter(Boolean);
    expect(texts.join("")).toContain("Let me grab that data");

    // Tool call from the forced retry is appended
    const toolCalls = chunks
      .map(c => c.choices?.[0]?.delta?.tool_calls)
      .filter(Boolean)
      .flat();
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].function.name).toBe("get_data");
  });

  it("does not retry when the model gave a real answer", async () => {
    mockCreate.mockImplementationOnce(() =>
      chunkStream([
        textChunk("Your data: 42 records found."),
        finishChunk("stop")
      ])
    );

    const stream = await generateLLMStreamResponse(PARAMS);
    await collect(stream);

    expect(mockCreate).toHaveBeenCalledTimes(1);
  });

  it("does not retry when the model already called a tool", async () => {
    mockCreate.mockImplementationOnce(() =>
      chunkStream([
        textChunk("Let me grab that data for you."),
        toolCallChunk(),
        finishChunk("tool_calls")
      ])
    );

    const stream = await generateLLMStreamResponse(PARAMS);
    await collect(stream);

    expect(mockCreate).toHaveBeenCalledTimes(1);
  });

  it("does not retry when no tools were provided", async () => {
    mockCreate.mockImplementationOnce(() =>
      chunkStream([
        textChunk("Let me check on that for you."),
        finishChunk("stop")
      ])
    );

    const stream = await generateLLMStreamResponse({
      ...PARAMS,
      functions: undefined
    });
    await collect(stream);

    expect(mockCreate).toHaveBeenCalledTimes(1);
  });

  it("still delivers the original response if the forced retry fails", async () => {
    mockCreate
      .mockImplementationOnce(() =>
        chunkStream([
          textChunk("Let me grab that data for you."),
          finishChunk("stop")
        ])
      )
      .mockImplementationOnce(() => {
        throw new Error("provider unavailable");
      });

    const stream = await generateLLMStreamResponse(PARAMS);
    const chunks = await collect(stream);

    const texts = chunks
      .map(c => c.choices?.[0]?.delta?.content)
      .filter(Boolean);
    expect(texts.join("")).toContain("Let me grab that data");
  });

  it("filters duplicate promise text out of the forced retry stream", async () => {
    mockCreate
      .mockImplementationOnce(() =>
        chunkStream([
          textChunk("Let me grab that data for you."),
          finishChunk("stop")
        ])
      )
      .mockImplementationOnce(() =>
        chunkStream([
          textChunk("Let me grab that data for you."), // duplicate narration
          toolCallChunk(),
          finishChunk("tool_calls")
        ])
      );

    const stream = await generateLLMStreamResponse(PARAMS);
    const chunks = await collect(stream);

    const texts = chunks
      .map(c => c.choices?.[0]?.delta?.content)
      .filter(Boolean);
    // The promise text appears exactly once (from the original stream)
    expect(texts.join("")).toBe("Let me grab that data for you.");
  });
});
