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

const TOOL_CALL_CHUNK = {
  choices: [
    {
      index: 0,
      delta: {
        tool_calls: [
          {
            index: 0,
            id: "call_1",
            type: "function",
            function: { name: "set_item", arguments: '{"key":"a"}' }
          }
        ]
      },
      finish_reason: null
    }
  ]
};

const FINISH_CHUNK = {
  choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
};

describe("generateLLMStreamResponse with gemini-3+", () => {
  beforeEach(() => {
    mockCreate.mockReset();
  });

  it("passes gemini-3 stream chunks through unchanged", async () => {
    mockCreate.mockImplementationOnce(() =>
      chunkStream([TOOL_CALL_CHUNK, FINISH_CHUNK])
    );

    const stream = await generateLLMStreamResponse({
      model: "gemini-3.8-flash",
      messages: [
        { role: "user", content: "remember a" },
        {
          role: "assistant",
          content: null,
          function_call: { name: "set_item", arguments: '{"key":"b"}' }
        },
        { role: "function", name: "set_item", content: "b saved" }
      ] as any,
      credentials: {
        openAICompatibleProviderConfig: {
          openAICompatibleProviderKey: "test-key",
          baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai/"
        }
      }
    });

    const chunks: any[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual([TOOL_CALL_CHUNK, FINISH_CHUNK]);
  });
});
