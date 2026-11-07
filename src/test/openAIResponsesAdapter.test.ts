import {
  chatRequestToResponsesRequest,
  responsesStreamToChatChunks,
  responsesToChatCompletion,
  usesResponsesApi
} from "../utils/servicesUtils/openAIResponsesAdapter";

describe("usesResponsesApi", () => {
  it.each(["gpt-5", "gpt-5-mini", "gpt-5.2", "gpt-5.5-pro", "gpt-5.6-sol"])(
    "%s -> true",
    model => expect(usesResponsesApi(model)).toBe(true)
  );
  it.each(["gpt-4o", "gpt-4.1", "o3-mini", "gpt-3.5-turbo"])(
    "%s -> false",
    model => expect(usesResponsesApi(model)).toBe(false)
  );
});

describe("chatRequestToResponsesRequest", () => {
  it("translates messages, tools, and params", () => {
    const body = chatRequestToResponsesRequest({
      model: "gpt-5.2",
      temperature: 0,
      max_completion_tokens: 500,
      reasoning_effort: "low",
      verbosity: "medium",
      tool_choice: { type: "function", function: { name: "get_weather" } },
      stream: true,
      stream_options: { include_usage: true },
      tools: [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Weather",
            parameters: { type: "object", properties: { city: { type: "string" } } }
          }
        }
      ],
      messages: [
        { role: "system", content: "Be brief." },
        { role: "user", content: "Weather in Paris?" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "get_weather", arguments: '{"city":"Paris"}' }
            }
          ]
        },
        { role: "tool", tool_call_id: "call_1", content: "Sunny" }
      ]
    });

    expect(body).toEqual({
      model: "gpt-5.2",
      store: false,
      temperature: 0,
      max_output_tokens: 500,
      reasoning: { effort: "low" },
      text: { verbosity: "medium" },
      tool_choice: { type: "function", name: "get_weather" },
      stream: true,
      tools: [
        {
          type: "function",
          name: "get_weather",
          description: "Weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
          strict: false
        }
      ],
      input: [
        { role: "system", content: "Be brief." },
        { role: "user", content: "Weather in Paris?" },
        {
          type: "function_call",
          call_id: "call_1",
          name: "get_weather",
          arguments: '{"city":"Paris"}'
        },
        { type: "function_call_output", call_id: "call_1", output: "Sunny" }
      ]
    });
  });

  it("pairs legacy function_call / function messages", () => {
    const { input } = chatRequestToResponsesRequest({
      model: "gpt-5",
      messages: [
        {
          role: "assistant",
          content: "",
          function_call: { name: "lookup", arguments: { id: 1 } }
        },
        { role: "function", name: "lookup", content: "found" }
      ]
    });
    expect(input[0]).toMatchObject({
      type: "function_call",
      name: "lookup",
      arguments: '{"id":1}'
    });
    expect(input[1]).toEqual({
      type: "function_call_output",
      call_id: input[0].call_id,
      output: "found"
    });
  });

  it("converts multimodal content parts", () => {
    const { input } = chatRequestToResponsesRequest({
      model: "gpt-5",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            { type: "image_url", image_url: { url: "https://x/img.png" } }
          ]
        }
      ]
    });
    expect(input[0].content).toEqual([
      { type: "input_text", text: "What is this?" },
      { type: "input_image", image_url: "https://x/img.png", detail: "auto" }
    ]);
  });
});

describe("responsesToChatCompletion", () => {
  const usage = {
    input_tokens: 10,
    output_tokens: 5,
    total_tokens: 15,
    input_tokens_details: { cached_tokens: 2 },
    output_tokens_details: { reasoning_tokens: 3 }
  };

  it("maps text output", () => {
    const result = responsesToChatCompletion({
      id: "resp_1",
      created_at: 1700000000,
      model: "gpt-5.2",
      status: "completed",
      output: [
        { type: "reasoning", summary: [] },
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hello", annotations: [] }]
        }
      ],
      usage
    });

    expect(result.object).toBe("chat.completion");
    expect(result.choices[0]!.message.content).toBe("Hello");
    expect(result.choices[0]!.finish_reason).toBe("stop");
    expect(result.usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      prompt_tokens_details: { cached_tokens: 2 },
      completion_tokens_details: { reasoning_tokens: 3 }
    });
  });

  it("maps function calls to tool_calls", () => {
    const result = responsesToChatCompletion({
      id: "resp_2",
      model: "gpt-5.2",
      status: "completed",
      output: [
        {
          type: "function_call",
          id: "fc_1",
          call_id: "call_9",
          name: "get_weather",
          arguments: '{"city":"Paris"}'
        }
      ],
      usage
    });

    expect(result.choices[0]!.finish_reason).toBe("tool_calls");
    expect(result.choices[0]!.message.content).toBeNull();
    expect(result.choices[0]!.message.tool_calls).toEqual([
      {
        id: "call_9",
        type: "function",
        function: { name: "get_weather", arguments: '{"city":"Paris"}' }
      }
    ]);
  });

  it("maps max_output_tokens truncation to length", () => {
    const result = responsesToChatCompletion({
      status: "incomplete",
      incomplete_details: { reason: "max_output_tokens" },
      output: [],
      usage
    });
    expect(result.choices[0]!.finish_reason).toBe("length");
  });
});

describe("responsesStreamToChatChunks", () => {
  async function collect(events: any[]): Promise<any[]> {
    async function* gen(): AsyncGenerator<any> {
      for (const e of events) yield e;
    }
    const out: any[] = [];
    for await (const c of responsesStreamToChatChunks(gen())) out.push(c);
    return out;
  }

  it("streams text then usage", async () => {
    const chunks = await collect([
      { type: "response.created", response: { id: "resp_1", model: "gpt-5.2", created_at: 1 } },
      { type: "response.output_text.delta", delta: "Hel" },
      { type: "response.output_text.delta", delta: "lo" },
      {
        type: "response.completed",
        response: { id: "resp_1", model: "gpt-5.2", status: "completed", usage: { input_tokens: 1, output_tokens: 2 } }
      }
    ]);

    expect(chunks.map(c => c.choices[0]?.delta)).toEqual([
      { role: "assistant", content: "Hel" },
      { content: "lo" },
      {},
      undefined
    ]);
    expect(chunks[2].choices[0].finish_reason).toBe("stop");
    expect(chunks[3].choices).toEqual([]);
    expect(chunks[3].usage.total_tokens).toBe(3);
    expect(chunks.every(c => c.object === "chat.completion.chunk")).toBe(true);
  });

  it("streams tool calls", async () => {
    const chunks = await collect([
      { type: "response.created", response: { id: "r", model: "gpt-5" } },
      {
        type: "response.output_item.added",
        output_index: 0,
        item: { type: "function_call", id: "fc_1", call_id: "call_1", name: "f", arguments: "" }
      },
      { type: "response.function_call_arguments.delta", item_id: "fc_1", output_index: 0, delta: '{"a":' },
      { type: "response.function_call_arguments.delta", item_id: "fc_1", output_index: 0, delta: "1}" },
      { type: "response.completed", response: { status: "completed", usage: {} } }
    ]);

    expect(chunks[0].choices[0].delta.tool_calls[0]).toEqual({
      index: 0,
      id: "call_1",
      type: "function",
      function: { name: "f", arguments: "" }
    });
    expect(chunks[1].choices[0].delta.tool_calls[0].function.arguments).toBe('{"a":');
    expect(chunks[2].choices[0].delta.tool_calls[0].function.arguments).toBe("1}");
    expect(chunks[3].choices[0].finish_reason).toBe("tool_calls");
  });

  it("throws on response.failed", async () => {
    await expect(
      collect([{ type: "response.failed", response: { error: { message: "boom" } } }])
    ).rejects.toThrow("boom");
  });
});
