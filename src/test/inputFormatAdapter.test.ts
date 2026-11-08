import InputFormatAdapter from "../middleware/InputFormatAdapter";
import { chatRequestToResponsesRequest } from "../utils/servicesUtils/openAIResponsesAdapter";
import { Providers } from "../types";

const history = [
  { role: "system", content: "You are helpful." },
  { role: "user", content: "remember file_2015" },
  {
    role: "assistant",
    content: null,
    function_call: { name: "set_item", arguments: '{"key":"file_2015","value":"a"}' }
  },
  { role: "function", name: "set_item", content: "file_2015 saved" }
];

describe("InputFormatAdapter OpenAI function call history", () => {
  it("does not invent assistant text for a past function call on gpt-5.x", () => {
    const { adaptedMessages } = InputFormatAdapter.adaptMessages(
      history,
      Providers.OPENAI,
      "gpt-5.6-terra"
    );
    const body = chatRequestToResponsesRequest({
      model: "gpt-5.6-terra",
      messages: adaptedMessages
    });

    expect(body.input).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "remember file_2015" },
      expect.objectContaining({
        type: "function_call",
        name: "set_item",
        arguments: '{"key":"file_2015","value":"a"}'
      }),
      expect.objectContaining({
        type: "function_call_output",
        output: "file_2015 saved"
      })
    ]);
  });

  it("keeps the function call with null content for older models", () => {
    const { adaptedMessages } = InputFormatAdapter.adaptMessages(
      history,
      Providers.OPENAI,
      "gpt-4o"
    );

    expect(adaptedMessages[2]).toEqual({
      role: "assistant",
      content: null,
      function_call: { name: "set_item", arguments: '{"key":"file_2015","value":"a"}' }
    });
  });
});
