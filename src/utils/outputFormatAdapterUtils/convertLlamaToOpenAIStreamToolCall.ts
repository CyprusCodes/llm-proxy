// @ts-ignore: Ignore the any type in the msg
function convertLlamaToOpenAIStreamToolCall(chunk) {
  // @ts-ignore: Ignore the any type in the msg
  function parseFunctionCall(generation: string) {
    const functionCallRegex =
      /<function>\s*\{\s*"function_name":\s*"([^"]+)",\s*"parameters":\s*(\{.*?\})\s*\}/s;
    const match = generation.match(functionCallRegex);

    if (match) {
      const functionCall = {
        name: match[1],
        arguments: match[2],
      };
      return functionCall;
    }
    return null;
  }

  const functionCall = parseFunctionCall(chunk.generation);

  return {
    id: `stream-${Date.now()}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: "aws-bedrock-llama3x",
    system_fingerprint: null,
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          content: null,
          function_call: functionCall || { name: null, arguments: "{}" }, // Default empty arguments
          refusal: null,
        },
        logprobs: null,
        finish_reason: chunk.stop_reason || null,
      },
    ],
    usage: null,
  };
}

export default convertLlamaToOpenAIStreamToolCall;
