function convertLlamaToOpenAIStreamToolCall(chunk) {
  function parseFunctionCall(generation) {
    const functionCallRegex = /<function>\s*\{\s*"function_name":\s*"([^"]+)",\s*"parameter_name":\s*(\{[^}]+\})\s*\}<\/function>/;
    const match = generation.match(functionCallRegex);

    if (match) {
      const functionCall = {
        name: match[1],
        arguments: match[2] ? JSON.stringify(JSON.parse(match[2])) : ""
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
          function_call: functionCall || { name: null, arguments: "" },
          refusal: null
        },
        logprobs: null,
        finish_reason: chunk.stop_reason || null
      }
    ],
    usage: null
  };
}

export default convertLlamaToOpenAIStreamToolCall;
