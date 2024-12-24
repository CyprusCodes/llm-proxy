function convertLlamaToOpenAIStream(chunk: any): any {
  const isStop = chunk.stop_reason !== null;
  return {
    id: chunk.id, // Use input chunk's ID
    object: "chat.completion.chunk",
    created: chunk.created || Math.floor(Date.now() / 1000),
    model: chunk.model || "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        delta: {
          content: chunk.generation || chunk.delta?.content || "",
        },
        logprobs: null,
        finish_reason: isStop ? "stop" : null,
      },
    ],
    logprobs: null,
    finish_reason: chunk.stop_reason,
    usage: isStop
      ? {
          prompt_tokens: chunk.prompt_token_count,
          completion_tokens: chunk.generation_token_count,
          total_tokens: chunk.prompt_token_count + chunk.generation_token_count,
          prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
          completion_tokens_details: {
            reasoning_tokens: 0,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
          },
        }
      : null,
  };
}

export default convertLlamaToOpenAIStream;
