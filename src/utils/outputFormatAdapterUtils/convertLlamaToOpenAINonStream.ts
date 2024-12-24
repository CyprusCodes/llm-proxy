function convertLlamaToOpenAINonStream(chunk) {
  return {
    id: chunk.id || "default_id",
    object: "text_completion",
    created: Date.now(),
    model: "aws-bedrock-llama",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant", // Default role since it's not provided in the input
          content: chunk.generation.trim() // Extract content from 'generation'
        },
        logprobs: null,
        finish_reason: chunk.stop_reason || null // Use stop reason from input if available
      }
    ],
    usage: {
      prompt_tokens: chunk.prompt_token_count || 0, // Extract prompt tokens
      completion_tokens: chunk.generation_token_count || 0, // Extract generation tokens
      total_tokens:
        (chunk.prompt_token_count || 0) + (chunk.generation_token_count || 0), // Calculate total tokens
      prompt_tokens_details: { cached_tokens: 0 }, // Placeholder details
      completion_tokens_details: { reasoning_tokens: 0 } // Placeholder details
    },
    system_fingerprint: chunk.system_fingerprint || "default_fingerprint" // Default fingerprint if not provided
  };
}

export default convertLlamaToOpenAINonStream;
