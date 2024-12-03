const LLM_PROXY_ERROR_MESSAGES = {
  MISSING_API_KEY: "OpenAI API key is required for OpenAI models.",
  MISSING_AWS_CREDENTIALS: "AWS credentials are required for Bedrock models.",
  UNSUPPORTED_PROVIDER: "Unsupported provider.",
  MISSING_ANTHROPIC_SYSTEM_ROLE:
    "The first message must have a role of 'system' for Anthropic.",
  MESSAGES_ARRAY_CANNOT_BE_EMPTY: "Messages array cannot be empty",
  RESPONSE_OBJECT_NULL_OR_UNDEFINED: "Response object is null or undefined",
  FAILED_TO_ADAPT_RESPONSE: "Failed to adapt response",
  MISSING_MODEL: "Model is required",
};

export default LLM_PROXY_ERROR_MESSAGES;
