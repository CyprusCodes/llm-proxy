import {
  BedrockAnthropicSupportedLLMs,
  Llama3_1SupportedLLMs,
  OpenAISupportedLLMs,
  Providers,
} from "../types";

export default class ProviderFinder {
  static getProvider(model: string): Providers {
    // Check if the model belongs to OpenAI-supported LLMs
    if (
      Object.values(OpenAISupportedLLMs).includes(model as OpenAISupportedLLMs)
    ) {
      return Providers.OPENAI;
    }
    // Check if the model belongs to Bedrock Anthropic-supported LLMs
    if (
      Object.values(BedrockAnthropicSupportedLLMs).includes(
        model as BedrockAnthropicSupportedLLMs
      )
    ) {
      return Providers.ANTHROPIC_BEDROCK;
    }
    if (
      Object.values(Llama3_1SupportedLLMs).includes(
        model as Llama3_1SupportedLLMs
      )
    ) {
      return Providers.LLAMA_3_1_BEDROCK;
    }
    // Throw an error for unsupported models
    throw new Error(`Unsupported model: ${model}`);
  }
}
