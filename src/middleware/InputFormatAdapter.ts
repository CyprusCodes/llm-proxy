// utils/inputFormatAdapter.ts
import { OpenAIMessages, BedrockAnthropicMessage, Providers } from "../types";
import { adaptAnthropicMessages } from "../utils/inputFormatAdapterUtils/anthropicAdapter";
import { adaptOpenAIMessages } from "../utils/inputFormatAdapterUtils/openAIAdapter";

export default class InputFormatAdapter {
  static adaptMessages(
    messages: any,
    provider: Providers
  ): {
    adaptedMessages: OpenAIMessages | BedrockAnthropicMessage[];
    systemPrompt?: string;
  } {
    switch (provider) {
      case Providers.OPENAI:
        return adaptOpenAIMessages(messages);

      case Providers.ANTHROPIC_BEDROCK:
        return adaptAnthropicMessages(messages);

      default:
        throw new Error(`Unsupported provider: ${provider}`);
    }
  }
}
