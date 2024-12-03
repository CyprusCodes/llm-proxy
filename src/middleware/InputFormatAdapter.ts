import { OpenAIMessages } from "llm-proxy";
import LLM_PROXY_ERROR_MESSAGES from "../constants/errorMessages";
import { Providers } from "../types";
import adaptAnthropicMessages from "../utils/inputFormatAdapterUtils/adaptAnthropicMessages";
import adaptOpenAIMessages from "../utils/inputFormatAdapterUtils/adaptOpenAIMessages";

export default class InputFormatAdapter {
  static adaptMessages(
    messages: OpenAIMessages,
    provider: Providers
  ): {
    adaptedMessages: any;
    systemPrompt?: string;
  } {
    switch (provider) {
      case Providers.OPENAI:
        return {
          adaptedMessages: adaptOpenAIMessages(messages),
        };

      case Providers.ANTHROPIC_BEDROCK:
        return adaptAnthropicMessages(messages);

      default:
        throw new Error(LLM_PROXY_ERROR_MESSAGES.UNSUPPORTED_PROVIDER);
    }
  }
}
