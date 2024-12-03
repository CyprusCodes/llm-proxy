import { BedrockAnthropicMessages, Messages, OpenAIMessages } from "llm-proxy";
import LLM_PROXY_ERROR_MESSAGES from "../constants/errorMessages";
import { Providers } from "../types";
import adaptAnthropicMessages from "../utils/inputFormatAdapterUtils/adaptAnthropicMessages";
import adaptOpenAIMessages from "../utils/inputFormatAdapterUtils/adaptOpenAIMessages";

/*
system prompt is optional because in case of OpenAI, 
we don't return it as a separate field but its part of the messages
*/

interface AdaptMessagesOutput {
  adaptedMessages: unknown;
  systemPrompt?: string;
}

interface AdaptMessagesInput {
  messages: Messages;
  provider: Providers;
}

export default class InputFormatAdapter {
  static adaptMessages({
    messages,
    provider,
  }: AdaptMessagesInput): AdaptMessagesOutput {
    switch (provider) {
      case Providers.OPENAI:
        return {
          adaptedMessages: adaptOpenAIMessages(messages as OpenAIMessages),
        };

      case Providers.ANTHROPIC_BEDROCK:
        return adaptAnthropicMessages(messages as BedrockAnthropicMessages);

      default:
        throw new Error(LLM_PROXY_ERROR_MESSAGES.UNSUPPORTED_PROVIDER);
    }
  }
}
