import {
  BedrockAnthropicParsedChunk,
  LLMResponse,
  Messages,
  OpenAIReasoningType,
} from "../types";

export interface ClientService {
  generateCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
    reasoning,
    systemPrompt,
  }: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    reasoning?: OpenAIReasoningType;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any; // TODO: Define the correct type
    systemPrompt?: string;
  }): Promise<LLMResponse>;

  generateStreamCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
    reasoning,
    systemPrompt,
  }: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    reasoning?: OpenAIReasoningType;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any; // TODO: Define the correct type it might be looking like below
    systemPrompt?: string;
  }): AsyncGenerator<BedrockAnthropicParsedChunk, void, unknown>;
}
