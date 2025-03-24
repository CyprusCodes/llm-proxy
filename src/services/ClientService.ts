import { BedrockAnthropicParsedChunk, LLMResponse, Messages } from "../types";

export interface ClientService {
  generateCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
    systemPrompt,
  }: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
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
    systemPrompt,
  }: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any; // TODO: Define the correct type it might be looking like below
    systemPrompt?: string;
  }): AsyncGenerator<BedrockAnthropicParsedChunk, void, unknown>;
}
