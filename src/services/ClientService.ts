import { BedrockAnthropicParsedChunk, LLMResponse, Messages } from "../types";

export interface ClientService {
  generateCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
    systemPrompt,
    toolChoice,
  }: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any; // TODO: Define the correct type
    systemPrompt?: string;
    // Provider-native tool choice ("required" for OpenAI-style APIs,
    // { type: "any" } for Anthropic-style APIs). Used to force a tool call.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toolChoice?: any;
  }): Promise<LLMResponse>;

  generateStreamCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
    systemPrompt,
    toolChoice,
  }: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any; // TODO: Define the correct type it might be looking like below
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toolChoice?: any;
  }): AsyncGenerator<BedrockAnthropicParsedChunk, void, unknown>;
}
