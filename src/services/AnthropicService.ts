import Anthropic from "@anthropic-ai/sdk";
import {
  BedrockAnthropicParsedChunk,
  BedrockAnthropicResponse,
  Messages,
} from "../types";
import { ClientService } from "./ClientService";
import replaceKeyInObjects from "../utils/servicesUtils/replaceObjectKey";

export default class AnthropicService implements ClientService {
  private client: Anthropic;

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  async generateCompletion(params: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    systemPrompt?: string;
  }): Promise<BedrockAnthropicResponse> {
    const { messages, model, max_tokens, temperature, systemPrompt, tools } =
      params;

    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for AnthropicService")
      );
    }

    const validatedTools = replaceKeyInObjects(
      tools,
      "parameters",
      "input_schema"
    ) as unknown as Anthropic.Tool[] | undefined;
    const hasTools =
      validatedTools &&
      Array.isArray(validatedTools) &&
      validatedTools.length > 0;

    const createParams: Anthropic.MessageCreateParamsNonStreaming = {
      model,
      max_tokens: max_tokens ?? 1024,
      temperature: temperature ?? 0,
      system: systemPrompt ?? "",
      messages: messages as Anthropic.MessageParam[],
      ...(hasTools ? { tools: validatedTools } : {}),
    };

    const response = await this.client.messages.create(createParams);

    return response as unknown as BedrockAnthropicResponse;
  }

  // eslint-disable-next-line consistent-return
  async *generateStreamCompletion(params: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    systemPrompt?: string;
  }): AsyncGenerator<BedrockAnthropicParsedChunk, void, unknown> {
    const { messages, model, max_tokens, temperature, tools, systemPrompt } =
      params;

    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for AnthropicService")
      );
    }

    const validatedTools = replaceKeyInObjects(
      tools,
      "parameters",
      "input_schema"
    ) as unknown as Anthropic.Tool[] | undefined;
    const hasTools =
      validatedTools &&
      Array.isArray(validatedTools) &&
      validatedTools.length > 0;

    const createParams: Anthropic.MessageCreateParamsStreaming = {
      model,
      max_tokens: max_tokens ?? 1024,
      temperature: temperature ?? 0,
      system: systemPrompt ?? "",
      messages: messages as Anthropic.MessageParam[],
      stream: true,
      ...(hasTools ? { tools: validatedTools } : {}),
    };

    const stream = await this.client.messages.create(createParams);

    let inputTokens = 0;
    let outputTokens = 0;

    for await (const event of stream) {
      if (event.type === "message_start" && event.message?.usage) {
        inputTokens = event.message.usage.input_tokens ?? 0;
        outputTokens = event.message.usage.output_tokens ?? 0;
      }
      if (event.type === "message_delta" && event.usage) {
        outputTokens = event.usage.output_tokens ?? 0;
      }

      if (event.type === "message_stop") {
        yield {
          type: "message_stop",
          usage: {
            input_tokens: inputTokens,
            output_tokens: outputTokens,
          },
        } as BedrockAnthropicParsedChunk;
        return;
      }

      yield event as unknown as BedrockAnthropicParsedChunk;
    }
  }
}
