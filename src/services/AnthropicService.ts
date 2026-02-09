import Anthropic from "@anthropic-ai/sdk";
import {
  BedrockAnthropicParsedChunk,
  BedrockAnthropicResponse,
  Messages
} from "../types";
import { ClientService } from "./ClientService";

const DEFAULT_INPUT_SCHEMA = {
  type: "object" as const,
  properties: {},
  required: [] as string[]
};

/**
 * Converts OpenAI tool format to Anthropic tool format.
 *
 * OpenAI:    { type: "function", function: { name, description, parameters } }
 * Anthropic: { type: "custom",   name, description, input_schema }
 *
 * Also handles tools that are already in Anthropic format or a mixed shape.
 */
function convertToolsToAnthropicFormat(
  tools: any[] | undefined | null
): Anthropic.Tool[] {
  if (!tools || !Array.isArray(tools) || tools.length === 0) {
    return [];
  }

  return tools.map((tool: any) => {
    // Already in Anthropic format
    if (
      tool.type === "custom" ||
      (tool.name && tool.description && !tool.function)
    ) {
      return {
        type: "custom",
        name: tool.name,
        description: tool.description || "",
        input_schema:
          tool.input_schema || tool.parameters || DEFAULT_INPUT_SCHEMA
      } as Anthropic.Tool;
    }

    // Standard OpenAI format → convert
    if (tool.type === "function" && tool.function) {
      const { name, description, parameters } = tool.function;
      return {
        type: "custom",
        name,
        description: description || "",
        input_schema: parameters || DEFAULT_INPUT_SCHEMA
      } as Anthropic.Tool;
    }

    // Unknown shape — best-effort conversion
    return {
      ...tool,
      type: "custom",
      input_schema: tool.input_schema || tool.parameters || DEFAULT_INPUT_SCHEMA
    } as Anthropic.Tool;
  });
}

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
    tools?: any;
    systemPrompt?: string;
  }): Promise<BedrockAnthropicResponse> {
    const {
      messages,
      model,
      max_tokens,
      temperature,
      systemPrompt,
      tools
    } = params;

    if (!model) {
      throw new Error("Model ID is required for AnthropicService");
    }

    const validatedTools = convertToolsToAnthropicFormat(tools);
    const hasTools = validatedTools.length > 0;

    const createParams: Anthropic.MessageCreateParamsNonStreaming = {
      model,
      max_tokens: max_tokens ?? 1024,
      temperature: temperature ?? 0,
      system: systemPrompt ?? "",
      messages: messages as Anthropic.MessageParam[],
      ...(hasTools ? { tools: validatedTools } : {})
    };

    const response = await this.client.messages.create(createParams);
    return (response as unknown) as BedrockAnthropicResponse;
  }

  async *generateStreamCompletion(params: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    tools?: any;
    systemPrompt?: string;
  }): AsyncGenerator<BedrockAnthropicParsedChunk, void, unknown> {
    const {
      messages,
      model,
      max_tokens,
      temperature,
      tools,
      systemPrompt
    } = params;

    if (!model) {
      throw new Error("Model ID is required for AnthropicService");
    }

    const validatedTools = convertToolsToAnthropicFormat(tools);
    const hasTools = validatedTools.length > 0;

    const createParams: Anthropic.MessageCreateParamsStreaming = {
      model,
      max_tokens: max_tokens ?? 1024,
      temperature: temperature ?? 0,
      system: systemPrompt ?? "",
      messages: messages as Anthropic.MessageParam[],
      stream: true,
      ...(hasTools ? { tools: validatedTools } : {})
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
            output_tokens: outputTokens
          }
        } as BedrockAnthropicParsedChunk;
        return;
      }

      yield (event as unknown) as BedrockAnthropicParsedChunk;
    }
  }
}
