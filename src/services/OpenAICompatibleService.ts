import OpenAI from "openai";
import { OpenAIMessages, OpenAIResponse } from "../types";
import { ClientService } from "./ClientService";

function normalizeTools(tools?: any[]): any[] | undefined {
  if (!Array.isArray(tools)) return undefined;

  return tools.map((tool) => {
    // Already correct
    if (tool?.type === "function" && tool.function) {
      return tool;
    }

    // Needs reformatting
    if (tool?.type === "function" && tool.name && tool.parameters) {
      return {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      };
    }

    // Unknown / unsupported tool shape
    throw new Error(
      `Invalid tool definition encountered: ${JSON.stringify(tool, null, 2)}`
    );
  });
}

export default class OpenAICompatibleService implements ClientService {
  private openai: OpenAI;

  constructor(apiKey: string, baseUrl?: string) {
    this.openai = new OpenAI({
      apiKey,
      baseURL: baseUrl,
    });
  }

  async generateCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    tools?: any;
  }): Promise<OpenAIResponse> {
    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for OpenAIService.")
      );
    }

    try {
      const response = await this.openai.chat.completions.create({
        model,
        messages,
        max_tokens,
        temperature,
        functions: tools,
      });
      return response as OpenAIResponse;
    } catch (error) {
      return Promise.reject(error);
    }
  }

  async *generateStreamCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    tools?: any;
  }): AsyncGenerator<any, void, unknown> {
    if (!model) {
      throw new Error("Model ID is required for OpenAIService.");
    }

    const normalizedTools = normalizeTools(tools);

    const stream = await this.openai.chat.completions.create({
      model,
      messages,
      max_tokens,
      temperature,
      ...(normalizedTools && { tools: normalizedTools }),
      stream: true,
      stream_options: {
        include_usage: true,
      },
    });

    for await (const chunk of stream) {
      yield chunk;
    }
  }
}
