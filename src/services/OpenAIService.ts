import OpenAI from "openai";
import { OpenAIMessages, OpenAIResponse } from "../types";
import { ClientService } from "./ClientService";

export default class OpenAIService implements ClientService {
  private openai: OpenAI;

  constructor(apiKey: string) {
    this.openai = new OpenAI({ apiKey });
  }

  async generateCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
    reasoning_effort,
    verbosity
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    reasoning_effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh";
    verbosity?: "low" | "medium" | "high";
  }): Promise<OpenAIResponse> {
    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for OpenAIService.")
      );
    }

    try {
      // Build the request object
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const requestBody: any = {
        model,
        messages,
        temperature
      };

      // Use max_completion_tokens for newer models (GPT-5+), fallback to max_tokens for older models
      if (
        model.startsWith("gpt-5") ||
        model.startsWith("o3") ||
        model.startsWith("o4")
      ) {
        requestBody.max_completion_tokens = max_tokens;
      } else {
        requestBody.max_tokens = max_tokens;
      }

      // Add tools if provided (modern API, replaces deprecated functions)
      if (tools) {
        requestBody.tools = tools;
      }

      // Add optional reasoning parameter for reasoning models
      if (reasoning_effort) {
        requestBody.reasoning_effort = reasoning_effort;
      }

      // Add optional verbosity parameter
      if (verbosity) {
        requestBody.verbosity = verbosity;
      }

      const response = await this.openai.chat.completions.create(requestBody);
      return response as OpenAIResponse;
    } catch (error) {
      return Promise.reject(error);
    }
  }

  // eslint-disable-next-line consistent-return
  async *generateStreamCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
    reasoning_effort,
    verbosity
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    reasoning_effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh";
    verbosity?: "low" | "medium" | "high";
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }): AsyncGenerator<any, void, unknown> {
    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for OpenAIService.")
      );
    }

    try {
      // Build the request object
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const requestBody: any = {
        model,
        messages,
        temperature,
        stream: true,
        stream_options: {
          include_usage: true
        }
      };

      // Use max_completion_tokens for newer models (GPT-5+), fallback to max_tokens for older models
      if (
        model.startsWith("gpt-5") ||
        model.startsWith("o3") ||
        model.startsWith("o4")
      ) {
        requestBody.max_completion_tokens = max_tokens;
      } else {
        requestBody.max_tokens = max_tokens;
      }

      // Add tools if provided (modern API, replaces deprecated functions)
      if (tools) {
        requestBody.tools = tools;
      }

      // Add optional reasoning parameter for reasoning models
      if (reasoning_effort) {
        requestBody.reasoning_effort = reasoning_effort;
      }

      // Add optional verbosity parameter
      if (verbosity) {
        requestBody.verbosity = verbosity;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const stream = await (this.openai.chat.completions.create(
        requestBody
      ) as any);

      for await (const chunk of stream) {
        yield chunk;
      }
    } catch (error) {
      return Promise.reject(error);
    }
  }
}
