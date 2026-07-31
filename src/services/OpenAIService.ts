import OpenAI from "openai";
import { OpenAIMessages, OpenAIResponse } from "../types";
import { ClientService } from "./ClientService";

// Some models reject tools + non-"none" reasoning_effort; retried with "none".
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isUnsupportedReasoningEffortError(error: any): boolean {
  return error?.status === 400 && error?.param === "reasoning_effort";
}

// Some models reject a custom temperature; retried with it omitted.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isUnsupportedTemperatureError(error: any): boolean {
  return (
    error?.status === 400 &&
    error?.param === "temperature" &&
    error?.code === "unsupported_value"
  );
}

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
    verbosity,
    toolChoice
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens?: number;
    temperature: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    reasoning_effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh";
    verbosity?: "low" | "medium" | "high";
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toolChoice?: any;
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

      // Only include token limit when provided
      if (typeof max_tokens === "number") {
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
      }

      // Add tools if provided (modern API, replaces deprecated functions)
      if (tools) {
        requestBody.tools = tools;
      }

      // Force tool usage when requested (e.g. "required")
      if (tools && toolChoice) {
        requestBody.tool_choice = toolChoice;
      }

      // Add optional reasoning parameter for reasoning models
      if (reasoning_effort) {
        requestBody.reasoning_effort = reasoning_effort;
      }

      // Add optional verbosity parameter
      if (verbosity) {
        requestBody.verbosity = verbosity;
      }

      try {
        const response = await this.openai.chat.completions.create(
          requestBody
        );
        return response as OpenAIResponse;
      } catch (error) {
        if (!reasoning_effort && isUnsupportedReasoningEffortError(error)) {
          const response = await this.openai.chat.completions.create({
            ...requestBody,
            reasoning_effort: "none"
          });
          return response as OpenAIResponse;
        }
        if (isUnsupportedTemperatureError(error)) {
          const { temperature: _temperature, ...retryBody } = requestBody;
          const response = await this.openai.chat.completions.create(
            retryBody
          );
          return response as OpenAIResponse;
        }
        throw error;
      }
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
    verbosity,
    toolChoice
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens?: number;
    temperature: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    reasoning_effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh";
    verbosity?: "low" | "medium" | "high";
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toolChoice?: any;
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

      // Only include token limit when provided
      if (typeof max_tokens === "number") {
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
      }

      // Add tools if provided (modern API, replaces deprecated functions)
      if (tools) {
        requestBody.tools = tools;
      }

      // Force tool usage when requested (e.g. "required")
      if (tools && toolChoice) {
        requestBody.tool_choice = toolChoice;
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
      let stream: any;
      try {
        stream = await (this.openai.chat.completions.create(
          requestBody
        ) as any);
      } catch (error) {
        if (!reasoning_effort && isUnsupportedReasoningEffortError(error)) {
          stream = await (this.openai.chat.completions.create({
            ...requestBody,
            reasoning_effort: "none"
          }) as any);
        } else if (isUnsupportedTemperatureError(error)) {
          const { temperature: _temperature, ...retryBody } = requestBody;
          stream = await (this.openai.chat.completions.create(
            retryBody
          ) as any);
        } else {
          throw error;
        }
      }

      for await (const chunk of stream) {
        yield chunk;
      }
    } catch (error) {
      return Promise.reject(error);
    }
  }
}
