import OpenAI from "openai";
import { OpenAIMessages, OpenAIResponse } from "../types";
import { ClientService } from "./ClientService";
import {
  avoidKnownUnsupportedParams,
  isUnsupportedReasoningEffortError,
  isUnsupportedTemperatureError
} from "../utils/servicesUtils/openAIUnsupportedParams";
import {
  chatRequestToResponsesRequest,
  responsesStreamToChatChunks,
  responsesToChatCompletion,
  usesResponsesApi
} from "../utils/servicesUtils/openAIResponsesAdapter";

export default class OpenAIService implements ClientService {
  private openai: OpenAI;

  constructor(apiKey: string) {
    this.openai = new OpenAI({ apiKey });
  }

  /**
   * Sends a Chat Completions-shaped request. GPT-5.x models are routed to the
   * Responses API and their output is translated back to Chat Completions
   * format, so callers always receive chat.completion(.chunk) objects.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async send(requestBody: any): Promise<any> {
    if (!usesResponsesApi(requestBody.model)) {
      return this.openai.chat.completions.create(requestBody);
    }
    const responsesBody = chatRequestToResponsesRequest(requestBody);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result: any = await this.openai.responses.create(responsesBody);
    return requestBody.stream
      ? responsesStreamToChatChunks(result)
      : responsesToChatCompletion(result);
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

      avoidKnownUnsupportedParams(requestBody, model, tools, reasoning_effort);

      try {
        const response = await this.send(requestBody);
        return response as OpenAIResponse;
      } catch (error) {
        if (!reasoning_effort && isUnsupportedReasoningEffortError(error)) {
          const response = await this.send({
            ...requestBody,
            reasoning_effort: "none"
          });
          return response as OpenAIResponse;
        }
        if (isUnsupportedTemperatureError(error)) {
          const { temperature: _temperature, ...retryBody } = requestBody;
          const response = await this.send(retryBody);
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

      avoidKnownUnsupportedParams(requestBody, model, tools, reasoning_effort);

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let stream: any;
      try {
        stream = await this.send(requestBody);
      } catch (error) {
        if (!reasoning_effort && isUnsupportedReasoningEffortError(error)) {
          stream = await this.send({
            ...requestBody,
            reasoning_effort: "none"
          });
        } else if (isUnsupportedTemperatureError(error)) {
          const { temperature: _temperature, ...retryBody } = requestBody;
          stream = await this.send(retryBody);
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
