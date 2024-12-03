import OpenAI from "openai";
import { Messages, OpenAIMessages, OpenAIResponse } from "../types";
import { ClientService } from "./ClientService";
import LLM_PROXY_ERROR_MESSAGES from "../constants/errorMessages";

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
  }: {
    messages: Messages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
  }): Promise<OpenAIResponse> {
    if (!model) {
      return Promise.reject(new Error(LLM_PROXY_ERROR_MESSAGES.MISSING_MODEL));
    }

    try {
      const response = await this.openai.chat.completions.create({
        model,
        messages: messages as OpenAIMessages,
        max_tokens,
        temperature,
        functions: tools,
      });
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
  }: {
    messages: Messages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    // TODO: fix the below any type
    tools?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    // TODO: fix the below any type
  }): AsyncGenerator<any, void, unknown> {
    if (!model) {
      return Promise.reject(new Error(LLM_PROXY_ERROR_MESSAGES.MISSING_MODEL));
    }

    try {
      const stream = await this.openai.chat.completions.create({
        model,
        messages: messages as OpenAIMessages,
        max_tokens,
        temperature,
        functions: tools,
        stream: true,
        stream_options: {
          include_usage: true,
        },
      });

      for await (const chunk of stream) {
        yield chunk;
      }
    } catch (error) {
      return Promise.reject(error);
    }
  }
}
