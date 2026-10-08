import { OpenAIMessages, OpenAIResponse } from "../types";
import OpenAICompatibleService from "./OpenAICompatibleService";

// Gemini 3.x+ deprecated temperature, top_p, and top_k — upcoming models return
// 400 INVALID_ARGUMENT if these are present. Older generations are unaffected and
// continue to use OpenAICompatibleService directly.
export default class GeminiService extends OpenAICompatibleService {
  async generateCompletion({
    messages,
    model,
    max_tokens,
    temperature: _temperature,
    tools,
    toolChoice,
    systemPrompt,
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens?: number;
    temperature?: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toolChoice?: any;
  }): Promise<OpenAIResponse> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const requestBody: any = { messages, model };
    if (typeof max_tokens === "number") requestBody.max_tokens = max_tokens;
    if (tools) requestBody.tools = tools;
    if (tools && toolChoice) requestBody.toolChoice = toolChoice;
    if (systemPrompt) requestBody.systemPrompt = systemPrompt;
    return super.generateCompletion(requestBody);
  }

  // eslint-disable-next-line consistent-return
  async *generateStreamCompletion({
    messages,
    model,
    max_tokens,
    temperature: _temperature,
    tools,
    toolChoice,
    systemPrompt,
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens?: number;
    temperature?: number;
    systemPrompt?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toolChoice?: any;
  }): AsyncGenerator<any, void, unknown> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const requestBody: any = { messages, model };
    if (typeof max_tokens === "number") requestBody.max_tokens = max_tokens;
    if (tools) requestBody.tools = tools;
    if (tools && toolChoice) requestBody.toolChoice = toolChoice;
    if (systemPrompt) requestBody.systemPrompt = systemPrompt;
    yield* super.generateStreamCompletion(requestBody);
  }
}
