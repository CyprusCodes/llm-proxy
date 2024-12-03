import { Messages, OpenAIResponse, Providers } from "llm-proxy";
import InputFormatAdapter from "../middleware/InputFormatAdapter";
import OutputFormatAdapter from "../middleware/OutputFormatAdapter";
import ProviderFinder from "../middleware/ProviderFinder";
import { GenerateLLMResponseParams } from "../types";
import initializeProviderService from "../utils/initializeProviderService";

async function generateLLMStreamResponse(
  params: GenerateLLMResponseParams
): Promise<AsyncGenerator<OpenAIResponse>> {
  const { messages, model, functions, max_tokens, temperature, credentials } =
    params;

  const provider = ProviderFinder.getProvider(model);
  const service = initializeProviderService(provider, credentials);

  const { adaptedMessages, systemPrompt } = InputFormatAdapter.adaptMessages({
    messages: messages as Messages,
    provider,
  });

  const stream = service.generateStreamCompletion({
    messages: adaptedMessages as Messages,
    model,
    max_tokens,
    temperature: temperature || 0,
    tools: functions,
    systemPrompt: systemPrompt || "",
  });

  async function* streamGenerator(): AsyncGenerator<OpenAIResponse> {
    for await (const chunk of stream) {
      yield provider === Providers.OPENAI
        ? (chunk as OpenAIResponse)
        : (OutputFormatAdapter.adaptResponse(
            chunk,
            provider
          ) as OpenAIResponse);
    }
  }

  return streamGenerator();
}

export default generateLLMStreamResponse;
