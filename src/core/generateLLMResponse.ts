import { Messages, OpenAIResponse, Providers } from "llm-proxy";
import InputFormatAdapter from "../middleware/InputFormatAdapter";
import OutputFormatAdapter from "../middleware/OutputFormatAdapter";
import ProviderFinder from "../middleware/ProviderFinder";
import { GenerateLLMResponseParams } from "../types";
import initializeProviderService from "../utils/initializeProviderService";

async function generateLLMResponse(
  params: GenerateLLMResponseParams
): Promise<OpenAIResponse> {
  const { messages, model, functions, max_tokens, temperature, credentials } =
    params;

  const provider = ProviderFinder.getProvider(model);
  const service = initializeProviderService(provider, credentials);

  const { adaptedMessages, systemPrompt } = InputFormatAdapter.adaptMessages({
    messages: messages as Messages,
    provider,
  });

  const response = await service.generateCompletion({
    messages: adaptedMessages as Messages,
    model,
    max_tokens,
    temperature: temperature || 0,
    tools: functions,
    systemPrompt: systemPrompt || "",
  });

  return provider === Providers.OPENAI
    ? (response as OpenAIResponse)
    : (OutputFormatAdapter.adaptResponse(response, provider) as OpenAIResponse);
}

export default generateLLMResponse;
