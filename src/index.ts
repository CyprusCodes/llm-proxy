import { Messages, OpenAIResponse, Providers } from "./types";
import OpenAIService from "./services/OpenAIService";
import AnthropicService from "./services/AnthropicService";
import AwsBedrockAnthropicService from "./services/AwsBedrockAnthropicService";
import ProviderFinder from "./middleware/ProviderFinder";
import InputFormatAdapter from "./middleware/InputFormatAdapter";
import OutputFormatAdapter from "./middleware/OutputFormatAdapter";
import AwsBedrockLlama3Service from "./services/AwsBedrockLlama3Service";
import OpenAICompatibleService from "./services/OpenAICompatibleService";

export { llmAsJudge } from "./utils/llmAsJudge";

interface Credentials {
  apiKey?: string;
  awsConfig?: { accessKeyId: string; secretAccessKey: string; region: string };
  openAICompatibleProviderConfig?: {
    openAICompatibleProviderKey: string;
    baseUrl: string;
  };
}

interface GenerateLLMResponseParams {
  messages: Messages;
  model: string;
  functions?: any; // TODO: align naming — this is "tools" everywhere else
  max_tokens?: number;
  temperature?: number;
  credentials: Credentials;
}

type ServiceInstance =
  | OpenAIService
  | AnthropicService
  | AwsBedrockAnthropicService
  | AwsBedrockLlama3Service
  | OpenAICompatibleService;

function createService(
  provider: Providers,
  credentials: Credentials
): ServiceInstance {
  switch (provider) {
    case Providers.OPENAI: {
      if (!credentials.apiKey) {
        throw new Error("OpenAI API key is required for OpenAI models.");
      }
      return new OpenAIService(credentials.apiKey);
    }
    case Providers.ANTHROPIC: {
      if (!credentials.apiKey) {
        throw new Error("Anthropic API key is required for Anthropic models.");
      }
      return new AnthropicService(credentials.apiKey);
    }
    case Providers.ANTHROPIC_BEDROCK: {
      const { awsConfig } = credentials;
      if (!awsConfig) {
        throw new Error("AWS credentials are required for Bedrock models.");
      }
      return new AwsBedrockAnthropicService(
        awsConfig.accessKeyId,
        awsConfig.secretAccessKey,
        awsConfig.region
      );
    }
    case Providers.LLAMA_3_1_BEDROCK: {
      const { awsConfig } = credentials;
      if (!awsConfig) {
        throw new Error("AWS credentials are required for Bedrock models.");
      }
      return new AwsBedrockLlama3Service(
        awsConfig.accessKeyId,
        awsConfig.secretAccessKey,
        awsConfig.region
      );
    }
    case Providers.OPENAI_COMPATIBLE_PROVIDER: {
      const config = credentials.openAICompatibleProviderConfig;
      if (!config?.openAICompatibleProviderKey || !config?.baseUrl) {
        throw new Error(
          "OpenAI Compatible Provider key and base URL are required."
        );
      }
      return new OpenAICompatibleService(
        config.openAICompatibleProviderKey,
        config.baseUrl
      );
    }
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

function reconstructAnthropicResponse(chunks: any[]): any {
  let id = "";
  let model = "";
  const contentBlocks: any[] = [];
  let stopReason = "";
  let inputTokens = 0;
  let outputTokens = 0;

  let currentBlock: any = null;
  let currentToolArgs: string[] = [];

  for (const chunk of chunks) {
    if (chunk.type === "message_start" && chunk.message) {
      id = chunk.message.id || id;
      model = chunk.message.model || model;
      inputTokens = chunk.message.usage?.input_tokens ?? inputTokens;
      outputTokens = chunk.message.usage?.output_tokens ?? outputTokens;
    }

    if (chunk.type === "content_block_start" && chunk.content_block) {
      currentBlock = { ...chunk.content_block };
      currentToolArgs = [];
    }

    if (chunk.type === "content_block_delta") {
      if (chunk.delta?.type === "text_delta" && currentBlock?.type === "text") {
        currentBlock.text =
          (currentBlock.text || "") + (chunk.delta.text || "");
      }
      if (
        chunk.delta?.type === "input_json_delta" &&
        chunk.delta?.partial_json
      ) {
        currentToolArgs.push(chunk.delta.partial_json);
      }
    }

    if (chunk.type === "content_block_stop" && currentBlock) {
      if (currentBlock.type === "tool_use") {
        try {
          currentBlock.input = JSON.parse(currentToolArgs.join(""));
        } catch {
          currentBlock.input = {};
        }
      }
      contentBlocks.push(currentBlock);
      currentBlock = null;
    }

    if (chunk.type === "message_delta") {
      stopReason = chunk.delta?.stop_reason || stopReason;
      outputTokens = chunk.usage?.output_tokens ?? outputTokens;
    }

    if (chunk.type === "message_stop" && chunk.usage) {
      inputTokens = chunk.usage.input_tokens ?? inputTokens;
      outputTokens = chunk.usage.output_tokens ?? outputTokens;
    }

    // Handle AWS Bedrock metrics format
    const metrics = chunk["amazon-bedrock-invocationMetrics"];
    if (metrics) {
      inputTokens = metrics.inputTokenCount ?? inputTokens;
      outputTokens = metrics.outputTokenCount ?? outputTokens;
    }
  }

  return {
    id,
    type: "message",
    role: "assistant",
    model,
    content: contentBlocks,
    stop_reason: stopReason,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens }
  };
}

function anthropicToolCallToStreamChunk(
  completeResponse: any,
  toolUseBlock: any
): OpenAIResponse {
  const argumentsString =
    typeof toolUseBlock.input === "string"
      ? toolUseBlock.input
      : JSON.stringify(toolUseBlock.input);

  return ({
    id: completeResponse.id || `anthropic-${Date.now()}`,
    object: "chat.completion.chunk",
    created: Date.now(),
    model: completeResponse.model,
    system_fingerprint: "default_fingerprint",
    choices: [
      {
        index: 0,
        delta: {
          function_call: {
            name: toolUseBlock.name,
            arguments: argumentsString
          }
        },
        logprobs: null,
        finish_reason: "stop"
      }
    ],
    usage: {
      prompt_tokens: completeResponse.usage?.input_tokens || 0,
      completion_tokens: completeResponse.usage?.output_tokens || 0,
      total_tokens:
        (completeResponse.usage?.input_tokens || 0) +
        (completeResponse.usage?.output_tokens || 0),
      prompt_tokens_details: { cached_tokens: 0 },
      completion_tokens_details: { reasoning_tokens: 0 }
    }
  } as unknown) as OpenAIResponse;
}

export async function generateLLMResponse(
  params: GenerateLLMResponseParams
): Promise<OpenAIResponse> {
  const {
    messages,
    model,
    functions,
    max_tokens,
    temperature,
    credentials
  } = params;

  const { openAICompatibleProviderConfig } = credentials;
  const { baseUrl } = openAICompatibleProviderConfig || {};

  const provider = ProviderFinder.getProvider(model, baseUrl);
  const service = createService(provider, credentials);

  const { adaptedMessages, systemPrompt } = InputFormatAdapter.adaptMessages(
    messages,
    provider,
    model
  );

  const response = await service.generateCompletion({
    messages: adaptedMessages as any,
    model,
    ...(typeof max_tokens === "number" ? { max_tokens } : {}),
    temperature: temperature || 0,
    tools: functions,
    systemPrompt: systemPrompt || ""
  });

  // OpenAI responses are already in the right format
  const isOpenAIFormat =
    provider === Providers.OPENAI ||
    provider === Providers.OPENAI_COMPATIBLE_PROVIDER;

  if (isOpenAIFormat) {
    return response as OpenAIResponse;
  }

  const adaptedResponse = await OutputFormatAdapter.adaptResponse({
    response,
    provider,
    isStream: false
  });
  return adaptedResponse as OpenAIResponse;
}

/** Streaming LLM completion. Returns an async generator of OpenAI-format chunks. */
export async function generateLLMStreamResponse(
  params: GenerateLLMResponseParams
): Promise<AsyncGenerator<OpenAIResponse>> {
  const {
    messages,
    model,
    functions,
    max_tokens,
    temperature,
    credentials
  } = params;

  const { openAICompatibleProviderConfig } = credentials;
  const { baseUrl } = openAICompatibleProviderConfig || {};

  const provider = ProviderFinder.getProvider(model, baseUrl);
  const service = createService(provider, credentials);

  const { adaptedMessages, systemPrompt } = InputFormatAdapter.adaptMessages(
    messages,
    provider,
    model
  );

  const stream = service.generateStreamCompletion({
    messages: adaptedMessages as any,
    model,
    ...(typeof max_tokens === "number" ? { max_tokens } : {}),
    temperature: temperature || 0,
    tools: functions,
    systemPrompt: systemPrompt || ""
  });

  async function* streamGenerator(): AsyncGenerator<OpenAIResponse> {
    // OpenAI / OpenAI-compatible — pass through as-is
    const isOpenAIFormat =
      provider === Providers.OPENAI ||
      provider === Providers.OPENAI_COMPATIBLE_PROVIDER;

    if (isOpenAIFormat) {
      for await (const chunk of stream) {
        yield chunk as OpenAIResponse;
      }
      return;
    }

    if (
      provider === Providers.ANTHROPIC ||
      provider === Providers.ANTHROPIC_BEDROCK
    ) {
      const allChunks: any[] = [];
      let hasToolUse = false;

      for await (const chunk of stream) {
        allChunks.push(chunk);
        if (
          chunk.type === "content_block_start" &&
          chunk.content_block?.type === "tool_use"
        ) {
          hasToolUse = true;
        }
      }

      if (hasToolUse) {
        const completeResponse = reconstructAnthropicResponse(allChunks);
        const toolUseBlock = completeResponse.content.find(
          (block: any) => block.type === "tool_use"
        );
        if (toolUseBlock) {
          yield anthropicToolCallToStreamChunk(completeResponse, toolUseBlock);
        }
      } else {
        // Text-only: yield each chunk through the streaming adapter
        for (const chunk of allChunks) {
          yield (await OutputFormatAdapter.adaptResponse({
            response: chunk,
            provider,
            isStream: true
          })) as OpenAIResponse;
        }
      }
      return;
    }

    // Llama 3.1 Bedrock — buffer first 3 chunks to detect function calls
    // via the `<function>` marker in the generation text.
    const buffer: any[] = [];
    let isFunctionCall = false;
    const accumulatedChunks: any[] = [];

    for await (const chunk of stream) {
      if (!isFunctionCall) {
        buffer.push(chunk);

        if (buffer.length === 3) {
          const [first, second, third] = buffer;

          if (second.generation === "<" && third.generation === "function") {
            isFunctionCall = true;
          }

          if (isFunctionCall) {
            accumulatedChunks.push(...buffer);
            buffer.length = 0;
          } else {
            yield (await OutputFormatAdapter.adaptResponse({
              response: first,
              provider,
              isStream: true,
              isFunctionCall: false
            })) as OpenAIResponse;
            buffer.shift();
          }
        }
      } else {
        accumulatedChunks.push(chunk);
      }
    }

    if (isFunctionCall) {
      const fullResponse = accumulatedChunks.reduce((acc, cur) => {
        acc.generation += cur.generation;
        return acc;
      });
      yield (await OutputFormatAdapter.adaptResponse({
        response: fullResponse,
        provider,
        isStream: false,
        isFunctionCall: true
      })) as OpenAIResponse;
    } else {
      while (buffer.length > 0) {
        const chunk = buffer.shift();
        yield (await OutputFormatAdapter.adaptResponse({
          response: chunk,
          provider,
          isStream: true
        })) as OpenAIResponse;
      }
    }
  }

  return streamGenerator();
}

export * from "./types";
