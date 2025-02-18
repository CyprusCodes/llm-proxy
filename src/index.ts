import { Messages, OpenAIResponse, Providers } from "./types";
import OpenAIService from "./services/OpenAIService";
import AwsBedrockAnthropicService from "./services/AwsBedrockAnthropicService";
import ProviderFinder from "./middleware/ProviderFinder";
import InputFormatAdapter from "./middleware/InputFormatAdapter";
import OutputFormatAdapter from "./middleware/OutputFormatAdapter";
import AwsBedrockLlama3Service from "./services/AwsBedrockLlama3Service";
import OpenAICompatibleService from "./services/OpenAICompatibleService";

// Define the credentials interface for flexibility
interface Credentials {
  apiKey?: string;
  awsConfig?: { accessKeyId: string; secretAccessKey: string; region: string };
  openAICompatibleProviderConfig?: {
    openAICompatibleProviderKey: string;
    baseUrl: string;
  };
}

// Define the input parameters interface for flexibility
interface GenerateLLMResponseParams {
  messages: Messages;
  model: string;
  functions?: any; // TODO : Fix this any more info in the ClientService.ts
  max_tokens: number;
  temperature?: number;
  credentials: Credentials;
}

// Main function for non-streaming requests
export async function generateLLMResponse(
  params: GenerateLLMResponseParams
): Promise<OpenAIResponse> {
  const { messages, model, functions, max_tokens, temperature, credentials } =
    params;

  const { openAICompatibleProviderConfig } = credentials;
  const { openAICompatibleProviderKey, baseUrl } =
    openAICompatibleProviderConfig || {};

  // Step 1: Identify the provider based on the model
  const provider = ProviderFinder.getProvider(model, baseUrl);

  // Initialize the correct service based on the provider
  let service:
    | OpenAIService
    | AwsBedrockAnthropicService
    | AwsBedrockLlama3Service;
  if (provider === Providers.OPENAI) {
    if (!credentials.apiKey) {
      return Promise.reject(
        new Error("OpenAI API key is required for OpenAI models.")
      );
    }
    service = new OpenAIService(credentials.apiKey);
  } else if (provider === Providers.ANTHROPIC_BEDROCK) {
    const { awsConfig } = credentials;
    if (!awsConfig) {
      return Promise.reject(
        new Error("AWS credentials are required for Bedrock models.")
      );
    }
    service = new AwsBedrockAnthropicService(
      awsConfig.accessKeyId,
      awsConfig.secretAccessKey,
      awsConfig.region
    );
  } else if (provider === Providers.LLAMA_3_1_BEDROCK) {
    const { awsConfig } = credentials;
    if (!awsConfig) {
      return Promise.reject(
        new Error("AWS credentials are required for Bedrock models.")
      );
    }
    service = new AwsBedrockLlama3Service(
      awsConfig.accessKeyId,
      awsConfig.secretAccessKey,
      awsConfig.region
    );
  } else if (provider === Providers.OPENAI_COMPATIBLE_PROVIDER) {
    if (!openAICompatibleProviderKey || !baseUrl) {
      return Promise.reject(
        new Error(
          "OpenAI Compatible Provider key and base URL are required for OpenAI Compatible models."
        )
      );
    }
    service = new OpenAICompatibleService(openAICompatibleProviderKey, baseUrl);
  } else {
    return Promise.reject(new Error("Unsupported provider 4"));
  }

  // Step 2: Adapt messages and extract the system prompt
  const { adaptedMessages, systemPrompt } = InputFormatAdapter.adaptMessages(
    messages,
    provider
  );

  // Step 3: Generate the completion
  const response = await service.generateCompletion({
    messages: adaptedMessages as any, // TODO: fix this any
    model,
    max_tokens,
    temperature: temperature || 0,
    tools: functions,
    systemPrompt: systemPrompt || "",
  });

  // Step 4: Adapt the response if needed
  const adaptedResponse =
    provider === Providers.OPENAI ||
    provider === Providers.OPENAI_COMPATIBLE_PROVIDER
      ? response
      : OutputFormatAdapter.adaptResponse({
          response,
          provider,
          isStream: false,
        });
  return adaptedResponse as OpenAIResponse;
}

// Main function for streaming requests
// Main function for streaming requests
// Main function for streaming requests
export async function generateLLMStreamResponse(
  params: GenerateLLMResponseParams
): Promise<AsyncGenerator<OpenAIResponse>> {
  const { messages, model, functions, max_tokens, temperature, credentials } =
    params;

  const { openAICompatibleProviderConfig } = credentials;
  const { openAICompatibleProviderKey, baseUrl } =
    openAICompatibleProviderConfig || {};

  // Step 1: Identify the provider based on the model
  const provider = ProviderFinder.getProvider(model, baseUrl);

  // Initialize the correct service based on the provider
  let service:
    | OpenAIService
    | AwsBedrockAnthropicService
    | AwsBedrockLlama3Service
    | OpenAICompatibleService;
  if (provider === Providers.OPENAI) {
    if (!credentials.apiKey) {
      return Promise.reject(
        new Error("OpenAI API key is required for OpenAI models.")
      );
    }
    service = new OpenAIService(credentials.apiKey);
  } else if (provider === Providers.ANTHROPIC_BEDROCK) {
    const { awsConfig } = credentials;
    if (!awsConfig) {
      return Promise.reject(
        new Error("AWS credentials are required for Bedrock models.")
      );
    }
    service = new AwsBedrockAnthropicService(
      awsConfig.accessKeyId,
      awsConfig.secretAccessKey,
      awsConfig.region
    );
  } else if (provider === Providers.LLAMA_3_1_BEDROCK) {
    const { awsConfig } = credentials;
    if (!awsConfig) {
      return Promise.reject(
        new Error("AWS credentials are required for Bedrock models.")
      );
    }
    service = new AwsBedrockLlama3Service(
      awsConfig.accessKeyId,
      awsConfig.secretAccessKey,
      awsConfig.region
    );
  } else if (provider === Providers.OPENAI_COMPATIBLE_PROVIDER) {
    if (!openAICompatibleProviderKey || !baseUrl) {
      return Promise.reject(
        new Error(
          "OpenAI Compatible Provider key and base URL are required for OpenAI Compatible models."
        )
      );
    }
    service = new OpenAICompatibleService(openAICompatibleProviderKey, baseUrl);
  } else {
    return Promise.reject(new Error("Unsupported provider"));
  }

  // Step 2: Adapt messages and extract the system prompt
  const { adaptedMessages, systemPrompt } = InputFormatAdapter.adaptMessages(
    messages,
    provider
  );

  // Step 3: Generate the streaming completion
  const stream = service.generateStreamCompletion({
    messages: adaptedMessages as any, // TODO: Fix this any
    model,
    max_tokens,
    temperature: temperature || 0,
    tools: functions,
    systemPrompt: systemPrompt || "",
  });

  // Step 4: Create and return the async generator
  async function* streamGenerator(): AsyncGenerator<OpenAIResponse> {
    const buffer: any[] = []; // Buffer to hold the first three chunks
    let isFunctionCall = false;
    const accumulatedChunks: any[] = []; // Accumulate chunks for function calls

    for await (const chunk of stream) {
      if (!isFunctionCall) {
        // Push the chunk to the buffer
        buffer.push(chunk);

        // Check condition if we have the first three chunks
        if (buffer.length === 3) {
          const [first, second, third] = buffer;

          // Evaluate the condition
          if (second.generation === "<" && third.generation === "function") {
            isFunctionCall = true;
          }

          // Clear the buffer if condition met, else continue streaming
          if (isFunctionCall) {
            accumulatedChunks.push(...buffer);
            buffer.length = 0;
          } else {
            // Yield the first chunk
            yield provider === Providers.OPENAI
              ? first
              : ((await OutputFormatAdapter.adaptResponse({
                  response: first,
                  provider,
                  isStream: true,
                  isFunctionCall: false,
                })) as OpenAIResponse);

            buffer.shift(); // Remove the first chunk from the buffer
          }
        }
      } else {
        // Accumulate chunks for function call
        accumulatedChunks.push(chunk);
      }
    }

    if (isFunctionCall) {
      // Pass the entire accumulated response to adaptResponse
      const fullResponse = accumulatedChunks.reduce((acc, cur) => {
        acc.generation += cur.generation;
        return acc;
      });

      const response =
        provider === Providers.OPENAI
          ? { ...fullResponse, isFunctionCall: true }
          : ((await OutputFormatAdapter.adaptResponse({
              response: fullResponse,
              provider,
              isStream: false,
              isFunctionCall: true,
            })) as OpenAIResponse);

      yield response;
    } else {
      // Handle any remaining chunks in the buffer for non-function calls
      while (buffer.length > 0) {
        const chunk = buffer.shift();
        const response =
          provider === Providers.OPENAI 
            ? chunk
            : ((await OutputFormatAdapter.adaptResponse({
                response: chunk,
                provider,
                isStream: true,
              })) as OpenAIResponse);
        yield response;
      }
    }
  }

  return streamGenerator();
}

export * from "./types";
