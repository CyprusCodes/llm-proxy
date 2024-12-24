import { Messages, OpenAIResponse, Providers } from "./types";
import OpenAIService from "./services/OpenAIService";
import AwsBedrockAnthropicService from "./services/AwsBedrockAnthropicService";
import ProviderFinder from "./middleware/ProviderFinder";
import InputFormatAdapter from "./middleware/InputFormatAdapter";
import OutputFormatAdapter from "./middleware/OutputFormatAdapter";
import AwsBedrockLlama3Service from "./services/AwsBedrockLlama3Service";

// Define the credentials interface for flexibility
interface Credentials {
  apiKey?: string;
  awsConfig?: { accessKeyId: string; secretAccessKey: string; region: string };
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

  // Step 1: Identify the provider based on the model
  const provider = ProviderFinder.getProvider(model);

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
    provider === Providers.OPENAI
      ? response
      : OutputFormatAdapter.adaptResponse({
          response,
          provider,
          isStream: false,
        });
  return adaptedResponse as OpenAIResponse;
}

// Main function for streaming requests
export async function generateLLMStreamResponse(
  params: GenerateLLMResponseParams
): Promise<AsyncGenerator<OpenAIResponse>> {
  const { messages, model, functions, max_tokens, temperature, credentials } =
    params;

  // Step 1: Identify the provider based on the model
  const provider = ProviderFinder.getProvider(model);

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
  } else {
    return Promise.reject(new Error("Unsupported provider 3"));
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
    for await (const chunk of stream) {
      yield provider === Providers.OPENAI
        ? (chunk as OpenAIResponse)
        : ((await OutputFormatAdapter.adaptResponse({
            response: chunk,
            provider,
            isStream: true,
          })) as OpenAIResponse);
    }
  }

  return streamGenerator();
}

export * from "./types";
