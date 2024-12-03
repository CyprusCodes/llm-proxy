import {
  InvokeModelCommand,
  BedrockRuntimeClient,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import {
  BedrockAnthropicParsedChunk,
  BedrockAnthropicResponse,
  Messages,
} from "../types";
import { ClientService } from "./ClientService";
import LLM_PROXY_ERROR_MESSAGES from "../constants/errorMessages";
import { BedrockAnthropicMessages } from "llm-proxy";

export default class AwsBedrockAnthropicService implements ClientService {
  private bedrock: BedrockRuntimeClient;

  constructor(awsAccessKey: string, awsSecretKey: string, region: string) {
    this.bedrock = new BedrockRuntimeClient({
      region,
      credentials: {
        accessKeyId: awsAccessKey,
        secretAccessKey: awsSecretKey,
      },
    });
  }

  async generateCompletion(params: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any; // TODO: Define the correct type
    systemPrompt?: string;
  }): Promise<BedrockAnthropicResponse> {
    const { messages, model, max_tokens, temperature, systemPrompt, tools } =
      params;

    if (!model) {
      return Promise.reject(new Error(LLM_PROXY_ERROR_MESSAGES.MISSING_MODEL));
    }

    const body = JSON.stringify({
      anthropic_version: "bedrock-2023-05-31",
      max_tokens,
      temperature,
      messages: messages as BedrockAnthropicMessages,
      system: systemPrompt,
      ...(tools && Array.isArray(tools) && tools.length ? { tools } : {}),
    });

    const command = new InvokeModelCommand({
      modelId: model,
      body,
      contentType: "application/json",
      accept: "application/json",
    });

    const response = await this.bedrock.send(command);
    return JSON.parse(new TextDecoder().decode(response.body));
  }

  // eslint-disable-next-line consistent-return
  async *generateStreamCompletion(params: {
    messages: Messages;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools?: any; // TODO: Define the correct type
    systemPrompt?: string;
  }): AsyncGenerator<BedrockAnthropicParsedChunk, void, unknown> {
    const { messages, model, max_tokens, temperature, tools, systemPrompt } =
      params;

    if (!model) {
      return Promise.reject(new Error(LLM_PROXY_ERROR_MESSAGES.MISSING_MODEL));
    }

    const body = JSON.stringify({
      anthropic_version: "bedrock-2023-05-31",
      max_tokens,
      temperature,
      messages: messages as BedrockAnthropicMessages,
      system: systemPrompt,
      ...(tools && Array.isArray(tools) && tools.length ? { tools } : {}),
    });

    const command = new InvokeModelWithResponseStreamCommand({
      modelId: model,
      body,
      contentType: "application/json",
      accept: "application/json",
    });

    const response = await this.bedrock.send(command);

    if (response.body) {
      const decoder = new TextDecoder("utf-8");

      for await (const payload of response.body) {
        const decodedString = decoder.decode(payload.chunk?.bytes, {
          stream: true,
        });

        try {
          const jsonObject = JSON.parse(decodedString);
          yield jsonObject;
        } catch (error) {
          return Promise.reject(error);
        }
      }
    }
  }
}
