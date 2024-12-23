import {
  InvokeModelCommand,
  BedrockRuntimeClient,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { BedrockAnthropicParsedChunk, Messages } from "../types";
import { ClientService } from "./ClientService";

export default class AwsBedrockLlama3Service implements ClientService {
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
  }): Promise<any> {
    const { messages, model, max_tokens, temperature } = params;

    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for AwsBedrockAnthropicService")
      );
    }

    const body = {
      prompt: messages,
      max_gen_len: max_tokens,
      temperature,
      top_p: 0.9,
    };

    const command = new InvokeModelCommand({
      modelId: model,
      body: JSON.stringify(body),
      contentType: "application/json",
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
    const { messages, model, max_tokens, temperature } = params;

    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for AwsBedrockAnthropicService")
      );
    }

    const body = JSON.stringify({
      prompt: messages,
      max_gen_len: max_tokens,
      temperature,
      top_p: 0.9,
    });

    const command = new InvokeModelWithResponseStreamCommand({
      modelId: model,
      body: JSON.stringify(body),
      contentType: "application/json",
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
