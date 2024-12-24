import {
  InvokeModelCommand,
  BedrockRuntimeClient,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";

export default class AwsBedrockLlama3Service {
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
    model?: string;
    max_tokens?: number;
    temperature?: number;
    messages: any;
  }): Promise<any> {
    const { messages, model, max_tokens, temperature } = params;

    if (!model) {
      throw new Error("Model ID is required.");
    }
    const body = {
      prompt: messages,
      max_gen_len: max_tokens || 100,
      temperature: temperature || 0.7,
      top_p: 0.9,
    };

    const command = new InvokeModelCommand({
      modelId: model,
      body: JSON.stringify(body),
      contentType: "application/json",
    });

    try {
      const response = await this.bedrock.send(command);
      return JSON.parse(new TextDecoder().decode(response.body));
    } catch (error) {
      console.error("Error invoking the Llama model:", error);
      throw error;
    }
  }

  async *generateStreamCompletion(params: {
    model?: string;
    max_tokens?: number;
    temperature?: number;
    messages: any;
  }): AsyncGenerator<any, void, unknown> {
    const { messages, model, max_tokens, temperature } = params;

    if (!model) {
      throw new Error("Model ID is required.");
    }
    const body = JSON.stringify({
      prompt: messages,
      max_gen_len: max_tokens || 100,
      temperature: temperature || 0.7,
      top_p: 0.9,
    });

    const command = new InvokeModelWithResponseStreamCommand({
      modelId: model,
      body,
      contentType: "application/json",
    });

    try {
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
            throw new Error("Failed to parse the streaming response.");
          }
        }
      }
    } catch (error) {
      console.error("Error invoking the Llama model with stream:", error);
      throw error;
    }
  }
}
