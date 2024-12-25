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
    tools?: any;
  }): AsyncGenerator<any, void, unknown> {
    const { messages, model, max_tokens, temperature, tools } = params;

    if (!model) {
      throw new Error("Model ID is required.");
    }

    const toolPrompt = `
You have access to the following functions:

${JSON.stringify(tools, null, 2)} 

If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function>
{
"function_name": "the name of the function you want to call"
"parameters": {
  "parameter_key": "the value of the parameter"
  }
}
</function>

Example:
<function>
{
"function_name": "get_current_weather"
"parameters": {
  "location": "Miami"
  }
}

</function>

Reminder:
- Function calls MUST follow the specified format, start with <function> and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
`;

    const systemMessageRegex =
      /(<\|start_header_id\|>system<\|end_header_id\|>\n+)([\s\S]*?)(<\|eot_id\|>)/;

    let updatedMessages = messages;
    const match = messages.match(systemMessageRegex);
    if (match) {
      updatedMessages = messages.replace(
        systemMessageRegex,
        // @ts-ignore: Ignore the any type in the msg
        (_, start, systemMessage, end) =>
          `${start}${systemMessage}\n${toolPrompt}${end}`
      );
    } else {
      console.error(
        "Regex did not match the expected structure. Please verify the messages content."
      );
    }

    const body = JSON.stringify({
      prompt: updatedMessages,
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
