import {
  BedrockAnthropicContent,
  BedrockAnthropicContentType,
  BedrockAnthropicTextContent,
  BedrockAnthropicToolResultContent,
  BedrockAnthropicToolUseContent,
  LLMResponse,
  Providers
} from "../types";
import convertLlamaToOpenAINonStream from "../utils/outputFormatAdapterUtils/convertLlamaToOpenAINonStream";
import convertLlamaToOpenAIStream from "../utils/outputFormatAdapterUtils/convertLlamaToOpenAIStream";
import convertLlamaToOpenAIStreamToolCall from "../utils/outputFormatAdapterUtils/convertLlamaToOpenAIStreamToolCall";

export default class OutputFormatAdapter {
  // Cached model name from message_start during streaming
  private static model: string | undefined;

  static async adaptResponse({
    response,
    provider,
    isStream,
    isFunctionCall
  }: {
    response: any;
    provider: Providers;
    isStream: boolean;
    isFunctionCall?: boolean;
  }): Promise<any> {
    if (!response) {
      throw new Error("Response object is null or undefined");
    }
    try {
      switch (provider) {
        case Providers.OPENAI:
        case Providers.OPENAI_COMPATIBLE_PROVIDER:
          return response as LLMResponse;

        case Providers.ANTHROPIC:
        case Providers.ANTHROPIC_BEDROCK:
          if (!isStream) {
            return this.adaptCompleteResponse(response);
          }
          return this.adaptStreamingChunk(response, provider);

        case Providers.LLAMA_3_1_BEDROCK: {
          if (!isStream && !isFunctionCall) {
            return convertLlamaToOpenAINonStream(response);
          }
          if (isFunctionCall) {
            return convertLlamaToOpenAIStreamToolCall(response);
          }
          return convertLlamaToOpenAIStream(response);
        }

        default:
          throw new Error(`Unsupported provider: ${provider}`);
      }
    } catch (error) {
      throw new Error(`Failed to adapt response: ${(error as Error).message}`);
    }
  }


  private static adaptCompleteResponse(response: any): any {
    const model = response.model || "unknown-model";
    const usage = {
      prompt_tokens: response.usage?.input_tokens || 0,
      completion_tokens: response.usage?.output_tokens || 0,
      total_tokens:
        (response.usage?.input_tokens || 0) +
        (response.usage?.output_tokens || 0),
      prompt_tokens_details: { cached_tokens: 0 },
      completion_tokens_details: { reasoning_tokens: 0 }
    };

    // Check if response contains a tool_use content block
    const toolUseBlock = response.content?.find(
      (block: BedrockAnthropicContent) =>
        block.type === BedrockAnthropicContentType.TOOL_USE
    ) as BedrockAnthropicToolUseContent | undefined;

    if (toolUseBlock) {
      // Anthropic returns input as an object; OpenAI expects arguments as a string
      const argumentsString =
        typeof toolUseBlock.input === "string"
          ? toolUseBlock.input
          : JSON.stringify(toolUseBlock.input);

      return {
        id: response.id,
        object: "text_completion",
        created: Date.now(),
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              function_call: {
                name: toolUseBlock.name,
                arguments: argumentsString
              }
            },
            logprobs: null,
            finish_reason: response.stop_reason || null
          }
        ],
        usage,
        system_fingerprint: "default_fingerprint"
      };
    }

    return {
      id: response.id,
      object: "text_completion",
      created: Date.now(),
      model,
      choices: response.content.map(
        (contentBlock: BedrockAnthropicContent, index: number) => ({
          index,
          message: {
            role: this.mapRole(contentBlock),
            content: this.extractContent(contentBlock)
          },
          logprobs: null,
          finish_reason: response.stop_reason || null
        })
      ),
      usage,
      system_fingerprint: "default_fingerprint"
    };
  }

  private static adaptStreamingChunk(chunk: any, provider?: Providers): any {
    const bedrockMetrics = chunk["amazon-bedrock-invocationMetrics"];
    const anthropicUsage =
      provider === Providers.ANTHROPIC && chunk.usage
        ? {
            inputTokenCount: chunk.usage.input_tokens ?? 0,
            outputTokenCount: chunk.usage.output_tokens ?? 0
          }
        : null;
    const usageMetrics = anthropicUsage ?? bedrockMetrics;

    // Cache model from the first message_start chunk
    if (chunk.type === "message_start" && chunk.message?.model) {
      this.model = chunk.message.model;
    }

    const isMessageEnd = chunk.type === "message_stop";
    const content = chunk.delta?.text || "";
    const inputTokens = usageMetrics?.inputTokenCount ?? 0;
    const outputTokens = usageMetrics?.outputTokenCount ?? 0;

    const response = {
      id: `stream-${Date.now()}`,
      object: "chat.completion.chunk",
      created: Date.now(),
      model: this.model || "unknown-model",
      choices: [
        {
          index: 0,
          delta: { content },
          logprobs: null,
          finish_reason: isMessageEnd ? "stop" : null
        }
      ],
      usage: isMessageEnd
        ? {
            prompt_tokens: inputTokens,
            completion_tokens: outputTokens,
            total_tokens: inputTokens + outputTokens,
            prompt_tokens_details: { cached_tokens: 0 },
            completion_tokens_details: { reasoning_tokens: 0 }
          }
        : null
    };

    // Clean up cached state when message ends
    if (isMessageEnd) {
      this.model = undefined;
    }

    return response;
  }

  private static mapRole(content: BedrockAnthropicContent): string {
    if (!content || !content.type) {
      throw new Error("Invalid content block structure");
    }

    switch (content.type) {
      case BedrockAnthropicContentType.TOOL_USE:
      case BedrockAnthropicContentType.TOOL_RESULT:
        return "tool";
      case BedrockAnthropicContentType.TEXT:
      default:
        return "assistant";
    }
  }

  private static extractContent(content: BedrockAnthropicContent): string {
    if (!content || !content.type) {
      throw new Error("Invalid content block structure");
    }

    switch (content.type) {
      case BedrockAnthropicContentType.TEXT:
        return (content as BedrockAnthropicTextContent).text || "";
      case BedrockAnthropicContentType.TOOL_RESULT:
        return (content as BedrockAnthropicToolResultContent).content || "";
      case BedrockAnthropicContentType.TOOL_USE:
        return (content as BedrockAnthropicToolUseContent).id || "";
      default:
        return "";
    }
  }
}
