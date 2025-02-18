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
  private static isToolUseStream = false;

  private static toolArguments: string[] = [];

  private static model: string | undefined;

  private static toolName: string | undefined; // New: To store the tool name

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
        case Providers.ANTHROPIC_BEDROCK:
          if (!isStream) {
            return this.adaptCompleteResponse(response);
          }
          return this.adaptStreamingResponse(response);

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
          throw new Error(`Unsupported provider 2: ${provider}`);
      }
    } catch (error) {
      throw new Error(`Failed to adapt response: ${(error as Error).message}`);
    }
  }

  private static adaptCompleteResponse(response: any): any {
    return {
      id: response.id,
      object: "text_completion",
      created: Date.now(),
      model: this.model || "unknown-model",
      choices: response.content.map(
        (contentBlock: BedrockAnthropicContent, index: any) => ({
          index,
          message: {
            role: this.mapRole(contentBlock),
            content: this.extractContent(contentBlock)
          },
          logprobs: null,
          finish_reason: response.stop_reason || null
        })
      ),
      usage: {
        prompt_tokens: response.usage?.input_tokens || 0,
        completion_tokens: response.usage?.output_tokens || 0,
        total_tokens:
          (response.usage?.input_tokens || 0) +
          (response.usage?.output_tokens || 0),
        prompt_tokens_details: { cached_tokens: 0 },
        completion_tokens_details: { reasoning_tokens: 0 }
      },
      system_fingerprint: response.system_fingerprint || "default_fingerprint"
    };
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
        return "assistant";
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

  private static adaptStreamingResponse(chunk: any): any {
    const metrics = chunk["amazon-bedrock-invocationMetrics"];
    const isStop =
      chunk.type === "content_block_stop" || chunk.type === "message_stop";

    // Cache model on the first `message_start` chunk
    if (chunk.type === "message_start" && chunk.message?.model) {
      this.model = chunk.message.model;
    }

    // Detect tool use for the current chunk and cache the tool name
    if (
      chunk.type === "content_block_start" &&
      chunk.content_block?.type === "tool_use"
    ) {
      this.isToolUseStream = true;
      this.toolName = chunk.content_block?.name || "unknown_tool"; // Capture tool name
    }

    // Accumulate tool arguments for tool-use streams
    if (
      chunk.type === "content_block_delta" &&
      chunk.delta?.type === "input_json_delta"
    ) {
      this.toolArguments.push(chunk.delta?.partial_json || "nothing");
    }

    // Handle the end of the stream
    if (isStop) {
      const response = this.isToolUseStream
        ? this.createToolUseResponse(metrics, isStop)
        : this.createNonToolUseResponse(metrics, isStop, chunk);

      // Reset state after processing the end of the stream
      this.resetState();

      return response;
    }

    // Handle intermediate chunks (non-stop chunks)
    return this.isToolUseStream
      ? this.createToolUseResponse(metrics, isStop, chunk)
      : this.createNonToolUseResponse(metrics, isStop, chunk);
  }

  private static createToolUseResponse(
    metrics: any,
    isStop: boolean,
    chunk?: any
  ): any {
    return {
      id: `stream-${Date.now()}`,
      object: "chat.completion.chunk",
      created: Date.now(),
      model: this.model || "unknown-model",
      system_fingerprint: "anthropic_translation",
      choices: [
        {
          index: 0,
          delta: {
            function_call: {
              name: this.toolName || "unknown_tool", // Include tool name
              arguments:
                chunk &&
                chunk.type === "content_block_delta" &&
                chunk.delta?.type === "input_json_delta"
                  ? chunk.delta?.partial_json
                  : this.toolArguments.join(", ")
            }
          },
          logprobs: null,
          finish_reason: isStop ? "stop" : null
        }
      ],
      usage: isStop
        ? {
            prompt_tokens: metrics?.inputTokenCount || 0,
            completion_tokens: metrics?.outputTokenCount || 0,
            total_tokens:
              (metrics?.inputTokenCount || 0) +
              (metrics?.outputTokenCount || 0),
            prompt_tokens_details: {
              cached_tokens: 0
            },
            completion_tokens_details: {
              reasoning_tokens: 0
            }
          }
        : null
    };
  }

  private static createNonToolUseResponse(
    metrics: any,
    isStop: boolean,
    chunk: any
  ): any {
    const content = chunk.content_block?.text || chunk.delta?.text || "";
    return {
      id: `stream-${Date.now()}`,
      object: "chat.completion.chunk",
      created: Date.now(),
      model: this.model || "unknown-model",
      choices: [
        {
          index: 0,
          delta: {
            content
          },
          logprobs: null,
          finish_reason: isStop ? "stop" : null
        }
      ],
      usage: isStop
        ? {
            prompt_tokens: metrics?.inputTokenCount || 0,
            completion_tokens: metrics?.outputTokenCount || 0,
            total_tokens:
              (metrics?.inputTokenCount || 0) +
              (metrics?.outputTokenCount || 0),
            prompt_tokens_details: {
              cached_tokens: 0
            },
            completion_tokens_details: {
              reasoning_tokens: 0
            }
          }
        : null
    };
  }

  private static resetState() {
    this.isToolUseStream = false;
    this.toolArguments = [];
    this.model = undefined;
    this.toolName = undefined;
  }
}
