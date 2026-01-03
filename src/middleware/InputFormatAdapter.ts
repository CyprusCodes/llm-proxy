import {
  BedrockAnthropicContentType,
  BedrockAnthropicMessage,
  OpenAIFunctionMessage,
  OpenAIMessages,
  Providers,
} from "../types";
import openaiToLlamaMessage from "../utils/inputFormatAdapterUtils/openAIToLlamaMessage";

export default class InputFormatAdapter {
  static adaptMessages(
    messages: any,
    provider: Providers
  ): {
    adaptedMessages: OpenAIMessages | BedrockAnthropicMessage[] | any;
    systemPrompt?: string;
  } {
    switch (provider) {
      case Providers.OPENAI:
      case Providers.OPENAI_COMPATIBLE_PROVIDER:
        return {
          // @ts-ignore: Ignore the any type in the msg
          adaptedMessages: messages
            .map((msg: any) => {
              // Skip 'function' role messages - they should have been converted to tool format earlier
              if (msg.role === "function") {
                return null;
              }

              // Preserve the original message structure, including tool_calls if present
              const adaptedMsg: any = {
                role: msg.role,
                content: msg.content,
                ...(msg.thought_signature && {
                  thought_signature: msg.thought_signature,
                }),
              };

              // If it's an assistant message with tool_calls, preserve them
              if (msg.role === "assistant" && msg.tool_calls) {
                adaptedMsg.tool_calls = msg.tool_calls;
                // content can be null when tool_calls is present
                if (adaptedMsg.content === undefined) {
                  adaptedMsg.content = null;
                }
              } else if (msg.role === "tool") {
                // For tool role, preserve tool_call_id and name
                adaptedMsg.tool_call_id = msg.tool_call_id;
                adaptedMsg.name = msg.name;
                // Ensure content is a string for tool messages
                if (!adaptedMsg.content) {
                  adaptedMsg.content = "";
                }
              } else {
                // For other messages, ensure content is not null/undefined
                if (
                  adaptedMsg.content === null ||
                  adaptedMsg.content === undefined
                ) {
                  adaptedMsg.content = "";
                }
              }

              return adaptedMsg;
            })
            .filter(Boolean) as OpenAIMessages,
        };

      case Providers.ANTHROPIC_BEDROCK: {
        if (!messages.length) {
          throw new Error("Messages array cannot be empty for Anthropic.");
        }

        // Extract the first message as the system prompt
        const [firstMessage, ...restMessages] = messages;

        if (firstMessage.role !== "system") {
          throw new Error(
            "The first message must have a role of 'system' for Anthropic."
          );
        }

        const systemPrompt = firstMessage.content ?? "";
        const adaptedMessages: any = [];
        // @ts-ignore: Ignore the any type in the msg
        restMessages.forEach((msg) => {
          if (msg.role !== "user" && msg.role !== "assistant") {
            // Add the "empty" message before the current one
            adaptedMessages.push({
              role: "user",
              content: [
                {
                  type: BedrockAnthropicContentType.TEXT,
                  text: ":",
                },
              ],
            });

            // Change the role to "assistant" for the current message
            adaptedMessages.push({
              role: "assistant",
              content: [
                {
                  type: BedrockAnthropicContentType.TEXT,
                  text: msg.content ?? "",
                },
              ],
            });
          } else {
            // Add the message as-is
            adaptedMessages.push({
              role: msg.role,
              content: [
                {
                  type: BedrockAnthropicContentType.TEXT,
                  text: msg.content ?? msg.function_call.arguments,
                },
              ],
            });
          }
        });

        // Ensure no two consecutive messages have the same role
        for (let i = 0; i < adaptedMessages.length - 1; i += 1) {
          if (adaptedMessages[i].role === adaptedMessages[i + 1].role) {
            // Insert a placeholder message with the opposite role
            adaptedMessages.splice(i + 1, 0, {
              role: adaptedMessages[i].role === "user" ? "assistant" : "user",
              content: [
                {
                  type: BedrockAnthropicContentType.TEXT,
                  text: ":",
                },
              ],
            });
            i += 1; // Skip the inserted message
          }
        }

        return { adaptedMessages, systemPrompt };
      }
      case Providers.LLAMA_3_1_BEDROCK: {
        const adaptedMessages = openaiToLlamaMessage(messages);
        return {
          adaptedMessages,
        };
      }

      default:
        throw new Error(`Unsupported provider 1: ${provider}`);
    }
  }
}
