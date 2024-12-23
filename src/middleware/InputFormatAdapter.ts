import {
  BedrockAnthropicContentType,
  BedrockAnthropicMessage,
  OpenAIFunctionMessage,
  OpenAIMessages,
  Providers
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
        return {
          adaptedMessages: messages.map(msg => {
            if (msg.role === "function") {
              return {
                role: msg.role,
                content: msg.content ?? "",
                name: (msg as OpenAIFunctionMessage).name
              };
            }
            return {
              role: msg.role,
              content: msg.content ?? "function call"
            };
          }) as OpenAIMessages
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
        restMessages.forEach(msg => {
          if (msg.role !== "user" && msg.role !== "assistant") {
            // Add the "empty" message before the current one
            adaptedMessages.push({
              role: "user",
              content: [
                {
                  type: BedrockAnthropicContentType.TEXT,
                  text: ":"
                }
              ]
            });

            // Change the role to "assistant" for the current message
            adaptedMessages.push({
              role: "assistant",
              content: [
                {
                  type: BedrockAnthropicContentType.TEXT,
                  text: msg.content ?? ""
                }
              ]
            });
          } else {
            // Add the message as-is
            adaptedMessages.push({
              role: msg.role,
              content: [
                {
                  type: BedrockAnthropicContentType.TEXT,
                  text: msg.content ?? msg.function_call.arguments
                }
              ]
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
                  text: ":"
                }
              ]
            });
            i += 1; // Skip the inserted message
          }
        }

        return { adaptedMessages, systemPrompt };
      }
      case Providers.LLAMA_3_1_BEDROCK: {
        const adaptedMessages = openaiToLlamaMessage(messages);
        return {
          adaptedMessages
        };
      }

      default:
        throw new Error(`Unsupported provider: ${provider}`);
    }
  }
}
