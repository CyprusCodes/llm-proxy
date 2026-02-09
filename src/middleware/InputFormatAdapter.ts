import { v4 as uuid } from "uuid";
import {
  BedrockAnthropicContentType,
  BedrockAnthropicMessage,
  OpenAIFunctionMessage,
  OpenAIMessages,
  Providers,
} from "../types";
import openaiToLlamaMessage from "../utils/inputFormatAdapterUtils/openAIToLlamaMessage";

// Helper function to convert function messages to tool messages for GPT 5.x models
function convertFunctionToToolMessages(
  messages: OpenAIMessages,
  model: string
): OpenAIMessages {
  // Check if this is a GPT 5.x model that requires tool format
  const isGPT5Model = model.startsWith("gpt-5");
  if (!isGPT5Model) {
    return messages;
  }

  const makeToolCallId = (): string => {
    const prefix = "call_";
    const base = uuid().replace(/-/g, ""); // 32 chars
    const id = `${prefix}${base}`; // 37 chars total
    return id.length > 40 ? id.slice(0, 40) : id;
  };

  const reformatted: OpenAIMessages = [];
  let i = 0;

  while (i < messages.length) {
    const message = messages[i];
    if (!message) {
      i += 1;
      continue;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const msg = message as any;

    // Check if this is an assistant message with function_call (old format)
    const hasFunctionCall =
      msg.role === "assistant" &&
      msg.function_call &&
      typeof msg.function_call === "object" &&
      msg.function_call.name &&
      msg.function_call.arguments;

    // Check if this is an assistant message with tool_calls (already converted format)
    const hasToolCalls =
      msg.role === "assistant" &&
      msg.tool_calls &&
      Array.isArray(msg.tool_calls) &&
      msg.tool_calls.length > 0;

    if (hasFunctionCall) {
      // Look ahead to find the corresponding function message
      const nextMessage = i + 1 < messages.length ? messages[i + 1] : null;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const nextMsg = nextMessage as any;

      // Only convert to tool_calls if there's a corresponding function message
      // Otherwise, GPT 5.2 will error because tool_calls must be followed by tool messages
      if (nextMsg?.role === "function" && nextMsg?.name === msg.function_call.name) {
        // Generate a unique ID for the tool call (<= 40 chars)
        const toolCallId = makeToolCallId();

        // Convert assistant message to have tool_calls instead of function_call
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const assistantMessage: any = {
          role: "assistant",
          content: msg.content || null,
          tool_calls: [
            {
              id: toolCallId,
              type: "function",
              function: {
                name: msg.function_call.name,
                arguments:
                  typeof msg.function_call.arguments === "string"
                    ? msg.function_call.arguments
                    : JSON.stringify(msg.function_call.arguments),
              },
            },
          ],
        };

        reformatted.push(assistantMessage);

        // Convert function message to a tool message
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const toolMessage: any = {
          role: "tool",
          name: nextMsg.name,
          tool_call_id: toolCallId,
          content: nextMsg.content ?? "",
        };

        reformatted.push(toolMessage);
        i += 2; // Skip both the assistant and function messages
      } else {
        // No corresponding function message - don't add tool_calls, just keep as regular assistant message
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const assistantMessage: any = {
          role: "assistant",
          content: msg.content || null,
        };
        reformatted.push(assistantMessage);
        i += 1;
      }
    } else if (hasToolCalls) {
      // This is an assistant message that already has tool_calls
      // GPT 5.2 requires that ALL tool_call_ids have corresponding tool messages immediately after
      // Check if the next messages are tool messages matching all tool_call_ids
      const toolCallIds = msg.tool_calls.map((tc: any) => tc.id);
      const toolMessages: any[] = [];
      let j = i + 1;
      let foundAllToolMessages = true;

      // Collect all consecutive tool messages that match the tool_call_ids
      while (j < messages.length && toolMessages.length < toolCallIds.length) {
        const nextMsg = messages[j];
        if (!nextMsg) break; // Safety check
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const nextMsgAny = nextMsg as any;
        if (
          nextMsgAny?.role === "tool" &&
          toolCallIds.includes(nextMsgAny?.tool_call_id)
        ) {
          toolMessages.push(nextMsg);
          j += 1;
        } else {
          // Stop if we hit a non-tool message
          break;
        }
      }

      // Check if we found tool messages for all tool_call_ids
      const foundToolCallIds = toolMessages.map((tm: any) => tm.tool_call_id);
      foundAllToolMessages =
        toolCallIds.every((id: string) => foundToolCallIds.includes(id)) &&
        toolMessages.length > 0;

      if (foundAllToolMessages) {
        // Keep tool_calls and add all matching tool messages
        reformatted.push(message);
        toolMessages.filter((tm) => tm !== undefined).forEach((tm) => {
          if (tm) reformatted.push(tm);
        });
        i = j; // Skip assistant message and all tool messages
      } else {
        // Not all tool_call_ids have corresponding tool messages - remove tool_calls to avoid GPT 5.2 error
        // This happens when a tool call failed or hasn't been executed yet
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const assistantMessage: any = {
          role: "assistant",
          content: msg.content || null,
        };
        reformatted.push(assistantMessage);
        i += 1;
      }
    } else if (msg.role === "function") {
      // This is a function message that needs to be converted to tool format
      // Look backwards in the reformatted array to find the corresponding assistant message
      let toolCallId: string | null = null;
      let foundAssistant = false;

      // Check if the last message in reformatted is an assistant with matching tool_calls
      if (reformatted.length > 0) {
        const lastMsg = reformatted[reformatted.length - 1];
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const lastMsgAny = lastMsg as any;
        if (
          lastMsgAny?.role === "assistant" &&
          lastMsgAny?.tool_calls?.[0]?.function?.name === msg.name
        ) {
          toolCallId = lastMsgAny.tool_calls[0].id;
          foundAssistant = true;
        }
      }

      // If not found, check the previous message in the original array
      if (!foundAssistant && i > 0) {
        const prevMsg = messages[i - 1];
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const prevMsgAny = prevMsg as any;
        if (prevMsgAny?.role === "assistant") {
          // Convert it to have tool_calls
          toolCallId = makeToolCallId();
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const assistantMessage: any = {
            role: "assistant",
            content: prevMsgAny.content === "function call" ? null : prevMsgAny.content || null,
            tool_calls: [
              {
                id: toolCallId,
                type: "function",
                function: {
                  name: msg.name,
                  arguments: prevMsgAny.function_call?.arguments || "{}",
                },
              },
            ],
          };

          // Replace the last message if it was this assistant message
          if (
            reformatted.length > 0 &&
            reformatted[reformatted.length - 1] === prevMsg
          ) {
            reformatted[reformatted.length - 1] = assistantMessage;
          } else {
            reformatted.push(assistantMessage);
          }
          foundAssistant = true;
        }
      }

      // If still no assistant found, create a new tool_call_id (shouldn't happen normally)
      if (!toolCallId) {
        toolCallId = makeToolCallId();
      }

      // Convert function message to tool message
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const toolMessage: any = {
        role: "tool",
        name: msg.name,
        tool_call_id: toolCallId,
        content: msg.content ?? "",
      };
      reformatted.push(toolMessage);
      i += 1;
    } else {
      // Return message as-is if it doesn't need reformatting
      reformatted.push(message);
      i += 1;
    }
  }

  return reformatted;
}

export default class InputFormatAdapter {
  static adaptMessages(
    messages: any,
    provider: Providers,
    model?: string
  ): {
    adaptedMessages: OpenAIMessages | BedrockAnthropicMessage[] | any;
    systemPrompt?: string;
  } {
    switch (provider) {
      case Providers.OPENAI:
        // First convert function messages to tool messages for GPT 5.x models
        let processedMessages = messages.map((msg: any) => {
          if (msg.role === "function") {
            return {
              role: msg.role,
              content: msg.content ?? "",
              name: (msg as OpenAIFunctionMessage).name,
            };
          }
          return {
            role: msg.role,
            content: msg.content ?? "function call",
            ...(msg.function_call && { function_call: msg.function_call }),
          };
        }) as OpenAIMessages;

        // Convert to tool format if needed
        if (model) {
          processedMessages = convertFunctionToToolMessages(
            processedMessages,
            model
          );
        }

        return {
          adaptedMessages: processedMessages,
        };

      case Providers.OPENAI_COMPATIBLE_PROVIDER:
        return {
          // @ts-ignore: Ignore the any type in the msg
          adaptedMessages: messages.map((msg) => {
            if (msg.role === "function") {
              return {
                role: msg.role,
                content: msg.content ?? "",
                ...(msg.thought_signature && {
                  thought_signature: msg.thought_signature,
                }),
                name: (msg as OpenAIFunctionMessage).name,
              };
            }
            return {
              role: msg.role,
              content: msg.content ?? msg.function_call,
            };
          }) as OpenAIMessages,
        };

      case Providers.ANTHROPIC:
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
