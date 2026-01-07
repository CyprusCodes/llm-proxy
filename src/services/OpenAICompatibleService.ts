import OpenAI from "openai";
import { v4 as uuid } from "uuid";
import { OpenAIMessages, OpenAIResponse } from "../types";
import { ClientService } from "./ClientService";

function normalizeTools(tools?: any[]): any[] | undefined {
  if (!Array.isArray(tools)) return undefined;

  return tools.map((tool) => {
    // Already correct
    if (tool?.type === "function" && tool.function) {
      return tool;
    }

    // Needs reformatting
    if (tool?.type === "function" && tool.name && tool.parameters) {
      return {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      };
    }

    // Unknown / unsupported tool shape
    throw new Error(
      `Invalid tool definition encountered: ${JSON.stringify(tool, null, 2)}`
    );
  });
}

function reformatMessagesWithThoughtSignature(
  messages: OpenAIMessages
): OpenAIMessages {
  const reformatted: OpenAIMessages = [];
  let i = 0;

  const makeToolCallId = (): string => {
    const prefix = "call_";
    const base = uuid().replace(/-/g, ""); // 32 chars
    const id = `${prefix}${base}`; // 37 chars total
    return id.length > 40 ? id.slice(0, 40) : id;
  };

  while (i < messages.length) {
    const message = messages[i];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const msg = message as any;

    // Check if this is an assistant message with content as an object containing name and arguments
    const isAssistantWithToolContent =
      msg.role === "assistant" &&
      typeof msg.content === "object" &&
      msg.content !== null &&
      !Array.isArray(msg.content) &&
      msg.content.name &&
      msg.content.arguments;

    if (isAssistantWithToolContent) {
      // Look ahead to find the corresponding function message
      const nextMessage = i + 1 < messages.length ? messages[i + 1] : null;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const nextMsg = nextMessage as any;

      // Generate a unique ID for the tool call (<= 40 chars)
      const toolCallId = makeToolCallId();

      // Extract thought_signature from the next function message if it exists
      const thoughtSignature =
        nextMsg?.role === "function" && nextMsg?.thought_signature
          ? nextMsg.thought_signature
          : undefined;

      // Convert assistant message to have tool_calls
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const assistantMessage: any = {
        role: "assistant",
        tool_calls: [
          {
            id: toolCallId,
            type: "function",
            function: {
              name: msg.content.name,
              arguments: msg.content.arguments,
            },
            ...(thoughtSignature && {
              extra_content: {
                google: {
                  thought_signature: thoughtSignature,
                },
              },
            }),
          },
        ],
      };

      reformatted.push(assistantMessage);

      // If there's a corresponding function message, convert it to a tool message
      if (nextMsg?.role === "function" && nextMsg?.name === msg.content.name) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const toolMessage: any = {
          role: "tool",
          name: nextMsg.name,
          tool_call_id: toolCallId,
          content: nextMsg.content,
        };

        reformatted.push(toolMessage);
        i += 2; // Skip both the assistant and function messages
      } else {
        i += 1;
      }
    } else {
      // Check if this is a function message that wasn't already processed
      const prevMsg = reformatted[reformatted.length - 1];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const prevToolCalls = (prevMsg as any)?.tool_calls?.[0]?.function?.name;
      const isUnprocessedFunction =
        msg.role === "function" &&
        !(prevMsg?.role === "assistant" && prevToolCalls === msg.name);

      if (isUnprocessedFunction) {
        // Check if message has thought_signature and function_call (old format - legacy support)
        if (msg.thought_signature && msg.function_call) {
          const { thought_signature, function_call, ...rest } = msg;

          // Convert to new tool_calls format
          reformatted.push({
            ...rest,
            role: "assistant",
            tool_calls: [
              {
                id: makeToolCallId(),
                type: "function",
                function: {
                  name: function_call.name,
                  arguments: function_call.arguments,
                },
                ...(thought_signature && {
                  extra_content: {
                    google: {
                      thought_signature,
                    },
                  },
                }),
              },
            ],
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          } as any);

          i += 1;
        } else {
          // Return message as-is if it doesn't need reformatting
          if (message) {
            reformatted.push(message);
          }
          i += 1;
        }
      } else {
        // Return message as-is if it doesn't need reformatting
        if (message) {
          reformatted.push(message);
        }
        i += 1;
      }
    }
  }

  return reformatted;
}

export default class OpenAICompatibleService implements ClientService {
  private openai: OpenAI;

  constructor(apiKey: string, baseUrl?: string) {
    this.openai = new OpenAI({
      apiKey,
      baseURL: baseUrl,
    });
  }

  async generateCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    tools?: any;
  }): Promise<OpenAIResponse> {
    if (!model) {
      return Promise.reject(
        new Error("Model ID is required for OpenAIService.")
      );
    }

    try {
      const reformattedMessages =
        reformatMessagesWithThoughtSignature(messages);
      const normalizedTools = normalizeTools(tools);
      const response = await this.openai.chat.completions.create({
        model,
        messages: reformattedMessages,
        max_tokens,
        temperature,
        ...(normalizedTools && { tools: normalizedTools }),
      });
      return response as OpenAIResponse;
    } catch (error) {
      return Promise.reject(error);
    }
  }

  async *generateStreamCompletion({
    messages,
    model,
    max_tokens,
    temperature,
    tools,
  }: {
    messages: OpenAIMessages;
    model: string;
    max_tokens: number;
    temperature: number;
    systemPrompt?: string;
    tools?: any;
  }): AsyncGenerator<any, void, unknown> {
    if (!model) {
      throw new Error("Model ID is required for OpenAIService.");
    }

    const reformattedMessages = reformatMessagesWithThoughtSignature(messages);
    const normalizedTools = normalizeTools(tools);
    const stream = await this.openai.chat.completions.create({
      model,
      messages: reformattedMessages,
      max_tokens,
      temperature,
      ...(normalizedTools && { tools: normalizedTools }),
      stream: true,
      stream_options: {
        include_usage: true,
      },
    });

    for await (const chunk of stream) {
      yield chunk;
    }
  }
}
