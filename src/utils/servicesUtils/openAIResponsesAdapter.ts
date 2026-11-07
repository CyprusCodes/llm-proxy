/* eslint-disable @typescript-eslint/no-explicit-any */
import { OpenAIResponse } from "../../types";

/**
 * GPT-5.x models are served through OpenAI's Responses API (/v1/responses)
 * instead of Chat Completions. Callers of this proxy still speak the Chat
 * Completions format, so this module translates requests into Responses API
 * bodies and translates responses / stream events back into
 * chat.completion / chat.completion.chunk objects.
 */
export function usesResponsesApi(model: string): boolean {
  return model.startsWith("gpt-5");
}

// ---------------------------------------------------------------------------
// Request: Chat Completions body -> Responses API body
// ---------------------------------------------------------------------------

function convertContentParts(content: any, role: string): any {
  if (!Array.isArray(content)) {
    return content ?? "";
  }
  const textType = role === "assistant" ? "output_text" : "input_text";
  return content.map((part: any) => {
    if (part?.type === "text") {
      return { type: textType, text: part.text ?? "" };
    }
    if (part?.type === "image_url") {
      const imageUrl =
        typeof part.image_url === "string" ? part.image_url : part.image_url?.url;
      return {
        type: "input_image",
        image_url: imageUrl,
        detail: part.image_url?.detail ?? "auto"
      };
    }
    if (part?.type === "file") {
      return { type: "input_file", ...part.file };
    }
    return part;
  });
}

function stringifyArguments(args: any): string {
  if (typeof args === "string") return args;
  return JSON.stringify(args ?? {});
}

function contentToString(content: any): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part: any) => (typeof part === "string" ? part : part?.text ?? ""))
      .join("");
  }
  if (content == null) return "";
  return JSON.stringify(content);
}

export function chatMessagesToResponsesInput(messages: any[]): any[] {
  const input: any[] = [];
  // Legacy `function` messages reference a call by name only; remember the
  // most recent call id for each function name so outputs can be paired.
  const lastCallIdByName: Record<string, string> = {};
  let syntheticId = 0;
  const newCallId = (): string => {
    syntheticId += 1;
    return `call_proxy_${Date.now()}_${syntheticId}`;
  };

  for (const msg of messages || []) {
    if (!msg) continue;

    switch (msg.role) {
      case "system":
      case "developer":
      case "user": {
        input.push({
          role: msg.role,
          content: convertContentParts(msg.content, msg.role)
        });
        break;
      }

      case "assistant": {
        const hasText =
          (typeof msg.content === "string" && msg.content.length > 0) ||
          (Array.isArray(msg.content) && msg.content.length > 0);
        if (hasText) {
          input.push({
            role: "assistant",
            content: convertContentParts(msg.content, "assistant")
          });
        }
        if (Array.isArray(msg.tool_calls)) {
          for (const toolCall of msg.tool_calls) {
            const name = toolCall?.function?.name;
            const callId = toolCall?.id || newCallId();
            if (name) lastCallIdByName[name] = callId;
            input.push({
              type: "function_call",
              call_id: callId,
              name,
              arguments: stringifyArguments(toolCall?.function?.arguments)
            });
          }
        }
        if (msg.function_call?.name) {
          const callId = newCallId();
          lastCallIdByName[msg.function_call.name] = callId;
          input.push({
            type: "function_call",
            call_id: callId,
            name: msg.function_call.name,
            arguments: stringifyArguments(msg.function_call.arguments)
          });
        }
        break;
      }

      case "tool": {
        input.push({
          type: "function_call_output",
          call_id: msg.tool_call_id,
          output: contentToString(msg.content)
        });
        break;
      }

      case "function": {
        const callId = lastCallIdByName[msg.name];
        if (callId) {
          input.push({
            type: "function_call_output",
            call_id: callId,
            output: contentToString(msg.content)
          });
        } else {
          // No matching call — surface the result as plain context instead
          // of sending an orphaned output that the API would reject.
          input.push({
            role: "user",
            content: `Result of function ${msg.name}: ${contentToString(
              msg.content
            )}`
          });
        }
        break;
      }

      default:
        input.push(msg);
    }
  }

  return input;
}

export function chatToolsToResponsesTools(tools: any): any {
  if (!Array.isArray(tools)) return tools;
  return tools.map((tool: any) => {
    if (tool?.type === "function" && tool.function) {
      const { name, description, parameters, strict } = tool.function;
      return {
        type: "function",
        name,
        ...(description !== undefined ? { description } : {}),
        parameters: parameters ?? { type: "object", properties: {} },
        strict: strict ?? false
      };
    }
    // Legacy `functions` shape: { name, description, parameters }
    if (!tool?.type && tool?.name) {
      return {
        type: "function",
        name: tool.name,
        ...(tool.description !== undefined
          ? { description: tool.description }
          : {}),
        parameters: tool.parameters ?? { type: "object", properties: {} },
        strict: tool.strict ?? false
      };
    }
    return tool;
  });
}

export function chatToolChoiceToResponsesToolChoice(toolChoice: any): any {
  if (toolChoice?.type === "function" && toolChoice.function?.name) {
    return { type: "function", name: toolChoice.function.name };
  }
  return toolChoice;
}

/** Translates a Chat Completions request body into a Responses API body. */
export function chatRequestToResponsesRequest(chatBody: any): any {
  const body: any = {
    model: chatBody.model,
    input: chatMessagesToResponsesInput(chatBody.messages),
    store: false
  };

  if (chatBody.temperature !== undefined) {
    body.temperature = chatBody.temperature;
  }

  const maxTokens = chatBody.max_completion_tokens ?? chatBody.max_tokens;
  if (typeof maxTokens === "number") {
    body.max_output_tokens = maxTokens;
  }

  if (chatBody.tools) {
    body.tools = chatToolsToResponsesTools(chatBody.tools);
  }
  if (chatBody.tool_choice !== undefined) {
    body.tool_choice = chatToolChoiceToResponsesToolChoice(
      chatBody.tool_choice
    );
  }
  if (chatBody.reasoning_effort) {
    body.reasoning = { effort: chatBody.reasoning_effort };
  }
  if (chatBody.verbosity) {
    body.text = { verbosity: chatBody.verbosity };
  }
  if (chatBody.stream) {
    body.stream = true;
  }

  return body;
}

// ---------------------------------------------------------------------------
// Response: Responses API -> Chat Completions
// ---------------------------------------------------------------------------

function mapUsage(usage: any): OpenAIResponse["usage"] {
  const promptTokens = usage?.input_tokens ?? 0;
  const completionTokens = usage?.output_tokens ?? 0;
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: usage?.total_tokens ?? promptTokens + completionTokens,
    prompt_tokens_details: {
      cached_tokens: usage?.input_tokens_details?.cached_tokens ?? 0
    },
    completion_tokens_details: {
      reasoning_tokens: usage?.output_tokens_details?.reasoning_tokens ?? 0
    }
  };
}

function mapFinishReason(response: any, hasToolCalls: boolean): string {
  if (response?.status === "incomplete") {
    const reason = response?.incomplete_details?.reason;
    if (reason === "max_output_tokens") return "length";
    if (reason === "content_filter") return "content_filter";
  }
  return hasToolCalls ? "tool_calls" : "stop";
}

function createdSeconds(response: any): number {
  return typeof response?.created_at === "number"
    ? Math.floor(response.created_at)
    : Math.floor(Date.now() / 1000);
}

/** Translates a non-streaming Responses API result into a chat.completion. */
export function responsesToChatCompletion(response: any): OpenAIResponse {
  let content = "";
  let refusal: string | null = null;
  const toolCalls: any[] = [];

  for (const item of response?.output || []) {
    if (item?.type === "message") {
      for (const part of item.content || []) {
        if (part?.type === "output_text") {
          content += part.text ?? "";
        } else if (part?.type === "refusal") {
          refusal = (refusal ?? "") + (part.refusal ?? "");
        }
      }
    } else if (item?.type === "function_call") {
      toolCalls.push({
        id: item.call_id,
        type: "function",
        function: { name: item.name, arguments: item.arguments ?? "" }
      });
    }
  }

  const message: any = {
    role: "assistant",
    content: content.length > 0 || toolCalls.length === 0 ? content : null,
    refusal,
    annotations: []
  };
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  return {
    id: response?.id,
    object: "chat.completion",
    created: createdSeconds(response),
    model: response?.model,
    choices: [
      {
        index: 0,
        message,
        logprobs: null,
        finish_reason: mapFinishReason(response, toolCalls.length > 0)
      }
    ],
    usage: mapUsage(response?.usage),
    system_fingerprint: ""
  } as OpenAIResponse;
}

/**
 * Translates a Responses API event stream into chat.completion.chunk objects,
 * mirroring Chat Completions streaming with `include_usage: true` (a final
 * chunk with empty `choices` carries the usage).
 */
export async function* responsesStreamToChatChunks(
  events: AsyncIterable<any>
): AsyncGenerator<any, void, unknown> {
  let id = "";
  let model = "";
  let created = Math.floor(Date.now() / 1000);
  let sentRole = false;
  // Maps Responses output_index -> chat tool_calls index
  const toolIndexByOutputIndex = new Map<number, number>();
  const toolIndexByItemId = new Map<string, number>();
  let toolCount = 0;

  const chunk = (delta: any, finishReason: string | null = null): any => {
    const fullDelta = sentRole ? delta : { role: "assistant", ...delta };
    sentRole = true;
    return {
      id,
      object: "chat.completion.chunk",
      created,
      model,
      system_fingerprint: "",
      choices: [
        { index: 0, delta: fullDelta, logprobs: null, finish_reason: finishReason }
      ]
    };
  };

  const toolIndexFor = (event: any): number | undefined => {
    if (event?.item_id && toolIndexByItemId.has(event.item_id)) {
      return toolIndexByItemId.get(event.item_id);
    }
    return toolIndexByOutputIndex.get(event?.output_index);
  };

  for await (const event of events) {
    switch (event?.type) {
      case "response.created":
      case "response.in_progress": {
        id = event.response?.id || id;
        model = event.response?.model || model;
        created = createdSeconds(event.response);
        break;
      }

      case "response.output_text.delta": {
        if (event.delta) {
          yield chunk({ content: event.delta });
        }
        break;
      }

      case "response.refusal.delta": {
        if (event.delta) {
          yield chunk({ refusal: event.delta });
        }
        break;
      }

      case "response.output_item.added": {
        const item = event.item;
        if (item?.type === "function_call") {
          const index = toolCount;
          toolCount += 1;
          toolIndexByOutputIndex.set(event.output_index, index);
          if (item.id) toolIndexByItemId.set(item.id, index);
          yield chunk({
            tool_calls: [
              {
                index,
                id: item.call_id,
                type: "function",
                function: { name: item.name, arguments: item.arguments ?? "" }
              }
            ]
          });
        }
        break;
      }

      case "response.function_call_arguments.delta": {
        const index = toolIndexFor(event);
        if (index !== undefined && event.delta) {
          yield chunk({
            tool_calls: [{ index, function: { arguments: event.delta } }]
          });
        }
        break;
      }

      case "response.completed":
      case "response.incomplete": {
        const response = event.response;
        id = response?.id || id;
        model = response?.model || model;
        yield chunk({}, mapFinishReason(response, toolCount > 0));
        yield {
          id,
          object: "chat.completion.chunk",
          created,
          model,
          system_fingerprint: "",
          choices: [],
          usage: mapUsage(response?.usage)
        };
        break;
      }

      case "response.failed": {
        const err = event.response?.error;
        throw Object.assign(
          new Error(err?.message || "OpenAI Responses API request failed."),
          { code: err?.code, type: "response.failed" }
        );
      }

      case "error": {
        throw Object.assign(
          new Error(event.message || "OpenAI Responses API stream error."),
          { code: event.code, param: event.param }
        );
      }

      default:
        break;
    }
  }
}
