import { OpenAIFunctionMessage, OpenAIMessages } from "../../types";

export const adaptOpenAIMessages = (
  messages: any
): { adaptedMessages: OpenAIMessages } => {
  const adaptedMessages = messages.map((msg: any) => {
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
    };
  }) as OpenAIMessages;

  return { adaptedMessages };
};
