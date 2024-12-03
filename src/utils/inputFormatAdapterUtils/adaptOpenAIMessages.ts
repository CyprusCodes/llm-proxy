import { OpenAIMessages } from "../../types";

function adaptOpenAIMessages(messages: OpenAIMessages) {
  return messages.map((msg) => {
    if (msg.role === "function") {
      return {
        role: msg.role,
        content: msg.content ?? "",
        name: msg.name,
      };
    }
    return {
      role: msg.role,
      content: msg.content ?? "function call",
    };
  });
}

export default adaptOpenAIMessages;
