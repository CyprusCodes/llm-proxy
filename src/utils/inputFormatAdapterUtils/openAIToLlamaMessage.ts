import { OpenAIMessages, OpenAIMessage } from "llm-proxy";
// Ensure OpenAIMessage is the base type for all message variants

const openaiToLlamaMessage = (openaiMessages: OpenAIMessages): string => {
  let llamaPrompt = "<|begin_of_text|>";

  // Explicitly type updatedMessages as an array of OpenAIMessage
  const updatedMessages: OpenAIMessage[] = [];

  openaiMessages.forEach(msg => {
    const { role, content } = msg;

    if (role === "function") {
      // Add a user message with empty content
      updatedMessages.push({ role: "user", content: "" });

      // Add an assistant message with the content of the function message
      updatedMessages.push({ role: "assistant", content });
    } else {
      // Add the original message for other roles
      updatedMessages.push(msg);
    }
  });

  updatedMessages.forEach(msg => {
    const { role, content } = msg;

    const validRoles = ["system", "user", "assistant"];
    if (!validRoles.includes(role)) {
      throw new Error(`Invalid role: ${role}`);
    }

    llamaPrompt += `\n<|start_header_id|>${role}<|end_header_id|>\n${content}\n<|eot_id|>`;
  });

  const lastMessage = updatedMessages[updatedMessages.length - 1];
  if (lastMessage?.role === "user") {
    llamaPrompt += `\n<|start_header_id|>assistant<|end_header_id|>`;
  }

  return llamaPrompt;
};

export default openaiToLlamaMessage;
