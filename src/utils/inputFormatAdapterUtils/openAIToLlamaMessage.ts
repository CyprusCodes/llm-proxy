import { OpenAIMessages } from "llm-proxy";

const openaiToLlamaMessage = (openaiMessages: OpenAIMessages): string => {
  let llamaPrompt = "<|begin_of_text|>";

  openaiMessages.forEach((msg) => {
    const { role, content } = msg;

    const validRoles = ["system", "user", "assistant"];
    if (!validRoles.includes(role)) {
      throw new Error(`Invalid role: ${role}`);
    }

    llamaPrompt += `\n<|start_header_id|>${role}<|end_header_id|>\n${content}\n<|eot_id|>`;
  });

  const lastMessage = openaiMessages[openaiMessages.length - 1];
  if (lastMessage?.role === "user") {
    llamaPrompt += `\n<|start_header_id|>assistant<|end_header_id|>`;
  }

  return llamaPrompt;
};

export default openaiToLlamaMessage;
