import { OpenAIMessages } from "llm-proxy";

const openaiToLlamaMessage = (openaiMessages: OpenAIMessages): string => {
  let llamaPrompt = "<|begin_of_text|>";

  openaiMessages.forEach((msg) => {
    const { role, content } = msg;

    const validRoles = ["system", "user", "assistant"];
    if (!validRoles.includes(role)) {
      throw new Error(`Invalid role: ${role}`);
    }

    llamaPrompt += `<|start_header_id|>${role}<|end_header_id|>\n${content}<|eot_id|>\n`;
  });

  return llamaPrompt;
};

export default openaiToLlamaMessage;
