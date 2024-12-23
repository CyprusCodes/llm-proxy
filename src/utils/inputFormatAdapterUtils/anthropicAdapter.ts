import { BedrockAnthropicMessage } from "../../types";
import { validateMessages } from "./anthropicInputFormatAdapterUtils/validators";
import { processMessages } from "./anthropicInputFormatAdapterUtils/messageProcessor";

export const adaptAnthropicMessages = (
  messages: any
): {
  adaptedMessages: BedrockAnthropicMessage[];
  systemPrompt: string;
} => {
  validateMessages(messages);
  return processMessages(messages);
};
