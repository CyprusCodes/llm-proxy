import { BedrockAnthropicMessage } from "../../../types";
import {
  formatNonStandardMessage,
  formatStandardMessage,
} from "./messageFormatters";
import { ensureAlternatingRoles } from "./roleHandler";

export const processMessages = (
  messages: any[]
): {
  systemPrompt: string;
  adaptedMessages: BedrockAnthropicMessage[];
} => {
  const [firstMessage, ...restMessages] = messages;
  const systemPrompt = firstMessage.content ?? "";
  const adaptedMessages: BedrockAnthropicMessage[] = [];

  restMessages.forEach((msg) => {
    if (msg.role !== "user" && msg.role !== "assistant") {
      adaptedMessages.push(...formatNonStandardMessage(msg));
    } else {
      adaptedMessages.push(formatStandardMessage(msg));
    }
  });

  return {
    systemPrompt,
    adaptedMessages: ensureAlternatingRoles(adaptedMessages),
  };
};
