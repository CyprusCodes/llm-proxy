import { BedrockAnthropicMessage } from "../../../types";
import { createPlaceholderMessage } from "./messageFormatters";

export const ensureAlternatingRoles = (
  messages: BedrockAnthropicMessage[] = []
): BedrockAnthropicMessage[] => {
  const result: BedrockAnthropicMessage[] = [...messages];

  for (let i = 0; i < result.length - 1; i++) {
    if (result[i]!.role === result[i + 1]!.role) {
      result.splice(
        i + 1,
        0,
        createPlaceholderMessage(
          result[i]!.role === "user" ? "assistant" : "user"
        )
      );
      i++;
    }
  }

  return result;
};
