import {
  BedrockAnthropicContentType,
  BedrockAnthropicMessage,
} from "../../../types";

export const createTextContent = (
  text: string
): { type: BedrockAnthropicContentType.TEXT; text: string } => ({
  type: BedrockAnthropicContentType.TEXT,
  text,
});

export const createPlaceholderMessage = (
  role: "user" | "assistant"
): BedrockAnthropicMessage => ({
  role,
  content: [createTextContent(":")],
});

export const formatNonStandardMessage = (
  msg: any
): BedrockAnthropicMessage[] => [
  createPlaceholderMessage("user"),
  {
    role: "assistant",
    content: [createTextContent(msg.content ?? "")],
  },
];

export const formatStandardMessage = (msg: any): BedrockAnthropicMessage => ({
  role: msg.role,
  content: [createTextContent(msg.content ?? msg.function_call?.arguments)],
});
