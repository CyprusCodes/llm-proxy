// GENERAL
export enum Providers {
  OPENAI = "OpenAI",
  ANTHROPIC_BEDROCK = "AnthropicBedrock",
  LLAMA_3_1_BEDROCK = "Llama3.1",
}

export type OpenAIMessagesRoles =
  | "system"
  | "user"
  | "assistant"
  | "tool"
  | "function";

export interface OpenAIStreamResponse {
  id: string; // some string id for the message
  object: string; // chat.completion.chunk
  created: number; // 1732633291
  model: string; // "gpt-4o-2024-08-06"
  system_fingerprint: string; // "fp_7f6be3efb0";
  choices: {
    index: number;
    delta: {
      content?: string;
      function_call?: {
        id?: string;
        name?: string;
        arguments?: string;
      };
    };
  };
  logprobs: string | null;
  finish_reason: string | null;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    prompt_tokens_details: { cached_tokens: number; audio_tokens: number };
    completion_tokens_details: {
      reasoning_tokens: number;
      audio_tokens: number;
      accepted_prediction_tokens: number;
      rejected_prediction_tokens: number;
    };
  };
}

export type OpenAISystemMessage = {
  role: "system";
  content: string;
};

export type OpenAIUserMessage = {
  role: "user";
  content: string;
};

export type OpenAIAssistantMessage = {
  role: "assistant";
  content: string;
};

export type OpenAIToolMessage = {
  role: "tool";
  content: string;
  tool_call_id: string;
};

export type OpenAIFunctionMessage = {
  role: "function";
  content: string;
  name: string;
};

export type OpenAIMessage =
  | OpenAISystemMessage
  | OpenAIUserMessage
  | OpenAIAssistantMessage
  | OpenAIToolMessage
  | OpenAIFunctionMessage;

export type OpenAIMessages = OpenAIMessage[];

export enum OpenAISupportedLLMs {
  GPT_4_O_LATEST = "chatgpt-4o-latest", // points to the latest version of gpt-4o
  GPT_4_O = "gpt-4o",
  GPT_4_O_2024_11_20 = "gpt-4o-2024-11-20",
  GPT_4_O_2024_08_06 = "gpt-4o-2024-08-06",
  GPT_4_O_2024_05_13 = "gpt-4o-2024-05-13",
  GPT_4_O_REALTIME_PREVIEW = "gpt-4o-realtime-preview",
  GPT_4_O_REALTIME_PREVIEW_2024_10_01 = "gpt-4o-realtime-preview-2024-10-01",
  GPT_4_O_AUDIO_PREVIEW = "gpt-4o-audio-preview",
  GPT_4_O_AUDIO_PREVIEW_2024_10_01 = "gpt-4o-audio-preview-2024-10-01",
  GPT_4_O_MINI = "gpt-4o-mini",
  GPT_4_O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18",
  O1_PREVIEW = "o1-preview",
  O1_PREVIEW_2024_09_12 = "o1-preview-2024-09-12",
  O1_MINI = "o1-mini",
  O1_MINI_2024_09_12 = "o1-mini-2024-09-12",
  GPT_4_TURBO = "gpt-4-turbo",
  GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09",
  GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview", // same as gpt-4-turbo-2024-04-09
  GPT_4_0125_PREVIEW = "gpt-4-0125-preview",
  GPT_4_1106_PREVIEW = "gpt-4-1106-preview",
  GPT_4_VISION_PREVIEW = "gpt-4-vision-preview",
  GPT_4 = "gpt-4",
  GPT_4_0314 = "gpt-4-0314",
  GPT_4_0613 = "gpt-4-0613",
  GPT_4_32K = "gpt-4-32k",
  GPT_4_32K_0314 = "gpt-4-32k-0314",
  GPT_4_32K_0613 = "gpt-4-32k-0613",
  GPT_3_5_TURBO = "gpt-3.5-turbo", // same as gpt-3.5-turbo-0125
  GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k",
  GPT_3_5_TURBO_0301 = "gpt-3.5-turbo-0301",
  GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613",
  GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106",
  GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125",
  GPT_3_5_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613",
}

export interface OpenAIChoices {
  index: number;
  message: any; // TODO: update this guys types as well -- this is an important one --
  logprobs: any; // TODO: define logprobs type
  finish_reason: string;
}

export interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details: any; // TODO: define type
  completion_tokens_details: any; // TODO: define type
}

export interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      function_call?: {
        name: string;
        arguments: any; // TODO: i guess this is supposed to be string
      };
    };
    logprobs: null | object;
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    prompt_tokens_details?: { cached_tokens: number };
    completion_tokens_details?: { reasoning_tokens: number };
  };
  system_fingerprint: string;
}

// AWS BEDROCK

// AWS BEDROCK ANTHROPIC

// https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
export enum BedrockAnthropicSupportedLLMs {
  // Claude 3 Family
  CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0",
  CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0",
  CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0",

  // Claude 3.5 Family
  CLAUDE_3_5_SONNET_V1 = "anthropic.claude-3-5-sonnet-20240620-v1:0",
  CLAUDE_3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0",
  CLAUDE_3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0",

  // Claude 2 Family
  CLAUDE_V2_1 = "anthropic.claude-v2:1",
  CLAUDE_V2 = "anthropic.claude-v2",
  CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1",

  // US Region Specific
  US_CLAUDE_3_HAIKU = "us.anthropic.claude-3-haiku-20240307-v1:0",
  US_CLAUDE_3_5_HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
  US_CLAUDE_3_5_SONNET = "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
  US_CALUDE_3_5_SONNET_V2 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}

export enum BedrockAnthropicContentType {
  TEXT = "text",
  IMAGE = "image",
  TOOL_USE = "tool_use",
  TOOL_RESULT = "tool_result",
}

export type BedrockAnthropicMessageRole = "user" | "assistant";

export interface BedrockAnthropicToolUseContent {
  type: BedrockAnthropicContentType.TOOL_USE;
  id: string;
  name: string;
  input: any;
}

export interface BedrockAnthropicTextContent {
  type: BedrockAnthropicContentType.TEXT;
  text: string;
}

interface BedrockAnthropicImageContent {
  type: BedrockAnthropicContentType.IMAGE;
  source: {
    type: string;
    media_type: string;
    data: string;
  };
}
export interface BedrockAnthropicToolResultContent {
  type: BedrockAnthropicContentType.TOOL_RESULT;
  content: string;
}

export type BedrockAnthropicContent =
  | BedrockAnthropicToolUseContent
  | BedrockAnthropicToolResultContent
  | BedrockAnthropicTextContent
  | BedrockAnthropicImageContent;

export interface BedrockAnthropicMessage {
  role: BedrockAnthropicMessageRole;
  content: BedrockAnthropicContent[];
}

export interface BedrockAnthropicFunctionCall {
  id: string;
  name: string;
  arguments: string;
}

export type BedrockAnthropicMessages = BedrockAnthropicMessage[];

export interface BedrockAnthropicOptions {
  outputTokenLength: number;
  temperature: number;
  systemPrompt: string;
  messages: BedrockAnthropicMessages;
  tools: any;
}

export interface BedrockAnthropicUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface BedrockAnthropicResponse {
  id: string;
  type: "message";
  role: BedrockAnthropicMessageRole;
  model: string;
  content: BedrockAnthropicContent[];
  stop_reason: string;
  stop_sequence: string;
  usage: BedrockAnthropicUsage;
}

export interface BedrockAnthropicMessageChunk {
  id: string;
  type: "message";
  model: string;
  role: BedrockAnthropicMessageRole;
  content: BedrockAnthropicContent[];
  stop_reason: string | null;
  stop_sequence: string | null;
  usage: BedrockAnthropicUsage;
}

export interface BedrockAnthropicContentBlock {
  type: string;
  text: string;
  name?: string;
  id?: string;
  partial_json?: string;
}

export interface BedrockAnthropicMetrics {
  inputTokenCount: number;
  outputTokenCount: number;
  invocationLatency: number;
  firstByteLatency: number;
}

export type BedrockAnthropicParsedChunk = {
  type: string;
  message?: BedrockAnthropicMessageChunk;
  content_block?: BedrockAnthropicContentBlock;
  delta?: BedrockAnthropicContentBlock;
  "amazon-bedrock-invocationMetrics"?: BedrockAnthropicMetrics;
};

// LLAMA 3.1

export enum Llama3_1SupportedLLMs {
  LLAMA_3_1_8B = "meta.llama3-1-8b-instruct-v1:0",
  LLAMA_3_1_70B_2 = "meta.llama3-1-70b-instruct-v1:0",
  LLAMA_3_1_405B = "meta.llama3-1-405b-instruct-v1:0",
  US_CROSS_REGION_LLAMA_3_1_8B = "us.meta.llama3-1-8b-instruct-v1:0",
  US_CROSS_REGION_LLAMA_3_1_70B_2 = "us.meta.llama3-1-70b-instruct-v1:0",
  US_CROSS_REGION_LLAMA_3_1_405B = "us.meta.llama3-1-405b-instruct-v1:0",
}

// GENERAL
export type Messages = OpenAIMessages | BedrockAnthropicMessages;
export type LLMResponse = OpenAIResponse | BedrockAnthropicResponse;
