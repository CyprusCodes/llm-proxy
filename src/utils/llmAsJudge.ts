import { generateLLMResponse, OpenAIMessages } from "../index";

interface JudgeCredentials {
  apiKey?: string;
  awsConfig?: {
    accessKeyId: string;
    secretAccessKey: string;
    region: string;
  };
  openAICompatibleProviderConfig?: {
    openAICompatibleProviderKey: string;
    baseUrl: string;
  };
}

interface JudgeParams {
  messages: OpenAIMessages;
  judgePrompt: string;
  model: string;
  provider: string;
  credentials: JudgeCredentials;
  maxTokens?: number;
  temperature?: number;
}

/**
 * Uses LLM-as-a-judge to evaluate an assistant's response or trajectory
 * Following LangChain multi-turn simulation patterns
 * Returns: { passed: boolean, score: number | null, judgeResponse: string, tokenUsage }
 */
export async function llmAsJudge({
  messages,
  judgePrompt,
  model,
  credentials,
  maxTokens = 500,
  temperature = 0,
}: JudgeParams): Promise<{
  passed: boolean;
  score: number | null;
  judgeResponse: string;
  tokenUsage?: any;
}> {
  try {
    // Format the trajectory for the judge
    // Following LangChain pattern: evaluate the entire conversation trajectory
    const trajectoryText = messages
      .map((msg, idx) => {
        const role =
          msg.role === "system"
            ? "System"
            : msg.role === "user"
              ? "User"
              : msg.role === "assistant"
                ? "Assistant"
                : msg.role === "tool" || msg.role === "function"
                  ? "Tool"
                  : "Unknown";
        const content =
          typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content);
        return `Turn ${idx + 1} [${role}]: ${content}`;
      })
      .join("\n\n");

    // Build judge prompt with trajectory context
    const enhancedPrompt = judgePrompt
      .replace(
        /{{messages\[-1\]}}/g,
        messages[messages.length - 1]?.content || "",
      )
      .replace(/{{messages}}/g, trajectoryText)
      .replace(/\{\{trajectory\}\}/g, trajectoryText);

    // Build judge messages with the enhanced prompt
    const judgeMessages: OpenAIMessages = [
      {
        role: "system",
        content: enhancedPrompt,
      },
      {
        role: "user",
        content: `Please evaluate the following conversation trajectory:\n\n${trajectoryText}\n\nRespond with JSON format: {"passed": true/false, "score": 0.0-1.0, "reasoning": "your explanation"}`,
      },
    ];

    const response = await generateLLMResponse({
      messages: judgeMessages,
      model,
      max_tokens: maxTokens,
      temperature,
      credentials,
    });

    const judgeResponse = response.choices[0]?.message?.content || "";

    // Parse judge response to extract passed/score
    // Expected format: JSON like {"passed": true, "score": 0.95, "reasoning": "..."}
    let passed = false;
    let score: number | null = null;

    try {
      const jsonMatch = judgeResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        passed =
          parsed.passed === true ||
          parsed.passed === "true" ||
          parsed.passed === true;

        // Handle score - normalize to 0-1 range if needed
        if (parsed.score !== undefined && parsed.score !== null) {
          score = parseFloat(String(parsed.score));
          // If score is > 1, assume it's 0-10 scale and normalize
          if (score > 1 && score <= 10) {
            score = score / 10;
          }
          // Ensure score is between 0 and 1
          score = Math.max(0, Math.min(1, score));
        }
      } else {
        // Fallback: check for keywords and extract score
        const lowerResponse = judgeResponse.toLowerCase();
        passed =
          lowerResponse.includes("pass") ||
          lowerResponse.includes("correct") ||
          lowerResponse.includes("yes") ||
          lowerResponse.includes("true") ||
          lowerResponse.includes("satisfied") ||
          lowerResponse.includes("successful");

        // Try to extract score from text (look for 0-1 or 0-10 scale)
        const scoreMatch = judgeResponse.match(/score[:\s]+([0-9.]+)/i);
        if (scoreMatch) {
          score = parseFloat(scoreMatch[1] || "0");
          if (score > 1 && score <= 10) {
            score = score / 10;
          }
          score = Math.max(0, Math.min(1, score));
        }
      }
    } catch (parseError) {
      // If parsing fails, use keyword detection
      const lowerResponse = judgeResponse.toLowerCase();
      passed =
        lowerResponse.includes("pass") ||
        lowerResponse.includes("correct") ||
        lowerResponse.includes("yes") ||
        lowerResponse.includes("satisfied");
    }

    return {
      passed,
      score,
      judgeResponse,
      tokenUsage: response.usage,
    };
  } catch (error) {
    return {
      passed: false,
      score: null,
      judgeResponse: `Error in judge evaluation: ${error instanceof Error ? error.message : String(error)}`,
      tokenUsage: undefined,
    };
  }
}
