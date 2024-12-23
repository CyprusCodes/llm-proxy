export const validateMessages = (messages: any[]): void => {
  if (!messages || !messages.length) {
    throw new Error("Messages array cannot be empty for Anthropic.");
  }

  if (messages[0].role !== "system") {
    throw new Error(
      "The first message must have a role of 'system' for Anthropic."
    );
  }
};
