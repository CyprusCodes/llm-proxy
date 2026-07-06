import { responseContainsFuturePromise } from "../utils/futurePromise";

describe("responseContainsFuturePromise", () => {
  describe("detects unfulfilled action promises", () => {
    const positives = [
      "Let me grab that data for you",
      "Let me check the database for that information.",
      "I'll fetch the latest records for you.",
      "I will look that up right away.",
      "Sure! Let me quickly search our knowledge base.",
      "I'll go ahead and retrieve your order details.",
      "I'm going to pull up your account information now.",
      "One moment while I check that.",
      "Just a second!",
      "Hold on, checking that for you.",
      "Give me a moment to find the right document.",
      "Allow me to verify that for you.",
      "Sure thing — I'll run that query now.",
      "I’ll get the info you requested." // curly apostrophe
    ];

    it.each(positives)("%s", text => {
      expect(responseContainsFuturePromise(text)).toBe(true);
    });
  });

  describe("does not flag completed answers", () => {
    const negatives = [
      "The weather in Paris is 22°C and sunny.",
      "Here is the data you requested: 42 orders were placed last week.",
      "Your order #1234 has shipped and will arrive on Tuesday.",
      "I checked the database and found 3 matching records.",
      "I found the following results for your search.",
      "You can check your order status anytime from the dashboard.",
      "Is there anything else I can help you with?"
    ];

    it.each(negatives)("%s", text => {
      expect(responseContainsFuturePromise(text)).toBe(false);
    });
  });

  it("ignores a promise phrase early in a long completed answer", () => {
    const text = `Let me check the sales figures for you. ${"Here are the detailed results of the analysis. ".repeat(
      12
    )}In summary, revenue grew 14% quarter over quarter and the top region was EMEA.`;
    expect(responseContainsFuturePromise(text)).toBe(false);
  });

  it("detects a promise at the end of a longer message", () => {
    const text = `${"Some earlier context about the conversation. ".repeat(
      10
    )}That's a great question. Let me grab that data for you.`;
    expect(responseContainsFuturePromise(text)).toBe(true);
  });

  it("handles empty and missing input", () => {
    expect(responseContainsFuturePromise("")).toBe(false);
    expect(responseContainsFuturePromise("   ")).toBe(false);
    expect(responseContainsFuturePromise(null)).toBe(false);
    expect(responseContainsFuturePromise(undefined)).toBe(false);
  });

  it("does not match partial words", () => {
    expect(
      responseContainsFuturePromise("The tomelette checks out fine.")
    ).toBe(false);
  });
});
