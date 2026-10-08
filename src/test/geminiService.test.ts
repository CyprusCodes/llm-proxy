import GeminiService from "../services/GeminiService";
import OpenAICompatibleService from "../services/OpenAICompatibleService";

const mockCreate = jest.fn();

jest.mock("openai", () => {
  return jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: mockCreate,
      },
    },
  }));
});

const MESSAGES = [{ role: "user" as const, content: "Hello" }];
const MODEL = "gemini-3.8-flash";
const BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/";
const API_KEY = "test-key";

beforeEach(() => {
  mockCreate.mockReset();
  mockCreate.mockResolvedValue({
    id: "test",
    object: "chat.completion",
    created: 0,
    model: MODEL,
    choices: [],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    system_fingerprint: "",
  });
});

describe("GeminiService", () => {
  it("does not send temperature to the API", async () => {
    const service = new GeminiService(API_KEY, BASE_URL);
    await service.generateCompletion({
      messages: MESSAGES,
      model: MODEL,
      temperature: 0.7,
    });

    const calledWith = mockCreate.mock.calls[0][0];
    expect(calledWith).not.toHaveProperty("temperature");
  });

  it("still sends other params like max_tokens", async () => {
    const service = new GeminiService(API_KEY, BASE_URL);
    await service.generateCompletion({
      messages: MESSAGES,
      model: MODEL,
      temperature: 0.7,
      max_tokens: 512,
    });

    const calledWith = mockCreate.mock.calls[0][0];
    expect(calledWith).not.toHaveProperty("temperature");
    expect(calledWith.max_tokens).toBe(512);
  });
});

describe("OpenAICompatibleService (older Gemini models)", () => {
  it("sends temperature to the API", async () => {
    const service = new OpenAICompatibleService(API_KEY, BASE_URL);
    await service.generateCompletion({
      messages: MESSAGES,
      model: "gemini-2.0-flash",
      temperature: 0.7,
    });

    const calledWith = mockCreate.mock.calls[0][0];
    expect(calledWith.temperature).toBe(0.7);
  });

  it("omits temperature when not provided", async () => {
    const service = new OpenAICompatibleService(API_KEY, BASE_URL);
    await service.generateCompletion({
      messages: MESSAGES,
      model: "gemini-2.0-flash",
    });

    const calledWith = mockCreate.mock.calls[0][0];
    expect(calledWith).not.toHaveProperty("temperature");
  });
});
