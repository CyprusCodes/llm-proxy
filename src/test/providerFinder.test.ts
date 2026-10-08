import ProviderFinder from "../middleware/ProviderFinder";
import { Providers } from "../types";

const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/";

describe("ProviderFinder.getProvider — Gemini routing", () => {
  describe("routes gemini-3.x+ to GEMINI (strips sampling params)", () => {
    const newGenModels = [
      "gemini-3.6-flash",
      "gemini-3.7-flash",
      "gemini-3.8-flash",
      "gemini-3.8-pro",
      "gemini-4.0-flash",
    ];

    it.each(newGenModels)("%s", model => {
      expect(ProviderFinder.getProvider(model, GEMINI_BASE_URL)).toBe(Providers.GEMINI);
    });
  });

  describe("routes older gemini models to OPENAI_COMPATIBLE_PROVIDER (keeps temperature)", () => {
    const oldGenModels = [
      "gemini-1.5-pro",
      "gemini-1.5-flash",
      "gemini-2.0-flash",
      "gemini-2.5-pro",
      "gemini-2.5-flash",
    ];

    it.each(oldGenModels)("%s", model => {
      expect(ProviderFinder.getProvider(model, GEMINI_BASE_URL)).toBe(Providers.OPENAI_COMPATIBLE_PROVIDER);
    });
  });
});
