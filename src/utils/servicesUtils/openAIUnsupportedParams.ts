// Some models reject tools + non-"none" reasoning_effort; retried with "none".
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function isUnsupportedReasoningEffortError(error: any): boolean {
  return error?.status === 400 && error?.param === "reasoning_effort";
}

// Some models reject a custom temperature; retried with it omitted.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function isUnsupportedTemperatureError(error: any): boolean {
  return (
    error?.status === 400 &&
    error?.param === "temperature" &&
    error?.code === "unsupported_value"
  );
}

// gpt-5.6 (sol, luna, terra, ...) only accepts default temperature (1) and
// rejects reasoning_effort + tools together
export function isGpt56ReasoningFamily(model: string): boolean {
  return model.startsWith("gpt-5.6-");
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function avoidKnownUnsupportedParams(
  requestBody: any,
  model: string,
  tools: unknown,
  reasoning_effort?: string
): void {
  if (!isGpt56ReasoningFamily(model)) {
    return;
  }
  if (requestBody.temperature !== 1) {
    delete requestBody.temperature;
  }
  if (tools && !reasoning_effort) {
    requestBody.reasoning_effort = "none";
  }
}
