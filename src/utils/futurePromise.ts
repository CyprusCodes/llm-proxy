/**
 * Detects "future promise" responses: the model says it is about to take an
 * action ("Let me grab that data for you", "I'll check the weather") but ends
 * its turn without calling a tool.
 *
 * Detection is anchored to the tail of the message. A promise phrase early in
 * a long response is usually followed by the actual answer, so only the final
 * portion of the text counts.
 */

// Only the tail of the message is scanned — a promise phrase followed by this
// much content means the model most likely delivered the answer already.
const TAIL_WINDOW_CHARS = 400;

const AGENT_PREFIXES = [
  "let me",
  "i'll",
  "i will",
  "i'm going to",
  "i am going to",
  "i'm about to",
  "i am about to",
  "allow me to",
  "i'm now",
  "i am now",
  "now i'll",
  "now i will"
];

const ACTION_VERBS = [
  "check",
  "look",
  "search",
  "find",
  "retrieve",
  "fetch",
  "get",
  "grab",
  "pull",
  "load",
  "query",
  "run",
  "call",
  "execute",
  "verify",
  "inspect",
  "access",
  "open",
  "read",
  "gather",
  "browse",
  "scan",
  "analyze",
  "use",
  "contact",
  "request",
  "prepare",
  "generate",
  "create",
  "compile",
  "collect",
  "process",
  "update",
  "start",
  "launch"
];

// Phrases that on their own signal the model believes work is still pending.
const WAITING_PHRASES = [
  "one moment",
  "just a moment",
  "just a second",
  "just a sec",
  "hold on",
  "give me a moment",
  "give me a second",
  "give me a sec",
  "give me a minute",
  "bear with me",
  "please wait",
  "hang on",
  "hang tight"
];

function escapeForRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const prefixAlternation = AGENT_PREFIXES.map(escapeForRegex).join("|");
const verbAlternation = ACTION_VERBS.join("|");

// Prefix, then up to three filler words ("quickly", "go ahead and"), then an
// action verb. Requires word boundaries so "will" doesn't match "willing".
const AGENT_PROMISE_REGEX = new RegExp(
  `\\b(?:${prefixAlternation})(?:\\s+\\w+){0,3}?\\s+(?:${verbAlternation})\\b`,
  "i"
);

const WAITING_PHRASE_REGEX = new RegExp(
  `\\b(?:${WAITING_PHRASES.map(escapeForRegex).join("|")})\\b`,
  "i"
);

export function responseContainsFuturePromise(
  text: string | null | undefined
): boolean {
  if (!text) {
    return false;
  }

  // Normalize curly apostrophes so "I’ll" matches "i'll"
  const normalized = text.replace(/[‘’]/g, "'").trim();
  if (!normalized) {
    return false;
  }

  const tail = normalized.slice(-TAIL_WINDOW_CHARS);

  return AGENT_PROMISE_REGEX.test(tail) || WAITING_PHRASE_REGEX.test(tail);
}

export default responseContainsFuturePromise;
