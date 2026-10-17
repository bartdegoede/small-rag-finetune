// Copied from github.com/bartdegoede/blog, assets/js/ask/prompt.js
/**
 * Prompt assembly for /ask/. Shared verbatim by the browser worker (Part 2)
 * and the eval harness, so the eval measures exactly what ships.
 * Zero imports.
 */

export const TOP_K = 4;
export const MAX_NEW_TOKENS = 120;
// Greedy decoding without a repetition penalty makes the smallest models loop
// (measured in the ask eval: most smollm2-135m replies ran to MAX_NEW_TOKENS).
// This is the default; a model can override it with `repetitionPenalty` in
// models.js. transformers.js also penalises prompt tokens, so the penalty
// pushes a model away from the passages' own words: Qwen3-0.6B and LFM2.5-350M
// don't loop without it and answer more faithfully (results doc §10).
export const REPETITION_PENALTY = 1.15;

// Best-chunk cosine below this: don't call the model at all. Chosen in the
// eval (results doc §3): blocks clearly off-topic questions only; near-miss
// questions score like answerable ones and still reach the model.
export const GATE_THRESHOLD = 0.45;
// The system prompt asks for at most 2 sentences; the worker stops there.
export const MAX_SENTENCES = 2;

/** @param {Array<{score: number}>} top - topChunks() output, best first */
export function passesGate(top) {
  return top.length > 0 && top[0].score >= GATE_THRESHOLD;
}
// ~1,200 tokens at ~4 chars/token. A safety net: unreachable with today's
// chunker at TOP_K=4; guards future chunk-size/TOP_K changes and keeps WASM
// prefill bounded.
export const MAX_PASSAGE_CHARS = 4800;
// Never add a passage with less body room than this (avoids mid-word fragments).
export const MIN_PASSAGE_CHARS = 100;

/** The system prompt asking for at most `n` sentences (the eval varies n; the page uses MAX_SENTENCES). */
export function systemPrompt(n) {
  return (
    "You answer questions about a personal technical blog. " +
    "Use only the numbered passages provided. " +
    `Answer in at most ${n} sentences. ` +
    "If the passages do not answer the question, say that the blog does not seem to cover it."
  );
}

export const SYSTEM_PROMPT = systemPrompt(MAX_SENTENCES);

function label(p, i) {
  return `[${i + 1}] ${p.title}${p.heading ? ` — ${p.heading}` : ""}`;
}

/**
 * @param {string} question
 * @param {Array<{title: string, heading: string|null, snippet: string}>} passages - best first
 * @param {{maxSentences?: number}} [opts] - sentence limit in the system prompt (eval only)
 * @returns {Array<{role: "system"|"user", content: string}>}
 */
export function buildMessages(question, passages, { maxSentences = MAX_SENTENCES } = {}) {
  const parts = [];
  let used = 0;
  for (let i = 0; i < passages.length; i++) {
    const sep = parts.length ? 2 : 0;
    const head = label(passages[i], i);
    const room = MAX_PASSAGE_CHARS - used - sep - head.length - 1;
    if (room < MIN_PASSAGE_CHARS) break;
    const body = passages[i].snippet.slice(0, room);
    parts.push(`${head}\n${body}`);
    used += sep + head.length + 1 + body.length;
  }
  const user = `Passages:\n\n${parts.join("\n\n")}\n\nQuestion: ${question}`;
  return [
    { role: "system", content: systemPrompt(maxSentences) },
    { role: "user", content: user },
  ];
}

/**
 * For chat templates without a system role: prepend the system prompt to the
 * first user message. Assumes the message after the system message is a user turn.
 */
export function foldSystem(messages) {
  const [sys, first, ...rest] = messages;
  if (!sys || sys.role !== "system" || !first) return messages;
  return [{ role: "user", content: `${sys.content}\n\n${first.content}` }, ...rest];
}
