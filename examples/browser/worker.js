// Runs the model off the main thread: loads it once, then streams answers.
// Messages in:  {type: "load", model, device} | {type: "generate", messages}
// Messages out: {type: "progress", file, loaded, total} | {type: "ready", device}
//               | {type: "token", text} | {type: "done", text} | {type: "error", message}
import { MAX_NEW_TOKENS, MAX_SENTENCES } from "./lib/prompt.js";
import { clampSentences, formatAnswer, stripLeadingMarkers } from "./lib/display.js";

// A full URL, so the page works without a bundler. Pinned: the model was
// tested with this version.
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/+esm";

let tf = null;
let generator = null;

const post = (msg) => self.postMessage(msg);

async function load({ model, device }) {
  tf = await import(TRANSFORMERS_URL);
  generator = await tf.pipeline("text-generation", model, {
    dtype: "q4f16",
    device,
    progress_callback: (e) => {
      if (e.status === "progress") post({ type: "progress", file: e.file, loaded: e.loaded, total: e.total });
    },
  });
  post({ type: "ready", device });
}

async function generate({ messages }) {
  const stopper = new tf.InterruptableStoppingCriteria();
  let text = "";
  const streamer = new tf.TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      text += chunk;
      // The model sometimes runs past two sentences: show at most two, and
      // stop generating once they're complete.
      const clamped = clampSentences(stripLeadingMarkers(text).trimStart(), MAX_SENTENCES, false);
      post({ type: "token", text: clamped.text });
      if (clamped.complete) stopper.interrupt();
    },
  });
  await generator(messages, {
    max_new_tokens: MAX_NEW_TOKENS,
    do_sample: false,
    // No repetition penalty: transformers.js applies it to the prompt too,
    // which pushes the model away from the passages' own words.
    repetition_penalty: 1.0,
    tokenizer_encode_kwargs: { enable_thinking: false },
    streamer,
    stopping_criteria: stopper,
  });
  post({ type: "done", text: formatAnswer(text, MAX_SENTENCES) });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data);
    else if (data.type === "generate") await generate(data);
  } catch (e) {
    post({ type: "error", message: String(e?.message ?? e) });
  }
};
