// Run local ONNX q4f16 models over the blog's eval prompts with transformers.js,
// the runtime the browser uses (CPU here; same decode settings as the blog's
// scripts/ask_eval/run_local.mjs). One JSON row per prompt on stdout, in the
// format the blog's judge.py reads ({arm, id, summary}).
//   node js/generate.mjs <model-dir> <prompts.jsonl> [n] > runs/eval/<arm>.jsonl
import { readFileSync } from "node:fs";
import path from "node:path";
import { pipeline, env } from "@huggingface/transformers";

const [dir, promptsPath, n] = process.argv.slice(2);
const arm = path.basename(dir);
env.allowRemoteModels = false;
env.localModelPath = path.dirname(path.resolve(dir));
const gen = await pipeline("text-generation", arm, { dtype: "q4f16", device: "cpu" });
const rows = readFileSync(promptsPath, "utf8").trim().split("\n").map(JSON.parse);
for (const r of n ? rows.slice(0, Number(n)) : rows) {
  const t0 = performance.now();
  const out = await gen(r.messages, {
    max_new_tokens: 120, do_sample: false, repetition_penalty: 1.15,
    tokenizer_encode_kwargs: { enable_thinking: false },
  });
  // Same clean-up as run_local.mjs: drop an empty Qwen3 think block.
  const summary = out[0].generated_text.at(-1).content.replace(/^\s*<think>\s*<\/think>\s*/, "").trim();
  console.log(JSON.stringify({ arm, id: r.id, summary, ms: Math.round(performance.now() - t0) }));
}
