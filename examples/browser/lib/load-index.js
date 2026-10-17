// Loads the search index `sss-eval build` writes (manifest.json plus five
// content-hashed files). Adapted from github.com/bartdegoede/blog,
// assets/js/semantic/index.js.
import { WordPiece } from "./tokenizer.js";

// potion-base-8M's tokenizer: BERT WordPiece defaults.
const WORDPIECE_OPTS = { unkToken: "[UNK]", prefix: "##", maxCharsPerWord: 100 };

function check(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`search index corrupt: ${label} is ${actual}, expected ${expected}`);
  }
}

/**
 * @param {string} base - URL of the directory holding manifest.json
 * @returns {Promise<object>} {manifest, docs, tokens, scales, chunks, wp}, what retrieve.js expects
 */
export async function loadIndex(base) {
  const res = await fetch(`${base}/manifest.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`failed to fetch ${base}/manifest.json: HTTP ${res.status}`);
  const manifest = await res.json();
  const f = manifest.files;
  const get = (name) => fetch(`${base}/${name}`);
  const [docs, tokens, scales, vocab, chunks] = await Promise.all([
    get(f.docs).then((r) => r.arrayBuffer()),
    get(f.tokens).then((r) => r.arrayBuffer()),
    get(f.scales).then((r) => r.arrayBuffer()),
    get(f.vocab).then((r) => r.json()),
    get(f.chunks).then((r) => r.json()),
  ]);
  const { n_chunks, dims, vocab_size } = manifest;
  check(docs.byteLength, n_chunks * dims, "docs.bin size");
  check(tokens.byteLength, vocab_size * dims, "tokens.bin size");
  check(scales.byteLength, vocab_size * 4, "scales.bin size");
  check(vocab.length, vocab_size, "vocab length");
  check(chunks.length, n_chunks, "chunks length");
  return {
    manifest,
    docs: new Int8Array(docs),
    tokens: new Int8Array(tokens),
    scales: new Float32Array(scales),
    chunks,
    wp: new WordPiece(vocab, WORDPIECE_OPTS),
  };
}
