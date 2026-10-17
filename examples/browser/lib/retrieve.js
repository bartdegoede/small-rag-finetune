// Copied from github.com/bartdegoede/blog, assets/js/ask/retrieve.js
/**
 * Chunk-level retrieval for /ask/.
 *
 * The homepage search rolls chunks up to posts; /ask/ needs the passages
 * themselves, so this keeps chunk granularity. Imports only the sibling semantic-search modules (no third-party code).
 * Runs in the browser, the Web Worker, and under `node --test` unchanged.
 */

import { embedQuery } from "./embed.js";
import { cosineScores } from "./search.js";

/**
 * Cosine score of `query` against every chunk. `idx` is what
 * `assets/js/semantic/index.js` `loadIndex()` returns (or the eval's
 * on-disk equivalent): {manifest, docs, tokens, scales, chunks, wp}.
 * @returns {Float32Array} all zeros for an empty or fully out-of-vocabulary query
 */
export function scoreQuery(idx, query) {
  const ids = idx.wp.encode(query);
  const { dims, n_chunks } = idx.manifest;
  if (ids.length === 0) return new Float32Array(n_chunks);
  return cosineScores(embedQuery(ids, idx.tokens, idx.scales, dims), idx.docs, n_chunks, dims);
}

/**
 * @param {Float32Array|number[]} scores - one cosine score per chunk
 * @param {object[]} chunks - chunks.json records, same order as scores
 * @param {number} k
 * @returns {Array<{index: number, score: number, chunk: object}>} best first;
 *   ties broken by chunk index ascending so results are deterministic
 */
export function topChunks(scores, chunks, k) {
  if (scores.length !== chunks.length) {
    throw new Error(`topChunks: scores length ${scores.length} != chunks length ${chunks.length}`);
  }
  const order = Array.from(scores, (score, index) => ({ index, score }));
  order.sort((a, b) => (b.score !== a.score ? b.score - a.score : a.index - b.index));
  return order.slice(0, k).map(({ index, score }) => ({ index, score, chunk: chunks[index] }));
}
