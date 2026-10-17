// Copied from github.com/bartdegoede/blog, assets/js/semantic/embed.js
/**
 * Reproduces model2vec's forward pass (model2vec/model.py::_encode_batch)
 * for a single query, entirely client-side:
 *
 *   ids  = tokenize(text)          # no [CLS]/[SEP]; [UNK] deleted
 *   emb  = embedding[ids]          # gather one row per token id
 *   out  = emb.mean(axis=0)        # mean-pool the UNNORMALIZED rows
 *   out  = out / (norm(out) + 1e-32)
 *
 * The rows are unnormalized on purpose: each row's magnitude carries the
 * model's zipf/SIF stopword weighting (frequent tokens like "the" have a
 * small row norm; rare/informative tokens have a large one). tokens.bin
 * ships as int8 with a per-row fp32 scale, so a row is reconstructed as
 * `tokens[id * dims + d] * scales[id]` - that reconstruction has to happen
 * before pooling, not after, or the weighting is lost.
 *
 * Zero imports by design: loaded as a plain ES module in the browser (no
 * bundler) and under `node --test`.
 */

/**
 * @param {number[]} ids - token ids (already [UNK]-filtered by the tokenizer)
 * @param {Int8Array} tokens - vocab_size * dims, C order
 * @param {Float32Array} scales - one per token row
 * @param {number} dims
 * @returns {Float32Array} length-`dims` unit vector, or all-zero if ids is empty
 */
export function embedQuery(ids, tokens, scales, dims) {
  const sum = new Float32Array(dims);

  if (ids.length > 0) {
    for (const id of ids) {
      const scale = scales[id];
      const base = id * dims;
      for (let d = 0; d < dims; d++) {
        sum[d] += tokens[base + d] * scale;
      }
    }
    for (let d = 0; d < dims; d++) {
      sum[d] /= ids.length;
    }
  }

  let normSq = 0;
  for (let d = 0; d < dims; d++) {
    normSq += sum[d] * sum[d];
  }
  const norm = Math.sqrt(normSq);

  if (norm > 0) {
    for (let d = 0; d < dims; d++) {
      sum[d] /= norm;
    }
  }
  // ids.length === 0, or a (near-impossible) exact all-zero pooled vector,
  // both fall through here as an all-zero Float32Array - never NaN.

  return sum;
}
