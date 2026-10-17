// Copied from github.com/bartdegoede/blog, assets/js/semantic/search.js
/**
 * Scoring and rank-fusion for the client-side search index.
 *
 * Zero imports by design: loaded as a plain ES module in the browser (no
 * bundler) and under `node --test`.
 */

export const RRF_K = 60;

/**
 * Cosine similarity between a single float32 query vector and every row of
 * a document matrix.
 *
 * IMPORTANT: `docs` is never dequantized. Cosine similarity is invariant to
 * a positive per-row scale, and docs.bin uses a single *global* scale for
 * every row (127), so a float32 query dotted against the raw int8 rows
 * produces the same ranking, to within 1e-6, as against dequantized floats
 * (verified in Python, and covered by the parity test in this file's
 * sibling test suite). Skipping dequantization means we never allocate a
 * second n_chunks*dims float32 buffer just to throw it away.
 *
 * @param {Float32Array} query - length `dims`
 * @param {Int8Array|Float32Array} docs - nChunks * dims, C order, raw
 *   (still-quantized) rows are fine - see note above
 * @param {number} nChunks
 * @param {number} dims
 * @returns {Float32Array} length `nChunks`
 */
export function cosineScores(query, docs, nChunks, dims) {
  let queryNormSq = 0;
  for (let d = 0; d < dims; d++) {
    queryNormSq += query[d] * query[d];
  }
  const queryNorm = Math.sqrt(queryNormSq);

  const scores = new Float32Array(nChunks);
  if (queryNorm === 0) return scores; // zero query -> all zeros, not NaN

  for (let row = 0; row < nChunks; row++) {
    const base = row * dims;
    // Accumulate in plain numbers, never in an Int8Array element - an
    // int8 accumulator silently wraps (a 300-dim dot product whose true
    // value is 3,000,000 comes back as -64 in numpy). Reading out of an
    // Int8Array promotes to Number automatically, so as long as `dot` and
    // `rowNormSq` themselves are declared as plain numbers (which they
    // are, below), this trap cannot fire here.
    let dot = 0;
    let rowNormSq = 0;
    for (let d = 0; d < dims; d++) {
      const v = docs[base + d];
      dot += query[d] * v;
      rowNormSq += v * v;
    }
    console.assert(typeof dot === "number" && Number.isFinite(dot), "dot accumulator must be a plain number");

    const rowNorm = Math.sqrt(rowNormSq);
    scores[row] = rowNorm === 0 ? 0 : dot / (queryNorm * rowNorm);
  }
  return scores;
}

/**
 * Collapse per-chunk scores to per-post scores, keeping each post's best
 * (highest-scoring) chunk.
 *
 * @param {Array} posts - length nChunks, posts[i] is the post id owning chunk i
 * @param {number[]|Float32Array} scores - length nChunks
 * @returns {Array<[post: any, score: number]>} sorted descending by score,
 *   ties broken by post id ascending (matches Python's `(-score, post_id)`
 *   sort key, so the two implementations cannot diverge)
 */
export function rollupToPosts(posts, scores) {
  const best = new Map();
  for (let i = 0; i < posts.length; i++) {
    const post = posts[i];
    const score = scores[i];
    const prev = best.get(post);
    if (prev === undefined || score > prev) {
      best.set(post, score);
    }
  }
  const entries = Array.from(best.entries());
  entries.sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
  });
  return entries;
}

/**
 * Reciprocal Rank Fusion.
 *
 * Fuse.js returns a fuzzy-match *distance* (lower is better, roughly
 * bounded 0-1 but shaped by its own internal scoring); the vector index
 * returns a *cosine similarity* (higher is better, bounded -1..1). These
 * two numbers live on incomparable scales - they are not the same unit,
 * not the same direction, and not the same distribution. Min-max
 * normalizing them into a common [0,1] range doesn't fix this: a
 * document's normalized score would then depend on the min/max of whatever
 * *other* documents happened to match that particular query, so the same
 * document could rank differently across two queries for reasons that have
 * nothing to do with the document itself. RRF sidesteps all of it by
 * discarding the scores entirely and keeping only each engine's ordinal
 * rank - ranks are always comparable, regardless of the scale or
 * distribution the underlying score came from.
 *
 * @param {Array<Array<any>>} rankedLists - each a list of doc ids, best
 *   first (rank 1). A doc absent from a list is simply not scored by it -
 *   not penalized.
 * @param {number} [k=RRF_K]
 * @returns {Array<[doc: any, score: number]>} sorted descending by fused
 *   score, ties broken lexicographically by doc id
 */
export function rrf(rankedLists, k = RRF_K) {
  const scores = new Map();
  for (const list of rankedLists) {
    for (let i = 0; i < list.length; i++) {
      const doc = list[i];
      const rank = i + 1; // 1-based
      const contribution = 1 / (k + rank);
      scores.set(doc, (scores.get(doc) || 0) + contribution);
    }
  }
  const entries = Array.from(scores.entries());
  entries.sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
  });
  return entries;
}
