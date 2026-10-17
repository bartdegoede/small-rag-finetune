// Copied from github.com/bartdegoede/blog, assets/js/semantic/tokenizer.js
/**
 * Dependency-free WordPiece tokenizer matching potion-base-8M's tokenizer.json
 * (which is BAAI/bge-base-en-v1.5's BertNormalizer + BertPreTokenizer + WordPiece).
 *
 * A model2vec model is a lookup table: tokenize -> gather one row per token id
 * from tokens.bin -> mean-pool -> L2-normalize. There is no attention, no
 * layers, nothing else to get "mostly right" - the tokenizer IS half the
 * model. Get it subtly wrong and every query vector is subtly wrong forever,
 * silently (it never throws, it just ranks badly).
 *
 * Three traps this file exists to avoid re-discovering the hard way:
 *
 * 1. NO [CLS] / [SEP]. tokenizer.json ships a TemplateProcessing
 *    post-processor that looks like it wants special tokens wrapped around
 *    every sequence. It is a decoy: model2vec.StaticModel.tokenize() calls
 *    encode_batch_fast(..., add_special_tokens=False). We must never add
 *    [CLS]/[SEP] here or our vectors won't match the ones baked into
 *    tokens.bin.
 *
 * 2. [UNK] ids are DELETED after encoding, not embedded. model2vec's encode
 *    path filters every id equal to unk_token_id out of the final id list
 *    (there is no dedicated embedding row for "unknown" in the pooled
 *    average - keeping it would dilute every vector with a meaningless
 *    constant). Consequence: WordPiece.tokenize(word) still legitimately
 *    returns ["[UNK]"] for a single out-of-vocabulary word, but
 *    WordPiece.encode(text) drops those ids, so an all-OOV query produces an
 *    empty id list, and downstream that has to become a zero vector (not an
 *    error, not a fallback embedding).
 *
 * 3. "strip_accents": null does NOT mean "leave accents alone". In
 *    HuggingFace `tokenizers`, a null strip_accents on a BertNormalizer
 *    inherits the value of `lowercase`, which is true here. So accents ARE
 *    stripped: "café" -> "cafe", "naïve" -> "naive". Treating null as "no
 *    stripping" (the intuitive but wrong reading) would silently mismatch
 *    any accented query against an unaccented vocab entry.
 *
 * Zero imports by design: this file is loaded as a plain ES module both in
 * the browser (no bundler) and under `node --test` for the unit tests.
 */

// CJK ranges BERT's handle_chinese_chars step wraps in spaces, expressed as
// [start, end] inclusive code point pairs.
const CJK_RANGES = [
  [0x4e00, 0x9fff],
  [0x3400, 0x4dbf],
  [0x20000, 0x2a6df],
  [0x2a700, 0x2b73f],
  [0x2b740, 0x2b81f],
  [0x2b820, 0x2ceaf],
  [0xf900, 0xfaff],
  [0x2f800, 0x2fa1f],
];

function isCjkCodePoint(cp) {
  for (const [start, end] of CJK_RANGES) {
    if (cp >= start && cp <= end) return true;
  }
  return false;
}

// A "control character" for BERT's clean_text step: any Unicode category C*
// (Cc, Cf, Co, Cs, Cn) that is not one of \t, \n, \r (those are handled
// separately and become spaces, not deleted).
function isControlCodePoint(cp) {
  if (cp === 0x09 || cp === 0x0a || cp === 0x0d) return false;
  const ch = String.fromCodePoint(cp);
  return /\p{C}/u.test(ch);
}

/**
 * BertNormalizer: clean_text, handle_chinese_chars, lowercase,
 * strip_accents (forced true because strip_accents:null inherits lowercase).
 */
export function normalize(text) {
  let out = [];
  for (const ch of text) {
    const cp = ch.codePointAt(0);

    // clean_text: drop NUL and replacement char, drop other control chars
    // outright, turn \t \n \r into a single space each.
    if (cp === 0x0000 || cp === 0xfffd) continue;
    if (cp === 0x09 || cp === 0x0a || cp === 0x0d) {
      out.push(" ");
      continue;
    }
    if (isControlCodePoint(cp)) continue;

    // handle_chinese_chars: pad CJK code points with spaces on both sides.
    if (isCjkCodePoint(cp)) {
      out.push(" ", ch, " ");
      continue;
    }

    out.push(ch);
  }
  let result = out.join("");

  // lowercase
  result = result.toLowerCase();

  // strip_accents (forced on): decompose then drop nonspacing marks, do not
  // re-compose (NFC).
  result = result.normalize("NFD").replace(/\p{Mn}/gu, "");

  return result;
}

const PUNCT_OR_SYMBOL = /\p{P}|\p{S}/u;

/**
 * BertPreTokenizer: split on whitespace runs, then split so every
 * punctuation/symbol character stands alone as its own token.
 */
export function preTokenize(text) {
  const tokens = [];
  for (const piece of text.split(/\s+/)) {
    if (piece === "") continue;
    let current = "";
    for (const ch of piece) {
      if (PUNCT_OR_SYMBOL.test(ch)) {
        if (current !== "") {
          tokens.push(current);
          current = "";
        }
        tokens.push(ch);
      } else {
        current += ch;
      }
    }
    if (current !== "") tokens.push(current);
  }
  return tokens;
}

export class WordPiece {
  /**
   * @param {string[]} vocabArray - id-indexed token strings.
   * @param {{unkToken: string, prefix: string, maxCharsPerWord: number}} opts
   */
  constructor(vocabArray, { unkToken, prefix, maxCharsPerWord }) {
    this.unkToken = unkToken;
    this.prefix = prefix;
    this.maxCharsPerWord = maxCharsPerWord;
    this.vocab = vocabArray;
    // A Map, not a plain object: 29,528 entries, and object literals collide
    // with inherited keys like "__proto__".
    this.vocabIndex = new Map();
    for (let i = 0; i < vocabArray.length; i++) {
      this.vocabIndex.set(vocabArray[i], i);
    }
    this.unkId = this.vocabIndex.has(unkToken) ? this.vocabIndex.get(unkToken) : -1;
  }

  /**
   * Greedy longest-match-first WordPiece tokenization of a single
   * whitespace/punctuation-delimited word.
   * @returns {string[]}
   */
  tokenize(word) {
    // Count by code point, matching how max_input_chars_per_word is applied
    // upstream in HuggingFace's Rust implementation (Unicode scalar values).
    const chars = Array.from(word);
    if (chars.length > this.maxCharsPerWord) {
      return [this.unkToken];
    }

    const subTokens = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let matched = null;
      while (end > start) {
        let candidate = chars.slice(start, end).join("");
        if (start > 0) candidate = this.prefix + candidate;
        if (this.vocabIndex.has(candidate)) {
          matched = candidate;
          break;
        }
        end -= 1;
      }
      if (matched === null) {
        return [this.unkToken];
      }
      subTokens.push(matched);
      start = end;
    }
    return subTokens;
  }

  /**
   * normalize -> preTokenize -> tokenize each piece -> map to ids -> drop
   * unk ids (trap #2: model2vec deletes unk ids rather than embedding them).
   * No [CLS]/[SEP] (trap #1).
   * @returns {number[]}
   */
  encode(text) {
    const normalized = normalize(text);
    const words = preTokenize(normalized);
    const ids = [];
    for (const word of words) {
      const pieces = this.tokenize(word);
      for (const piece of pieces) {
        const id = this.vocabIndex.get(piece);
        if (id === undefined) continue;
        if (id === this.unkId) continue;
        ids.push(id);
      }
    }
    return ids;
  }
}
