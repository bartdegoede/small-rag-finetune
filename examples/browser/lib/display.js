// Copied from github.com/bartdegoede/blog, assets/js/ask/display.js
/**
 * Post-processing for generated answers. The eval found small models emit
 * leading passage markers ("[3] ...") and run past the requested two
 * sentences; the worker uses clampSentences() to stop generating early and
 * the page shows formatAnswer(). Zero imports.
 */

const ABBREVIATION = /(?:\b(?:e\.g|i\.e|etc|vs|cf|approx|Mr|Mrs|Dr)\.)$/i;
const LIST_NUMBER = /(?:^|\s)\d+\.$/;

export function stripLeadingMarkers(text) {
  return text.replace(/^\s*(?:\[\d+\]\s*)+/, "");
}

/**
 * Cut `text` after its `n`th sentence.
 * @param {boolean} final - true when generation has finished: end of text then
 *   counts as a sentence end. While streaming, a "." with nothing after it may
 *   still become "3.5", so it doesn't count yet.
 * @returns {{text: string, complete: boolean}} complete = n sentences found
 */
export function clampSentences(text, n, final) {
  const re = final ? /[.!?](?=\s|$)/g : /[.!?](?=\s)/g;
  let count = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    const upto = text.slice(0, m.index + 1);
    if (ABBREVIATION.test(upto) || LIST_NUMBER.test(upto)) continue;
    count++;
    if (count === n) return { text: upto, complete: true };
  }
  return { text, complete: false };
}

export function formatAnswer(raw, n) {
  return clampSentences(stripLeadingMarkers(raw).trim(), n, true).text;
}

const FOOTNOTE_REF = /\s*\[\^[^\]]+\]/g;
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Display-only cleanup of an index snippet (the prompt gets the raw text).
 * The index keeps markdown footnote refs ("filters[^medium_bloom]") and
 * prepends the section heading to a section's first chunk ("Closing thoughts
 * Closing thoughts There are…"); drop both and collapse whitespace.
 * @param {string} snippet
 * @param {string|null} heading
 */
export function cleanSnippet(snippet, heading) {
  let text = snippet.replace(FOOTNOTE_REF, "");
  if (heading) text = text.replace(new RegExp(`^\\s*${escapeRe(heading.trim())}(?:\\s+|$)`), "");
  return text.replace(/\s+/g, " ").trim();
}

/**
 * Turn "[n]" passage markers into links to the matching source card.
 * Expects already-escaped HTML; markers outside 1..count are left as text.
 * @param {string} escapedHtml
 * @param {number} count - number of passages the model was given
 */
export function linkCitations(escapedHtml, count) {
  return escapedHtml.replace(/\[(\d+)\]/g, (whole, digits) => {
    const n = Number(digits);
    return n >= 1 && n <= count ? `<a class="ask-cite" href="#ask-src-${n}">${n}</a>` : whole;
  });
}
