# Ask your site, in the browser

A static page that answers questions about your posts with
[`bartdegoede/qwen3-0.6b-small-rag`](https://huggingface.co/bartdegoede/qwen3-0.6b-small-rag),
entirely in the visitor's browser: no server, no API key. It's a stripped-down
version of the "ask my blog" page on [bart.degoe.de](https://bart.degoe.de).

1. Semantic search finds the 4 passages closest to the question, using a static
   embedding model (`potion-base-8M`) that runs in plain JavaScript: about 4 MB
   to download, a few milliseconds per query.
2. If even the best passage is a poor match (cosine below 0.45), the page says
   the site doesn't cover it without calling the model.
3. Otherwise a Web Worker runs the model with transformers.js, on WebGPU when the
   browser has it and on WASM (slower) when it doesn't, and streams an answer of
   at most two sentences.

## Use it on your site

You need a folder of markdown posts with front matter (`title`, and `slug` if
your URLs use one), like Hugo's `content/post/` or Jekyll's `_posts/`. One flat
folder: posts in subfolders (Hugo page bundles) aren't picked up.

Build the search index into this folder. These settings are the ones the model
was trained with:

```bash
uvx --from static-site-search-eval==0.1.0 sss-eval build \
  --corpus path/to/your/posts --outdir examples/browser/search \
  --dims 128 --chunk-size 600 --chunk-overlap 120
```

Serve the folder and open it:

```bash
python3 -m http.server -d examples/browser 8000   # http://localhost:8000
```

The first question downloads the model (330 MB, cached by the browser after
that). On WebGPU an answer takes a second or two; on WASM, expect 10 to 30
seconds.

To put it on your site, copy `index.html`, `ask.js`, `worker.js`, `lib/` and
`search/` to a page of your own, and rebuild `search/` whenever posts change.
The source links use each post's URL as `sss-eval` derives it from the front
matter (`/slug/`, with `#anchor` for sections); if your site's URLs look
different, change `showSources()` in `ask.js`.

## Files

- `ask.js`: the page. `SEARCH_DIR` and `MODEL` at the top are the settings.
- `worker.js`: loads transformers.js and the model, streams answers, stops after
  two sentences.
- `lib/`: retrieval (`tokenizer.js`, `embed.js`, `search.js`, `retrieve.js`,
  `load-index.js`), the prompt the model was trained on (`prompt.js`) and answer
  clean-up (`display.js`). Copied from the blog, where they are unit-tested.

## Things to know

- The model says "The blog does not seem to cover this." when the passages don't
  answer the question: that sentence and the system prompt's "blog" are part of
  the training. For a documentation site you'd want to retrain with different
  wording (see the main README).
- It answers from the passages only, so it's only as good as retrieval. Broad
  questions ("What have you written about X?") often get the refusal even when
  the passages cover them.
- To check how well it does on your posts before you ship it, see "Evaluate it
  on your own site" in the main README.
