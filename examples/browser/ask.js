// The page: retrieve passages for a question, then let the worker answer from them.
import { loadIndex } from "./lib/load-index.js";
import { scoreQuery, topChunks } from "./lib/retrieve.js";
import { TOP_K, buildMessages, passesGate } from "./lib/prompt.js";
import { cleanSnippet } from "./lib/display.js";

// Change these two for your site.
const SEARCH_DIR = "./search"; // where `sss-eval build` wrote the index
const MODEL = "bartdegoede/qwen3-0.6b-small-rag";
// What the model says when the passages don't answer; also shown when the
// best passage scores below the relevance gate, without calling the model.
const REFUSAL = "The blog does not seem to cover this.";

const $ = (id) => document.getElementById(id);
const form = $("ask");
const status = $("status");
const answer = $("answer");
const sources = $("sources");

const indexPromise = loadIndex(SEARCH_DIR);
let worker = null;
let ready = null;

// Load the model on the first question: it is a 330 MB download, cached by the
// browser afterwards.
// WebGPU when the browser can actually give us a GPU: some expose
// navigator.gpu but return no adapter (headless Chrome, blocklisted GPUs).
async function pickDevice() {
  try {
    return (await navigator.gpu?.requestAdapter()) ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

function loadModel() {
  if (ready) return ready;
  ready = startWorker().catch((err) => {
    worker?.terminate();
    ready = null; // let the next question try again
    throw err;
  });
  return ready;
}

async function startWorker() {
  const device = await pickDevice();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const files = new Map();
  const loaded = new Promise((resolve, reject) => {
    worker.addEventListener("message", ({ data }) => {
      if (data.type === "progress" && data.total) {
        files.set(data.file, [data.loaded, data.total]);
        const [got, all] = [...files.values()].reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
        status.textContent = `Downloading the model: ${Math.round((100 * got) / all)}%`;
      } else if (data.type === "ready") resolve(data.device);
      else if (data.type === "error") reject(new Error(data.message));
    });
  });
  worker.postMessage({ type: "load", model: MODEL, device });
  return loaded;
}

function generate(messages) {
  return new Promise((resolve, reject) => {
    const onMessage = ({ data }) => {
      if (data.type === "token") answer.textContent = data.text;
      else if (data.type === "done" || data.type === "error") {
        worker.removeEventListener("message", onMessage);
        if (data.type === "done") resolve(data.text);
        else reject(new Error(data.message));
      }
    };
    worker.addEventListener("message", onMessage);
    worker.postMessage({ type: "generate", messages });
  });
}

function showSources(top) {
  sources.replaceChildren(
    ...top.map(({ chunk }, i) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = chunk.anchor ? `${chunk.href}#${chunk.anchor}` : chunk.href;
      a.textContent = `[${i + 1}] ${chunk.title}${chunk.heading ? ` — ${chunk.heading}` : ""}`;
      const p = document.createElement("p");
      p.textContent = cleanSnippet(chunk.snippet, chunk.heading).slice(0, 240) + "…";
      li.append(a, p);
      return li;
    })
  );
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = new FormData(form).get("q").trim();
  if (!question) return;
  const button = form.querySelector("button");
  button.disabled = true;
  answer.textContent = "";
  sources.replaceChildren();
  try {
    const idx = await indexPromise;
    const top = topChunks(scoreQuery(idx, question), idx.chunks, TOP_K);
    if (!passesGate(top)) {
      answer.textContent = REFUSAL;
      status.textContent = "";
      return;
    }
    showSources(top);
    status.textContent = "Loading the model…";
    const device = await loadModel();
    status.textContent = `Answering (${device === "webgpu" ? "WebGPU" : "CPU, slow"})…`;
    answer.textContent = await generate(buildMessages(question, top.map((h) => h.chunk)));
    status.textContent = "";
  } catch (err) {
    status.textContent = `Something went wrong: ${err.message}`;
  } finally {
    button.disabled = false;
  }
});
