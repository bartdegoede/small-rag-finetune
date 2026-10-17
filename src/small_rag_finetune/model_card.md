---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
datasets:
- bartdegoede/small-rag-answers
language:
- en
pipeline_tag: text-generation
library_name: transformers.js
tags:
- rag
- onnx
- transformers.js
- mlx
---

# Qwen3-0.6B, fine-tuned for short answers from retrieved passages

[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) with a LoRA fine-tune
(merged) that answers a question from 4 retrieved passages in one short
sentence, using the passages' own words, or says "The blog does not seem to
cover this." when they don't answer it. Built for "ask my site" pages that run
entirely in the browser: the q4f16 ONNX file is 330 MB and runs on WebGPU or
WASM through [transformers.js](https://huggingface.co/docs/transformers.js).

- Training data: [bartdegoede/small-rag-answers](https://huggingface.co/datasets/bartdegoede/small-rag-answers)
  (config `v3`), questions about the Go blog, Rust blog, Rust RFCs and Python
  PEPs, written by Claude Sonnet 5.5 and answered by Claude Opus 5.5.
- Code, browser example and full results:
  [github.com/bartdegoede/small-rag-finetune](https://github.com/bartdegoede/small-rag-finetune).

## Results

On a blog it never saw in training ([bart.degoe.de](https://bart.degoe.de)): 45
questions, 30 the blog answers and 15 it doesn't, judged by Claude Opus 5.5
against the passages. Greedy decoding, no repetition penalty.

| | faithful (of 30) | wrong answers | invented (of 15) |
|---|---|---|---|
| Qwen3-0.6B | 17 | 9 | 2 |
| **this model** | **16** | **3** | **0** |

It is right as often as the base model but rarely misleads: a third of the wrong
answers, and it doesn't make up answers to questions the site doesn't cover. The
price is more refusals on questions the passages do answer (11 of 30, against
4), mostly broad ones like "What has the blog written about X?".

## Prompt

The model was trained on exactly this format; other wording will work less well.
System prompt:

```
You answer questions about a personal technical blog. Use only the numbered passages provided. Answer in at most 2 sentences. If the passages do not answer the question, say that the blog does not seem to cover it.
```

User message: the passages, best first, each headed `[n] Title — Section` (or
`[n] Title` without a section), then the question:

```
Passages:

[1] Bloom filters — How they work
A Bloom filter is a bit array of m bits, all set to 0...

[2] ...

Question: How does a Bloom filter decide that an item is not in the set?
```

Retrieval in training: posts chunked into 600-character passages with 120
characters of overlap, each prefixed with its post title, embedded with
[`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M)
truncated to 128 dimensions, top 4 by cosine similarity
([`static-site-search-eval`](https://pypi.org/project/static-site-search-eval/)
builds that index for a folder of markdown posts). Other retrievers work, but
the model has learned to expect passages of about that size.

Turn off Qwen3's thinking (`enable_thinking: false`) and decode greedily. Don't
use a repetition penalty: transformers.js applies it to the prompt too, which
pushes the model away from the passages' words.

## transformers.js

```js
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline("text-generation", "bartdegoede/qwen3-0.6b-small-rag", {
  dtype: "q4f16",
  device: "webgpu", // or "wasm"
});
const messages = [
  { role: "system", content: SYSTEM_PROMPT },
  { role: "user", content: `Passages:\n\n${passages}\n\nQuestion: ${question}` },
];
const out = await generator(messages, {
  max_new_tokens: 120,
  do_sample: false,
  tokenizer_encode_kwargs: { enable_thinking: false },
});
console.log(out[0].generated_text.at(-1).content);
```

A complete static page (index build, retrieval, prompt, streaming worker) is in
the GitHub repo under `examples/browser/`.

## transformers (Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "bartdegoede/qwen3-0.6b-small-rag"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)
inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
)
out = model.generate(**inputs, max_new_tokens=120)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The root holds the merged weights in bfloat16 (also loadable with `mlx_lm`);
`onnx/model_q4f16.onnx` is the 4-bit browser build, exported with the
onnxruntime-genai model builder.

## Training

LoRA rank 8 on all layers, prompt masked, one epoch over 4,530 examples (batch 2,
gradient accumulation 2, learning rate 5e-5) with mlx-lm on an M4 Mac mini,
1h18m. The checkpoint was picked on Claude-judged answers to the dataset's
held-out test split, not on validation loss.

## Limitations

- The refusal is the fixed sentence "The blog does not seem to cover this.", and
  the system prompt says "blog". For a documentation site you'd want different
  wording, which means regenerating the training answers.
- Broad questions ("What has the blog written about X?", "How do I ...?") are
  often refused when the passages do cover them. Adding such questions to the
  training data made the model garble them instead.
- English technical writing only.
- The model only knows what the passages say; it will be wrong when retrieval
  is.

## Licence

Apache-2.0, like the base model. The training data is CC-BY-4.0 (see the dataset
card for its sources).
