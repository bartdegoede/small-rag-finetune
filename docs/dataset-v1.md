# Dataset v1 (2026-10-15)

Generated with `small-rag-finetune generate` (seed 0): Sonnet 5.5 questions, Opus 5.5 teacher, Message Batches API. Cost **$16.45** (batch rates). Batches: `msgbatch_01LVDiD8ACDMrDi7NDqmtGco` (1,432 question requests), `msgbatch_01GGZkn2ipdAVHehHwmEHjpF` (5,055 teacher requests).

| split | examples |
|---|---|
| train | 4,529 |
| valid | 256 |
| test | 259 |

Dropped by the target checks: 10 over 2 sentences, 1 mentioning passages.

**Train make-up**

| kind | answered | refused |
|---|---|---|
| answerable | 3,492 | 1 |
| near-miss | 391 | 259 |
| removed | 72 | 314 |

- Refusals are 12.8% of train (574 of 4,529), below the 15-20% aimed for: the teacher found answers in neighbouring passages for 60% of near-miss questions and 19% of "removed" ones (Go blog release posts repeat each other). The label is the teacher's reading of the passages, as designed.
- Gold position (answerable): [1] 1,409, [2] 869, [3] 632, [4] 583. Pilot fix worked (was 7/6/4/37 of 54).
- Sources are balanced: Go blog 1,135, PEPs 1,123, Rust blog 1,145, Rust RFCs 1,126.
- Answer length: median 186 characters, p90 317.
- About 9% of answers (396) are partial: they answer what the passages cover and add a clause like "the blog does not go into more detail about …". Watch whether the fine-tuned models overuse this.
- Some answers keep markdown emphasis (`**10 days**`) and inline code from the source text.

**Samples**

- *Where does the Community Outreach Working Group publish its work so its efforts are fully transparent?* → "The Community Outreach Working Group publishes its work in an open GitHub repo, where its main initiatives are run as GitHub projects with issues you can comment on or submit yourself."
- *Does the libs team require a waiting period before approving an ACP, and how long is it?* → "No, a waiting period isn't strictly required, but the libs team highly recommends waiting **10 days** from submission before approving an ACP."
- *Why is centril splitting libsyntax into data and logic parts?* (removed) → "The blog does not seem to cover this."
