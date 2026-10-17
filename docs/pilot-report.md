# Pilot report

Examples: 70 (answerable 54, near-miss 10, removed 6)
Gold chunk retrieved unaided: 20/54; gold at rank 1: 7/54

## Teachers

| teacher | answerable refused | removed answered | near-miss answered | answers over 2 sentences | mean answer chars |
|---|---|---|---|---|---|
| claude-opus-5-5 | 0/54 | 0/6 | 4/10 | 0/58 | 203 |
| claude-sonnet-5-5 | 0/54 | 0/6 | 2/10 | 1/56 | 216 |

Teachers agree on answer vs refuse: 68/70

## Cost (pilot, standard rates)

| role | model | calls | input tokens | output tokens | USD |
|---|---|---|---|---|---|
| questions | claude-sonnet-5-5 | 20 | 11,014 | 2,633 | 0.048 |
| teacher | claude-opus-5-5 | 70 | 61,002 | 6,917 | 0.382 |
| teacher | claude-sonnet-5-5 | 70 | 61,002 | 4,179 | 0.164 |

Projected for 5,000 examples with the Batch API (50% off):

- questions + teacher claude-opus-5-5: $0.0062 per example at standard rates -> $15.38
- questions + teacher claude-sonnet-5-5: $0.0030 per example at standard rates -> $7.58

## Examples for review

### answerable: What two things does the Go type checker verify at compile time?

1. Type Construction and Cycle Detection
2. Type Construction and Cycle Detection
3. Module Mirror and Checksum Database Launched — Checksum Database
4. Type Construction and Cycle Detection (gold)

> It's a step in the Go compiler that eliminates whole classes of errors at compile time. Specifically, the Go type checker verifies that: 1. Types appearing in the AST are valid (for example, a map's key type must be comparable). 2. Operations involving those types (or their values) are valid (for example, one can't add an int and a string). To accomplish this, the type checker constructs an internal representation for each type it encounters while traversing the AST—a process informally called type construction.

- **claude-opus-5-5**: The Go type checker verifies that types appearing in the AST are valid (for example, a map's key type must be comparable) and that operations involving those types or their values are valid (for example, you can't add an int and a string).
- **claude-sonnet-5-5**: The Go type checker verifies that types appearing in the AST are valid (for example, a map's key type must be comparable). It also verifies that operations involving those types or their values are valid (for example, you can't add an int and a string).

### answerable: Why must a map's key type be comparable in Go, according to the type checker's validity rules?

1. All your comparable types — Interface implementation vs constraint satisfaction
2. All your comparable types — Interface implementation vs constraint satisfaction
3. All your comparable types — Consequences and remedies
4. Type Construction and Cycle Detection (gold)

> It's a step in the Go compiler that eliminates whole classes of errors at compile time. Specifically, the Go type checker verifies that: 1. Types appearing in the AST are valid (for example, a map's key type must be comparable). 2. Operations involving those types (or their values) are valid (for example, one can't add an int and a string). To accomplish this, the type checker constructs an internal representation for each type it encounters while traversing the AST—a process informally called type construction.

- **claude-opus-5-5**: The type checker treats a comparable key type as one of the rules a map type must meet to be valid, and it enforces this at compile time to rule out whole classes of errors. The blog does not explain the underlying reason for the requirement beyond that.
- **claude-sonnet-5-5**: The Go type checker verifies that types appearing in the AST are valid, and one example of this is that a map's key type must be comparable. The blog doesn't go further into why the rule exists beyond treating it as a validity check that eliminates a class of errors at compile time.

### answerable: What is the informal name for how the Go type checker builds internal representations of types while walking the AST?

1. Type Construction and Cycle Detection (gold)
2. An Introduction To Generics — Introduction
3. Type Construction and Cycle Detection
4. Everything You Always Wanted to Know About Type Inference - And a Little Bit More — Unifying types with different structures

> It's a step in the Go compiler that eliminates whole classes of errors at compile time. Specifically, the Go type checker verifies that: 1. Types appearing in the AST are valid (for example, a map's key type must be comparable). 2. Operations involving those types (or their values) are valid (for example, one can't add an int and a string). To accomplish this, the type checker constructs an internal representation for each type it encounters while traversing the AST—a process informally called type construction.

- **claude-opus-5-5**: The Go type checker's process of building an internal representation for each type it encounters while traversing the AST is informally called type construction.
- **claude-sonnet-5-5**: It's informally called type construction: the type checker builds an internal representation for each type it encounters while traversing the AST.

### answerable: Which new packages were added to the standard library in Go 1.26?

1. Go: What's New in March 2010
2. Building StatHat with Go — Libraries
3. Go: What's New in March 2010
4. Go 1.26 is released — More improvements and changes (gold)

> More improvements and changes Go 1.26 introduces many improvements over Go 1.25 across its tools, the runtime, compiler, linker, and the standard library. This includes the addition of three new packages: crypto/hpke, crypto/mlkem/mlkemtest, and testing/cryptotest. There are port-specific changes and GODEBUG settings updates. Some of the additions in Go 1.26 are in an experimental stage and become exposed only when you explicitly opt in.

- **claude-opus-5-5**: Go 1.26 added three new packages to the standard library: crypto/hpke, crypto/mlkem/mlkemtest, and testing/cryptotest.
- **claude-sonnet-5-5**: Go 1.26 added three new packages to the standard library: crypto/hpke, crypto/mlkem/mlkemtest, and testing/cryptotest.

### answerable: How do I get access to the experimental features included in Go 1.26?

1. Go 1.25 is released
2. Go 1.25 is released
3. Go 1.23 is released
4. Go 1.26 is released — More improvements and changes (gold)

> More improvements and changes Go 1.26 introduces many improvements over Go 1.25 across its tools, the runtime, compiler, linker, and the standard library. This includes the addition of three new packages: crypto/hpke, crypto/mlkem/mlkemtest, and testing/cryptotest. There are port-specific changes and GODEBUG settings updates. Some of the additions in Go 1.26 are in an experimental stage and become exposed only when you explicitly opt in.

- **claude-opus-5-5**: Go 1.26's experimental additions become available only when you explicitly opt in to them. The blog does not explain the specific steps for opting in.
- **claude-sonnet-5-5**: Some of the additions in Go 1.26 are experimental and are only exposed when you explicitly opt in. The blog does not seem to cover the specific steps or settings for enabling them.

### answerable: Besides new packages, which parts of the toolchain does Go 1.26 improve over Go 1.25?

1. Perfectly Reproducible, Verified Go Toolchains — Perfectly Reproducible Builds for Go {#go}
2. Perfectly Reproducible, Verified Go Toolchains — Reproducibility in Go 1.20 {#go120}
3. Perfectly Reproducible, Verified Go Toolchains — Reproducibility in Go 1.21 {#go121}
4. Go 1.26 is released — More improvements and changes (gold)

> More improvements and changes Go 1.26 introduces many improvements over Go 1.25 across its tools, the runtime, compiler, linker, and the standard library. This includes the addition of three new packages: crypto/hpke, crypto/mlkem/mlkemtest, and testing/cryptotest. There are port-specific changes and GODEBUG settings updates. Some of the additions in Go 1.26 are in an experimental stage and become exposed only when you explicitly opt in.

- **claude-opus-5-5**: Go 1.26 improves on Go 1.25 across its tools, the runtime, the compiler, the linker, and the standard library. It also includes port-specific changes and updates to GODEBUG settings.
- **claude-sonnet-5-5**: Go 1.26 improves its tools, the runtime, the compiler, the linker, and the standard library over Go 1.25. It also includes port-specific changes and GODEBUG settings updates.

### removed: What did Go survey respondents most often say would improve Go support in their editor?

1. Participate in the 2016 Go User Survey and Company Questionnaire — The Go User Survey
2. Go 2018 Survey Results — Development environments
3. Go 2018 Survey Results — Attitudes towards Go
4. Go Developer Survey 2023 H2 Results — Developer goals for error handling {#err_handling}

- **claude-opus-5-5**: The blog does not seem to cover this.
- **claude-sonnet-5-5**: The blog does not seem to cover this.

### removed: What does the Go runtime's memory limit apply to, just the heap or the whole footprint?

1. Getting to Go: The Journey of Go's Garbage Collector — The Transcript
2. Getting to Go: The Journey of Go's Garbage Collector — The Transcript
3. Getting to Go: The Journey of Go's Garbage Collector — The Transcript
4. Go GC: Prioritizing low latency and simplicity — The Embellishment

- **claude-opus-5-5**: The blog does not seem to cover this.
- **claude-sonnet-5-5**: The blog does not seem to cover this.

### near-miss: How does the Go type checker detect cycles in recursive type declarations?

1. Type Construction and Cycle Detection — Recursive types
2. Type Construction and Cycle Detection — Recursive types
3. Type Construction and Cycle Detection
4. Type Construction and Cycle Detection — Recursive types

- **claude-opus-5-5**: The blog does not seem to cover this.
- **claude-sonnet-5-5**: The blog does not seem to cover this.

### near-miss: What new methods does the crypto/hpke package in Go 1.26 provide?

1. Go Cryptography Security Audit — What’s Next
2. Go 1.26 is released — More improvements and changes
3. Flight Recorder in Go 1.25 — Example
4. Go Cryptography Security Audit — crypto/pbkdf2: PBKDF2 does not enforce output length limitations (TOB-GOCL-5)

- **claude-opus-5-5**: The blog does not seem to cover this.
- **claude-sonnet-5-5**: The blog does not seem to cover this.

## Findings

- **Cost is small.** The pilot cost $0.59. 5,000 examples through the Batch API: about $15 with Opus 5.5 as teacher, about $8 with Sonnet 5.5.
- **Both teachers are good and agree on answer vs refuse in 68/70 cases.** No answerable example was refused and every removed example was refused. Opus is a little cleaner: it never mentions "passages" (Sonnet did twice, once in a long one-sentence list) and never goes over 2 sentences (Sonnet once).
- **Near-miss questions are not always misses.** The teacher answered 4/10 (Opus) and 2/10 (Sonnet), and the answers it gave were supported by the passages: a neighbouring chunk of the same post often answers them. Taking the label from the teacher, not from the construction, handles this as designed. The effective refusal share is about 17%.
- **Gold position is badly skewed.** Retrieval found the gold chunk unaided for only 20/54 questions: questions are paraphrased by design, and potion-base-8M is a small static model. When retrieval misses, the gold chunk is swapped in last and keeps the lowest score, so 37/54 answerable examples have the answer in passage [4]. A model trained on that would learn "the answer is in [4]". Fix: put a swapped-in gold chunk at a uniformly random position.
- **Partial-answer hedges.** About 5% of answers say what the passages cover and then add "the blog does not explain …". That's honest, and the judge counts it as faithful. Keep these, but tighten the teacher guidance so it never says "passages".
- **Go blog headings keep `{#anchor}` attributes**, e.g. "Perfectly Reproducible Builds for Go {#go}". Strip them in `normalize.go_blog`.
- One generated question ("Why must a map's key type be comparable…") wasn't answered by its passage; both teachers handled it honestly with a partial answer.
