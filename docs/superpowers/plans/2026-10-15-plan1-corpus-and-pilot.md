# Plan 1: corpus, passages and the Claude pilot

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn four openly licensed technical corpora into blog-style retrieval prompts, generate questions and teacher answers with Claude for 20 documents, and report the measured cost per example before any batch spend.

**Architecture:** Each source repo is cloned at a pinned commit and normalized into markdown files with YAML front matter, so the blog's own chunker (`sss_eval.chunk_post`, 600/120) reads them unchanged. A per-source index embeds chunks with `potion-base-8M` at 128 dimensions, like the blog's search. Claude (Sonnet 5.5) writes questions from a gold chunk; each question is retrieved into the exact `prompt.js` messages; two teachers (Opus 5.5 and Sonnet 5.5) answer those messages; a report compares them and projects batch cost for 5,000 examples.

**Tech Stack:** Python 3.12, uv, click, `static-site-search-eval==0.1.0` (sss_eval), model2vec, numpy, PyYAML, `anthropic` SDK, pytest.

Spec: `docs/superpowers/specs/2026-10-15-dataset-and-finetune-design.md`. Plan 2 (batch generation, training, eval, export fix, publishing) is written after the pilot numbers are in.

**Conventions for every task**

- Work in `~/projects/small-rag-finetune`. The user's shell has another virtualenv active: prefix commands with `unset VIRTUAL_ENV;` and use `uv run --no-sync` after the first `uv sync`.
- The Anthropic API key exists only in the user's interactive zsh profile. Commands that call the API run as `zsh -ic 'unset VIRTUAL_ENV; uv run --no-sync …'`.
- Format with `uv run --no-sync black src tests` and lint with `uv run --no-sync ruff check src tests` before each commit.
- Tests never call the API or download models: they use fake clients and a fake embedding model.

## File structure

| file | responsibility |
|---|---|
| `src/small_rag_finetune/normalize.py` | per-source file → `Doc(slug, title, body)` (markdown body) |
| `src/small_rag_finetune/sources.py` | the four pinned sources and `fetch()` (sparse git checkout) |
| `src/small_rag_finetune/corpus.py` | `split_of()` and `write_corpus()` (normalized `.md` files) |
| `src/small_rag_finetune/index.py` | `SourceIndex`: blog chunking + potion-base-8M@128 retrieval |
| `src/small_rag_finetune/claude.py` | prompts, request params, response parsing, token cost |
| `src/small_rag_finetune/examples.py` | `Example` and turning (gold chunk, questions) into prompts |
| `src/small_rag_finetune/pilot.py` | pilot orchestration and `pilot_report()` |
| `src/small_rag_finetune/__init__.py` | CLI: add `fetch-sources`, `build-corpus`, `pilot` |
| `tests/fakes.py` | `FakeModel` (bag-of-words embeddings) and `FakeClient` |

Existing smoke-test code (`data.py`, `prompt.py`, `vendor/`, `scripts/export_onnx.py`, `js/generate.mjs`) is committed as is in Task 1 and not changed by this plan.

---

### Task 1: Commit the smoke-test scaffold

**Files:**
- Add (existing, uncommitted): `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `package.json`, `pnpm-lock.yaml`, `src/`, `tests/`, `scripts/`, `js/`, `eval/`, `docs/superpowers/specs/2026-10-15-dataset-and-finetune-design.md` (retrieval fix), this plan.

- [ ] **Step 1: Check what will be committed**

Run: `git status --short`
Expected: the files above as `??` or `M`; nothing under `data/`, `runs/`, `logs/`, `node_modules/`, `.venv/` (all gitignored).

- [ ] **Step 2: Run the existing tests**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest -v`
Expected: 2 passed (`tests/test_prompt_parity.py`).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock .python-version README.md package.json pnpm-lock.yaml src tests scripts js eval docs
git commit -m "Smoke-test pipeline: neural-bridge data, MLX LoRA, ONNX q4f16 export, transformers.js runner"
```

---

### Task 2: Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the runtime dependencies**

Run: `unset VIRTUAL_ENV; uv add anthropic pyyaml "static-site-search-eval==0.1.0"`
Expected: `pyproject.toml` lists `anthropic`, `pyyaml`, `static-site-search-eval==0.1.0`; `uv.lock` updated.

- [ ] **Step 2: Check the imports**

Run: `unset VIRTUAL_ENV; uv run --no-sync python -c "import anthropic, yaml; from sss_eval.chunk import chunk_post; from sss_eval.corpus import load_corpus; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add anthropic, pyyaml and sss_eval dependencies"
```

---

### Task 3: Normalizers for Go blog, Rust blog and Rust RFCs

**Files:**
- Create: `src/small_rag_finetune/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_normalize.py
from small_rag_finetune.normalize import go_blog, rust_blog, rust_rfc


def write(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_go_blog_reads_yaml_title_and_drops_template_actions(tmp_path):
    p = write(tmp_path, "go1.21.md", (
        "---\ntitle: Go 1.21 is released!\ndate: 2023-08-08\nby:\n- Eli\n---\n\n"
        "Today we release Go 1.21.\n\n{{image \"go1.21/gopher.png\"}}\n\n## Tools\n\nFaster builds.\n"
    ))
    doc = go_blog(p)
    assert doc.slug == "go1.21"
    assert doc.title == "Go 1.21 is released!"
    assert "{{" not in doc.body
    assert "## Tools" in doc.body and "Faster builds." in doc.body


def test_go_blog_without_front_matter_is_skipped(tmp_path):
    assert go_blog(write(tmp_path, "x.md", "no front matter")) is None


def test_rust_blog_reads_toml_and_prefixes_inside_rust(tmp_path):
    fm = '+++\npath = "2025/12/05/x"\ntitle = "crates.io: Malicious crates"\n\n[extra]\nteam = "t"\n+++\n\n'
    body = "## Summary\n\nTwo crates were removed.\n\n[socket]: https://socket.dev\n"
    top = rust_blog(write(tmp_path, "content/malicious.md", fm + body))
    inside = rust_blog(write(tmp_path, "content/inside-rust/team-update.md", fm + body))
    nested = rust_blog(write(tmp_path, "content/survey-2024/index.md", fm + body))
    assert top.slug == "malicious" and top.title == "crates.io: Malicious crates"
    assert inside.slug == "inside-rust-team-update"
    assert nested.slug == "survey-2024"
    assert "https://socket.dev" not in top.body
    assert "Two crates were removed." in top.body


def test_rust_blog_skips_section_index(tmp_path):
    assert rust_blog(write(tmp_path, "content/_index.md", "+++\ntitle = \"x\"\n+++\n")) is None


def test_rust_rfc_titles_from_filename_and_drops_metadata(tmp_path):
    p = write(tmp_path, "text/2094-nll.md", (
        "- Feature Name: nll\n- Start Date: 2017-08-02\n"
        "- RFC PR: [rust-lang/rfcs#2094](https://github.com/rust-lang/rfcs/pull/2094)\n\n"
        "## Summary\n[summary]: #summary\n\nExtend the borrow checker.\n\n"
        "- Note: a list item in the body stays.\n"
    ))
    doc = rust_rfc(p)
    assert doc.slug == "2094-nll"
    assert doc.title == "RFC 2094: nll"
    assert "Start Date" not in doc.body and "#summary" not in doc.body
    assert "Extend the borrow checker." in doc.body
    assert "- Note: a list item in the body stays." in doc.body


def test_rust_rfc_skips_non_rfc_files(tmp_path):
    assert rust_rfc(write(tmp_path, "text/README.md", "# hi")) is None
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'small_rag_finetune.normalize'`

- [ ] **Step 3: Implement**

```python
# src/small_rag_finetune/normalize.py
"""Turn each source's files into documents the blog's chunker reads.

sss_eval parses markdown with a YAML front-matter block and `#` headings, so
every source is normalized to that: a Doc with a slug, a title and a markdown
body. A normalizer returns None for files that aren't documents.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Doc:
    slug: str
    title: str
    body: str


_YAML_FM = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_TOML_FM = re.compile(r"\A\+\+\+\n(.*?)\n\+\+\+\n", re.DOTALL)
_GO_ACTION = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_LINK_DEF = re.compile(r"^\[[^\]]+\]:\s*\S.*$", re.MULTILINE)
_RFC_META = re.compile(r"^- [A-Za-z ]+:.*$", re.MULTILINE)


def go_blog(path: Path) -> Doc | None:
    """golang/website _content/blog/*.md: YAML front matter, Go template actions."""
    raw = path.read_text()
    m = _YAML_FM.match(raw)
    if not m:
        return None
    fm = yaml.safe_load(m.group(1)) or {}
    body = _GO_ACTION.sub("", raw[m.end() :])
    return Doc(path.stem, str(fm.get("title", "")).strip(), body)


def rust_blog(path: Path) -> Doc | None:
    """rust-lang/blog.rust-lang.org content/**/*.md: Zola TOML front matter."""
    if path.name == "_index.md":
        return None
    raw = path.read_text()
    m = _TOML_FM.match(raw)
    if not m:
        return None
    fm = tomllib.loads(m.group(1))
    slug = path.parent.name if path.name == "index.md" else path.stem
    if "inside-rust" in path.parts:
        slug = f"inside-rust-{slug}"
    body = _LINK_DEF.sub("", raw[m.end() :])
    return Doc(slug, str(fm.get("title", "")).strip(), body)


def rust_rfc(path: Path) -> Doc | None:
    """rust-lang/rfcs text/NNNN-name.md: a "- Key: value" header list, no title."""
    num, _, name = path.stem.partition("-")
    if not num.isdigit() or not name:
        return None
    raw = path.read_text()
    first = re.search(r"^#", raw, re.MULTILINE)
    cut = first.start() if first else 0
    body = _LINK_DEF.sub("", _RFC_META.sub("", raw[:cut]) + raw[cut:])
    title = f"RFC {int(num)}: {name.replace('-', ' ').replace('_', ' ')}"
    return Doc(path.stem, title, body)
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_normalize.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/small_rag_finetune/normalize.py tests/test_normalize.py
git commit -m "Normalize Go blog, Rust blog and Rust RFC files into markdown docs"
```

---

### Task 4: PEP normalizer (reStructuredText → markdown)

**Files:**
- Modify: `src/small_rag_finetune/normalize.py` (append)
- Test: `tests/test_normalize.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_normalize.py
from small_rag_finetune.normalize import pep, rst_to_markdown

PEP_TEXT = """PEP: 9999
Title: Example Things
Author: A <a@b.c>,
        B <b@c.d>
Status: Final

Abstract
========

This uses ``x := 1`` and :pep:`8` and `a link <https://x.org>`_.

Rationale
=========

Example::

    code here
    more code

Details
-------

Text after.

.. note::

   Dropped note.

Final line.
"""


def test_pep_title_and_header_dropped(tmp_path):
    doc = pep(write(tmp_path, "peps/pep-9999.rst", PEP_TEXT))
    assert doc.slug == "pep-9999"
    assert doc.title == "PEP 9999: Example Things"
    assert "Author" not in doc.body and "B <b@c.d>" not in doc.body


def test_rst_headings_become_markdown_levels_in_order_of_appearance():
    md = pep_body()
    assert "# Abstract" in md.splitlines()
    assert "# Rationale" in md.splitlines()
    assert "## Details" in md.splitlines()


def test_rst_inline_markup_and_blocks():
    md = pep_body()
    assert "This uses `x := 1` and PEP 8 and a link." in md
    assert "Example:" in md.splitlines()
    assert "code here" not in md
    assert "Dropped note." not in md
    assert "Text after." in md and "Final line." in md


def test_overlined_title():
    md = rst_to_markdown("=====\nTitle\n=====\n\nBody.\n")
    assert md.splitlines()[0] == "# Title"


def test_non_pep_file_is_skipped(tmp_path):
    assert pep(write(tmp_path, "peps/conf.rst", "x")) is None


def pep_body() -> str:
    return rst_to_markdown(PEP_TEXT.split("\n\n", 1)[1])
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_normalize.py -v`
Expected: FAIL with `ImportError: cannot import name 'pep'`

- [ ] **Step 3: Implement**

Append to `src/small_rag_finetune/normalize.py`:

```python
_PEP_FILE = re.compile(r"^pep-(\d+)\.rst$")
_UNDERLINE = re.compile(r"^([=\-~^`#*+'\"])\1{2,}\s*$")
_RST_LINK = re.compile(r"`([^`<]+?)\s*<[^>]+>`__?")
_PEP_ROLE = re.compile(r":pep:`(\d+)`")
_ROLE = re.compile(r":[a-z]+(?::[a-z]+)?:`([^`]*)`")
_LITERAL = re.compile(r"``(.+?)``")
_REF = re.compile(r"(?<!`)`([^`]+)`_")


def _inline(s: str) -> str:
    s = _RST_LINK.sub(r"\1", s)
    s = _PEP_ROLE.sub(r"PEP \1", s)
    s = _ROLE.sub(r"\1", s)
    s = _LITERAL.sub(r"`\1`", s)
    return _REF.sub(r"\1", s)


def rst_to_markdown(text: str) -> str:
    """Just enough reStructuredText for PEP prose: section titles become `#`
    headings (levels in order of first appearance, like docutils), directives
    and literal blocks are dropped, inline markup is unwrapped."""
    lines = text.splitlines()
    out: list[str] = []
    levels: list[str] = []
    skipping = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if skipping:
            if not line.strip() or line[:1].isspace():
                i += 1
                continue
            skipping = False
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        title = None
        if (
            _UNDERLINE.match(line)
            and nxt.strip()
            and i + 2 < len(lines)
            and lines[i + 2].strip() == line.strip()
        ):
            title, style, step = nxt.strip(), "over" + line.strip()[0], 3
        elif (
            line.strip()
            and not line[:1].isspace()
            and _UNDERLINE.match(nxt)
            and len(nxt.strip()) >= len(line.strip())
        ):
            title, style, step = line.strip(), nxt.strip()[0], 2
        if title is not None:
            if style not in levels:
                levels.append(style)
            out += ["#" * min(levels.index(style) + 1, 6) + " " + _inline(title), ""]
            i += step
            continue
        if line.startswith(".. "):
            skipping = True
        elif line.rstrip().endswith("::"):
            lead = line.rstrip()[:-2].rstrip()
            if lead:
                out.append(_inline(lead + ":"))
            skipping = True
        else:
            out.append(_inline(line.strip() if line[:1].isspace() else line))
        i += 1
    return "\n".join(out)


def _pep_header(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """RFC 822 style header block up to the first blank line."""
    header: dict[str, str] = {}
    key = None
    i = 0
    while i < len(lines) and lines[i].strip():
        line = lines[i]
        if line[:1].isspace() and key:
            header[key] += " " + line.strip()
        else:
            key, _, value = line.partition(":")
            key = key.strip()
            header[key] = value.strip()
        i += 1
    return header, lines[i:]


def pep(path: Path) -> Doc | None:
    """python/peps peps/pep-NNNN.rst: an RFC 822 header, then reStructuredText."""
    m = _PEP_FILE.match(path.name)
    if not m:
        return None
    header, rest = _pep_header(path.read_text().splitlines())
    if "Title" not in header:
        return None
    num = int(m.group(1))
    return Doc(f"pep-{num:04d}", f"PEP {num}: {header['Title']}", rst_to_markdown("\n".join(rest)))
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_normalize.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/small_rag_finetune/normalize.py tests/test_normalize.py
git commit -m "Normalize PEPs: reStructuredText sections and inline markup to markdown"
```

---

### Task 5: Sources, splits and the normalized corpus

**Files:**
- Create: `src/small_rag_finetune/sources.py`
- Create: `src/small_rag_finetune/corpus.py`
- Modify: `src/small_rag_finetune/__init__.py` (add two commands)
- Test: `tests/test_corpus.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corpus.py
from collections import Counter

from sss_eval.post import parse_post

from small_rag_finetune.corpus import MIN_BODY_CHARS, split_of, write_corpus
from small_rag_finetune.normalize import Doc
from small_rag_finetune.sources import SOURCES, Source


def test_split_is_deterministic_and_roughly_90_5_5():
    assert split_of("go-blog", "go1.21") == split_of("go-blog", "go1.21")
    counts = Counter(split_of("s", f"doc-{i}") for i in range(4000))
    assert 0.87 < counts["train"] / 4000 < 0.93
    assert 0.03 < counts["valid"] / 4000 < 0.07
    assert 0.03 < counts["test"] / 4000 < 0.07


def test_split_depends_on_source():
    pairs = [(split_of("a", f"d{i}"), split_of("b", f"d{i}")) for i in range(200)]
    assert any(x != y for x, y in pairs)


def test_write_corpus_writes_files_sss_eval_can_parse(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "long.txt").write_text("long")
    (checkout / "short.txt").write_text("short")
    (checkout / "none.txt").write_text("none")

    def fake(path):
        if path.stem == "none":
            return None
        n = MIN_BODY_CHARS if path.stem == "long" else 10
        return Doc(path.stem, f"Title: {path.stem}", "## Part\n\n" + "x" * n)

    src = Source("fake", "https://example.invalid/repo", "abc", ".", "*.txt", "MIT", fake)
    n = write_corpus(src, checkout, tmp_path / "corpus")
    assert n == 1
    post = parse_post(tmp_path / "corpus" / "fake" / "long.md")
    assert post.slug == "long"
    assert post.title == "Title: long"
    assert post.sections[0].heading == "Part"


def test_four_sources_are_pinned():
    assert set(SOURCES) == {"go-blog", "rust-blog", "rust-rfcs", "peps"}
    assert all(len(s.commit) == 40 for s in SOURCES.values())
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'small_rag_finetune.corpus'`

- [ ] **Step 3: Implement `sources.py`**

```python
# src/small_rag_finetune/sources.py
"""The corpora, pinned to a commit, and a sparse git checkout of each."""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import normalize
from .normalize import Doc


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    commit: str
    sparse: str  # directory to check out
    pattern: str  # glob, relative to the checkout
    licence: str
    normalize: Callable[[Path], Doc | None]


SOURCES = {
    s.name: s
    for s in [
        Source(
            "go-blog",
            "https://github.com/golang/website",
            "bacaf18c1c9234fabbd08847eb33073c50d9254b",
            "_content/blog",
            "_content/blog/*.md",
            "CC-BY-4.0",
            normalize.go_blog,
        ),
        Source(
            "rust-blog",
            "https://github.com/rust-lang/blog.rust-lang.org",
            "c76891ecae17a22ee76a29096c532709be853b87",
            "content",
            "content/**/*.md",
            "MIT OR Apache-2.0",
            normalize.rust_blog,
        ),
        Source(
            "rust-rfcs",
            "https://github.com/rust-lang/rfcs",
            "d2705ad64ba47e812a2b6b19d6f613fb4c14c5e0",
            "text",
            "text/*.md",
            "MIT OR Apache-2.0",
            normalize.rust_rfc,
        ),
        Source(
            "peps",
            "https://github.com/python/peps",
            "b80c7bf8d5a89fc47b732504a9e1219b79677531",
            "peps",
            "peps/pep-*.rst",
            "public domain / CC0-1.0",
            normalize.pep,
        ),
    ]
}


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def fetch(src: Source, root: Path) -> Path:
    """Partial clone (no blobs up front), sparse to one directory, at the pinned commit."""
    dest = root / src.name
    if not (dest / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", src.repo, str(dest))
    _git("-C", str(dest), "sparse-checkout", "set", src.sparse)
    _git("-C", str(dest), "checkout", "--quiet", src.commit)
    return dest
```

- [ ] **Step 4: Implement `corpus.py`**

```python
# src/small_rag_finetune/corpus.py
"""Normalized corpus on disk: one directory of markdown posts per source."""

import hashlib
from pathlib import Path

import yaml

from .sources import Source

# Shorter documents give one or two chunks: too little for look-alike passages.
MIN_BODY_CHARS = 1500


def split_of(source: str, slug: str) -> str:
    """train / valid / test at 90 / 5 / 5, by document, stable across runs."""
    h = int(hashlib.sha1(f"{source}/{slug}".encode()).hexdigest(), 16) % 100
    return "test" if h < 5 else "valid" if h < 10 else "train"


def write_corpus(src: Source, checkout: Path, out_root: Path) -> int:
    out = out_root / src.name
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(checkout.glob(src.pattern)):
        doc = src.normalize(path)
        if doc is None or not doc.title or len(doc.body) < MIN_BODY_CHARS:
            continue
        fm = yaml.safe_dump(
            {"title": doc.title, "slug": doc.slug}, allow_unicode=True, sort_keys=False
        )
        (out / f"{doc.slug}.md").write_text(f"---\n{fm}---\n\n{doc.body.strip()}\n")
        n += 1
    return n
```

- [ ] **Step 5: Run the tests to check they pass**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_corpus.py -v`
Expected: 4 passed

- [ ] **Step 6: Add the CLI commands**

Append to `src/small_rag_finetune/__init__.py`:

```python
@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=Path("data/sources"))
def fetch_sources(root: Path) -> None:
    """Check out every source at its pinned commit."""
    from .sources import SOURCES, fetch

    for src in SOURCES.values():
        click.echo(f"{src.name}: {fetch(src, root)}")


@main.command()
@click.option("--sources", type=click.Path(path_type=Path), default=Path("data/sources"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/corpus"))
def build_corpus(sources: Path, out: Path) -> None:
    """Normalize every source into markdown posts the blog's chunker reads."""
    from .corpus import write_corpus
    from .sources import SOURCES

    for src in SOURCES.values():
        n = write_corpus(src, sources / src.name, out)
        click.echo(f"{src.name}: {n} documents -> {out / src.name}")
```

- [ ] **Step 7: Fetch and build the real corpus**

Run: `unset VIRTUAL_ENV; uv run --no-sync small-rag-finetune fetch-sources && uv run --no-sync small-rag-finetune build-corpus`
Expected: four checkouts under `data/sources/`, then document counts per source. Rough expectations (files found on 2026-10-15: 343 Go blog, ~700 Rust blog, 745 RFCs, ~700 PEPs): a few hundred each after the 1,500-character filter. A source with fewer than 100 documents means its normalizer is skipping files; open three of its files and the matching outputs in `data/corpus/<source>/` before going on.

- [ ] **Step 8: Eyeball the output**

Run: `for s in go-blog rust-blog rust-rfcs peps; do f=$(ls data/corpus/$s | sed -n 7p); echo "== $s/$f"; head -25 data/corpus/$s/$f; done`
Expected: front matter with a real title and slug, then readable prose with `#` headings; no `{{`, no RST underlines, no `- Start Date:` lines.

- [ ] **Step 9: Commit**

```bash
git add src/small_rag_finetune/sources.py src/small_rag_finetune/corpus.py src/small_rag_finetune/__init__.py tests/test_corpus.py
git commit -m "Pinned sources, 90/5/5 document splits and the normalized corpus"
```

---

### Task 6: Test fakes and the per-source index

**Files:**
- Create: `tests/fakes.py`
- Create: `src/small_rag_finetune/index.py`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write the fakes**

```python
# tests/fakes.py
"""Stand-ins for the embedding model and the Anthropic client."""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class FakeModel:
    """Bag-of-words over a tiny vocabulary: similar enough to model2vec for ranking tests."""

    VOCAB = ["go", "rust", "python", "generics", "async", "walrus", "lifetimes", "modules"]

    def encode(self, texts):
        return np.array(
            [[t.lower().count(w) + 0.01 for w in self.VOCAB] for t in texts], dtype=np.float32
        )


def write_post(root: Path, slug: str, title: str, body: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{slug}.md").write_text(f"---\ntitle: {title}\nslug: {slug}\n---\n\n{body}\n")


def small_corpus(root: Path) -> None:
    write_post(root, "generics", "Go generics", "Go generics add type parameters. " * 12)
    write_post(root, "async", "Async Rust", "Async rust uses futures and executors. " * 12)
    write_post(root, "walrus", "The walrus", "The python walrus operator assigns. " * 12)
    write_post(root, "lifetimes", "Lifetimes", "Rust lifetimes track borrows. " * 12)


class FakeClient:
    """client.messages.create(**params) with canned replies and fixed usage."""

    QUESTIONS = "Q1: What do go generics add?\nQ2: How are go generics written?\nQ3: NONE\nNEAR: When will rust get generics?"
    ANSWER = "Go generics add type parameters to functions and types."

    def __init__(self):
        self.messages = self
        self.calls = []
        self._lock = threading.Lock()

    def create(self, **params):
        with self._lock:
            self.calls.append(params)
        text = self.ANSWER if "system" in params else self.QUESTIONS
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            stop_reason="end_turn",
        )
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_index.py
from fakes import FakeModel, small_corpus

from small_rag_finetune.index import CHUNK_OVERLAP, CHUNK_SIZE, SourceIndex


def build(tmp_path):
    small_corpus(tmp_path)
    return SourceIndex.build(tmp_path, model=FakeModel())


def test_matches_blog_index_settings():
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (600, 120)


def test_chunks_carry_title_heading_and_snippet(tmp_path):
    idx = build(tmp_path)
    c = next(c for c in idx.chunks if c.post == "generics")
    assert c.title == "Go generics"
    assert c.text.startswith("Go generics\n\n")
    assert len(c.snippet) <= CHUNK_SIZE


def test_top_ranks_by_cosine(tmp_path):
    idx = build(tmp_path)
    hits = idx.top("go generics", k=2)
    assert hits[0].chunk.post == "generics"
    assert hits[0].score >= hits[1].score


def test_exclude_post_removes_the_document(tmp_path):
    idx = build(tmp_path)
    hits = idx.top("go generics", k=4, exclude_post="generics")
    assert hits and all(h.chunk.post != "generics" for h in hits)


def test_scores_has_one_value_per_chunk(tmp_path):
    idx = build(tmp_path)
    assert len(idx.scores("rust")) == len(idx.chunks)
```

Add `pythonpath = ["tests"]` so `from fakes import …` works, in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["tests"]
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'small_rag_finetune.index'`

- [ ] **Step 4: Implement**

```python
# src/small_rag_finetune/index.py
"""The blog's retrieval, one index per source.

Same chunker and settings as the blog's scripts/build_search_index.py
(sss_eval chunk_post, 600 chars, 120 overlap, title prefix) and the same
model (potion-base-8M truncated to 128 dims). The browser ships an int8 copy
of these vectors; float32 here ranks the same up to near-ties.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sss_eval.chunk import Chunk, chunk_post
from sss_eval.corpus import load_corpus

from .prompt import TOP_K

MODEL_ID = "minishlab/potion-base-8M"
DIMS = 128
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class Hit:
    index: int
    score: float
    chunk: Chunk


class SourceIndex:
    def __init__(self, chunks: list[Chunk], model) -> None:
        self.chunks = chunks
        self.model = model
        self.post = np.array([c.post for c in chunks])
        self.emb = self._encode([c.text for c in chunks])

    @classmethod
    def build(cls, corpus_dir: Path, model=None) -> "SourceIndex":
        if model is None:
            from model2vec import StaticModel

            model = StaticModel.from_pretrained(MODEL_ID, dimensionality=DIMS)
        chunks = [
            c
            for post in load_corpus(corpus_dir)
            for c in chunk_post(post, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, title_prefix=True)
        ]
        return cls(chunks, model)

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self.model.encode(texts), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def scores(self, query: str, exclude_post: str | None = None) -> np.ndarray:
        s = self.emb @ self._encode([query])[0]
        if exclude_post is not None:
            s = np.where(self.post == exclude_post, -np.inf, s)
        return s

    def top(self, query: str, k: int = TOP_K, exclude_post: str | None = None) -> list[Hit]:
        """Best first; ties broken by chunk order, like retrieve.js topChunks."""
        s = self.scores(query, exclude_post)
        order = np.lexsort((np.arange(len(s)), -s))[:k]
        return [Hit(int(i), float(s[i]), self.chunks[i]) for i in order if np.isfinite(s[i])]
```

- [ ] **Step 5: Run the tests to check they pass**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_index.py -v`
Expected: 5 passed

- [ ] **Step 6: Smoke-check on the real corpus**

Run: `unset VIRTUAL_ENV; uv run --no-sync python -c "from pathlib import Path; from small_rag_finetune.index import SourceIndex; i = SourceIndex.build(Path('data/corpus/go-blog')); print(len(i.chunks)); [print(round(h.score,3), h.chunk.title, '|', h.chunk.heading) for h in i.top('How do I write a generic function in Go?')]"`
Expected: a chunk count in the thousands, then 4 hits whose titles are about generics.

- [ ] **Step 7: Commit**

```bash
git add tests/fakes.py tests/test_index.py src/small_rag_finetune/index.py pyproject.toml
git commit -m "Per-source index with the blog's chunking and potion-base-8M@128 retrieval"
```

---

### Task 7: Claude prompts, requests, parsing and cost

**Files:**
- Create: `src/small_rag_finetune/claude.py`
- Test: `tests/test_claude.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_claude.py
from types import SimpleNamespace

from fakes import FakeClient

from small_rag_finetune.claude import (
    QUESTION_MODEL,
    TEACHERS,
    complete,
    cost,
    normalize_answer,
    parse_questions,
    question_params,
    target_key,
    teacher_params,
)
from small_rag_finetune.prompt import REFUSAL, build_messages


def test_parse_questions_skips_none():
    qs = parse_questions("Q1: What is X?\nQ2: NONE\nQ3: Why Y?\nNEAR: When Z?\n")
    assert qs.answerable == ["What is X?", "Why Y?"]
    assert qs.near_miss == "When Z?"


def test_parse_questions_without_near_miss():
    assert parse_questions("Q1: A?\nNEAR: NONE").near_miss is None


def test_normalize_answer_collapses_whitespace_and_refusals():
    assert normalize_answer("  Go 1.21\n adds PGO.  ") == "Go 1.21 adds PGO."
    assert normalize_answer("The blog does not seem to cover this topic.") == REFUSAL


def test_question_params_carry_the_passage():
    chunk = SimpleNamespace(title="Go 1.21", heading="Tools", snippet="PGO is now GA.")
    p = question_params(chunk)
    assert p["model"] == QUESTION_MODEL
    content = p["messages"][0]["content"]
    assert "Go 1.21" in content and "Tools" in content and "PGO is now GA." in content


def test_teacher_sees_the_student_prompt_plus_guidance():
    msgs = build_messages("What is PGO?", [{"title": "T", "heading": None, "snippet": "s" * 200}])
    p = teacher_params(TEACHERS[0], msgs)
    assert p["system"].startswith(msgs[0]["content"])
    assert len(p["system"]) > len(msgs[0]["content"])
    assert p["messages"] == [{"role": "user", "content": msgs[1]["content"]}]


def test_complete_returns_the_text_block():
    text, usage = complete(FakeClient(), {"model": "m", "messages": []})
    assert text.startswith("Q1:")
    assert usage.input_tokens == 1000


def test_cost_and_batch_discount():
    assert cost("claude-opus-5-5", 1_000_000, 100_000) == 4.0 + 2.0
    assert cost("claude-sonnet-5-5", 1_000_000, 0, batch=True) == 1.0


def test_target_keys():
    assert [target_key(m) for m in TEACHERS] == ["target_opus", "target_sonnet"]
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_claude.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'small_rag_finetune.claude'`

- [ ] **Step 3: Implement**

```python
# src/small_rag_finetune/claude.py
"""Everything that talks to Claude: prompts, request params, parsing, cost.

Requests are plain dicts so the same params work for messages.create (pilot)
and for Message Batches (plan 2). Replies are plain text with a fixed line
format; no structured-output features needed.
"""

import re
from dataclasses import dataclass

from .prompt import REFUSAL

QUESTION_MODEL = "claude-sonnet-5-5"
TEACHERS = ("claude-opus-5-5", "claude-sonnet-5-5")
# USD per 1M tokens (input, output), standard rates.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
BATCH_DISCOUNT = 0.5

QUESTION_PROMPT = """You are writing questions for a site-search assistant that answers from retrieved passages.

<title>{title}</title>
<section>{heading}</section>
<passage>
{passage}
</passage>

Write 3 questions that a reader of this site might type into its search box and that THIS passage answers on its own.
- Each question must make sense without seeing the passage: name the subject (the project, feature, release, PEP or RFC) instead of writing "this" or "the passage".
- The answer must be stated in the passage, not inferred from general knowledge.
- Mix styles: a fact lookup, a how or why question, and one that needs a detail from the middle or end of the passage.
- Under 20 words each, plain wording; do not copy more than three words in a row from the passage.
- If the passage cannot support a question (for example it is only a list of names or links), write NONE on that line.

Then write one NEAR-MISS question: on the same subject, something a reader could plausibly ask, that this passage does NOT answer, not even partly. Write NONE if you cannot.

Reply with exactly these four lines and nothing else:
Q1: ...
Q2: ...
Q3: ...
NEAR: ..."""

TEACHER_GUIDANCE = """

How to answer:
- Answer the question directly in plain declarative sentences. Do not mention the passages, passage numbers like [2], or phrases such as "according to".
- Use only facts stated in the passages. Prefer one sentence; use a second only when the question needs it.
- If the passages do not answer the question, reply with exactly: The blog does not seem to cover this."""

_Q = re.compile(r"^Q\d+:\s*(.+?)\s*$", re.MULTILINE)
_NEAR = re.compile(r"^NEAR:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Questions:
    answerable: list[str]
    near_miss: str | None


@dataclass(frozen=True)
class Call:
    role: str  # "questions" | "teacher"
    model: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens, which are billed as output


def question_params(chunk) -> dict:
    content = QUESTION_PROMPT.format(
        title=chunk.title, heading=chunk.heading or "(none)", passage=chunk.snippet
    )
    return {
        "model": QUESTION_MODEL,
        "max_tokens": 4000,
        "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": content}],
    }


def teacher_params(model: str, messages: list[dict]) -> dict:
    """The student's exact system + user messages; the guidance is teacher-only."""
    system, user = messages
    return {
        "model": model,
        "max_tokens": 4000,
        "output_config": {"effort": "low"},
        "system": system["content"] + TEACHER_GUIDANCE,
        "messages": [{"role": "user", "content": user["content"]}],
    }


def complete(client, params: dict):
    msg = client.messages.create(**params)
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return text, msg.usage


def parse_questions(text: str) -> Questions:
    answerable = [q for q in _Q.findall(text) if q.upper() != "NONE"]
    m = _NEAR.search(text)
    near = m.group(1) if m and m.group(1).upper() != "NONE" else None
    return Questions(answerable, near)


def normalize_answer(text: str) -> str:
    t = " ".join(text.split())
    if t.lower().startswith("the blog does not seem to cover"):
        return REFUSAL
    return t


def cost(model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
    pin, pout = PRICES[model]
    usd = input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
    return usd * BATCH_DISCOUNT if batch else usd


def target_key(model: str) -> str:
    """claude-opus-5-5 -> target_opus"""
    return "target_" + model.removeprefix("claude-").split("-")[0]
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_claude.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/small_rag_finetune/claude.py tests/test_claude.py
git commit -m "Claude prompts, request params, reply parsing and token cost"
```

---

### Task 8: Examples (gold insertion, removed, near-miss)

**Files:**
- Create: `src/small_rag_finetune/examples.py`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_examples.py
from fakes import FakeModel, small_corpus

from small_rag_finetune.claude import Questions
from small_rag_finetune.examples import examples_for, with_gold
from small_rag_finetune.index import Hit, SourceIndex


def index(tmp_path):
    small_corpus(tmp_path)
    return SourceIndex.build(tmp_path, model=FakeModel())


def gold_of(idx, post):
    return next(i for i, c in enumerate(idx.chunks) if c.post == post)


def test_with_gold_keeps_hits_when_gold_is_retrieved():
    hits = [Hit(i, 1.0 - i / 10, None) for i in range(4)]
    out, found = with_gold(hits, hits[2])
    assert found and out == hits


def test_with_gold_replaces_the_fourth_and_resorts():
    hits = [Hit(i, 0.9 - i / 10, None) for i in range(4)]  # 0.9 .. 0.6
    gold = Hit(9, 0.65, None)
    out, found = with_gold(hits, gold)
    assert not found
    assert [h.index for h in out] == [0, 1, 2, 9]
    gold_hi = Hit(9, 0.95, None)
    assert [h.index for h in with_gold(hits, gold_hi)[0]] == [9, 0, 1, 2]


def test_answerable_examples_contain_the_gold_passage(tmp_path):
    idx = index(tmp_path)
    gi = gold_of(idx, "walrus")
    qs = Questions(["What does the python walrus operator do?"], None)
    [ex] = examples_for(idx, "src", gi, qs, use_near_miss=False, removed=set())
    assert ex.kind == "answerable"
    assert ex.passages[ex.gold_rank - 1]["title"] == "The walrus"
    assert ex.gold_retrieved is True
    assert ex.messages()[1]["content"].endswith("Question: What does the python walrus operator do?")


def test_removed_examples_exclude_the_whole_document(tmp_path):
    idx = index(tmp_path)
    gi = gold_of(idx, "walrus")
    qs = Questions(["What does the python walrus operator do?"], None)
    [ex] = examples_for(idx, "src", gi, qs, use_near_miss=False, removed={0})
    assert ex.kind == "removed"
    assert ex.gold_rank is None
    assert all(p["title"] != "The walrus" for p in ex.passages)


def test_near_miss_is_added_only_when_asked(tmp_path):
    idx = index(tmp_path)
    gi = gold_of(idx, "generics")
    qs = Questions(["What do go generics add?"], "When will rust get generics?")
    kinds = [e.kind for e in examples_for(idx, "src", gi, qs, use_near_miss=True, removed=set())]
    assert kinds == ["answerable", "near-miss"]
    assert len(examples_for(idx, "src", gi, qs, use_near_miss=False, removed=set())) == 1


def test_example_ids_are_unique_and_to_dict_has_messages(tmp_path):
    idx = index(tmp_path)
    gi = gold_of(idx, "generics")
    qs = Questions(["What do go generics add?", "How are go generics written?"], "Rust?")
    exs = examples_for(idx, "src", gi, qs, use_near_miss=True, removed={1})
    assert len({e.id for e in exs}) == 3
    d = exs[0].to_dict()
    assert d["messages"][0]["role"] == "system" and d["split"] in {"train", "valid", "test"}
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_examples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'small_rag_finetune.examples'`

- [ ] **Step 3: Implement**

```python
# src/small_rag_finetune/examples.py
"""From a gold chunk and Claude's questions to the exact prompts the student sees.

- answerable: retrieval for the question; if the gold chunk isn't in the top 4
  it replaces the 4th and the list is re-sorted by score, so the answer is not
  always passage [1].
- removed: same question, every chunk of the gold document excluded.
- near-miss: Claude's related question the gold chunk doesn't answer.
"""

from dataclasses import asdict, dataclass

from .claude import Questions
from .corpus import split_of
from .index import Hit, SourceIndex
from .prompt import TOP_K, build_messages


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    doc: str
    split: str
    kind: str  # answerable | removed | near-miss
    question: str
    passages: list[dict]
    gold_rank: int | None  # 1-based position of the gold chunk, if present
    gold_retrieved: bool | None  # answerable only: retrieval found it unaided

    def messages(self) -> list[dict]:
        return build_messages(self.question, self.passages)

    def to_dict(self) -> dict:
        return {**asdict(self), "messages": self.messages()}


def passage(chunk) -> dict:
    return {"title": chunk.title, "heading": chunk.heading, "snippet": chunk.snippet}


def with_gold(hits: list[Hit], gold: Hit) -> tuple[list[Hit], bool]:
    if any(h.index == gold.index for h in hits):
        return hits, True
    kept = [*hits[: TOP_K - 1], gold]
    return sorted(kept, key=lambda h: (-h.score, h.index)), False


def examples_for(
    index: SourceIndex,
    source: str,
    gold_index: int,
    qs: Questions,
    *,
    use_near_miss: bool,
    removed: set[int],
) -> list[Example]:
    gold_chunk = index.chunks[gold_index]
    doc = gold_chunk.post
    split = split_of(source, doc)
    base = f"{source}/{doc}/{gold_index}"

    def make(tag, kind, question, hits, rank, found) -> Example:
        return Example(
            f"{base}/{tag}", source, doc, split, kind, question,
            [passage(h.chunk) for h in hits], rank, found,
        )

    out = []
    for j, q in enumerate(qs.answerable):
        if j in removed:
            out.append(make(j, "removed", q, index.top(q, exclude_post=doc), None, None))
            continue
        gold = Hit(gold_index, float(index.scores(q)[gold_index]), gold_chunk)
        hits, found = with_gold(index.top(q), gold)
        rank = next(i + 1 for i, h in enumerate(hits) if h.index == gold_index)
        out.append(make(j, "answerable", q, hits, rank, found))
    if use_near_miss and qs.near_miss:
        out.append(make("near", "near-miss", qs.near_miss, index.top(qs.near_miss), None, None))
    return out
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_examples.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/small_rag_finetune/examples.py tests/test_examples.py
git commit -m "Build answerable, removed and near-miss prompts around a gold chunk"
```

---

### Task 9: Pilot orchestration, report and CLI

**Files:**
- Create: `src/small_rag_finetune/pilot.py`
- Modify: `src/small_rag_finetune/__init__.py` (add `pilot` command)
- Test: `tests/test_pilot.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pilot.py
import json

from fakes import FakeClient, FakeModel, write_post

from small_rag_finetune.claude import TEACHERS
from small_rag_finetune.pilot import run_pilot

TOPICS = ["generics", "async", "walrus", "lifetimes", "modules", "go", "rust", "python"]


def corpus(root):
    for i, t in enumerate(TOPICS):
        write_post(root / "src", f"{t}-{i}", f"About {t}", f"All about {t} in practice. " * 20)


def test_pilot_writes_examples_calls_and_report(tmp_path):
    corpus(tmp_path / "corpus")
    client = FakeClient()
    rows, calls = run_pilot(
        tmp_path / "corpus", tmp_path / "out", ["src"], 3,
        client=client, model=FakeModel(), workers=2,
    )
    assert rows
    assert all(r["target_opus"] and r["target_sonnet"] for r in rows)
    assert all(r["split"] == "train" for r in rows)
    n_questions = sum(c.role == "questions" for c in calls)
    assert n_questions == 3
    assert sum(c.role == "teacher" for c in calls) == len(rows) * len(TEACHERS)
    lines = (tmp_path / "out" / "examples.jsonl").read_text().splitlines()
    assert len(lines) == len(rows) and "messages" in json.loads(lines[0])
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Projected for 5,000 examples" in report
    assert "claude-opus-5-5" in report and "claude-sonnet-5-5" in report


def test_pilot_is_deterministic(tmp_path):
    corpus(tmp_path / "corpus")
    a, _ = run_pilot(tmp_path / "corpus", tmp_path / "a", ["src"], 3, client=FakeClient(), model=FakeModel(), workers=2)
    b, _ = run_pilot(tmp_path / "corpus", tmp_path / "b", ["src"], 3, client=FakeClient(), model=FakeModel(), workers=2)
    assert [r["id"] for r in a] == [r["id"] for r in b]
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_pilot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'small_rag_finetune.pilot'`

- [ ] **Step 3: Implement**

```python
# src/small_rag_finetune/pilot.py
"""The pilot: a few documents per source through the whole pipeline, both
teachers on every example, and a report with measured cost per example."""

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from sss_eval.chunk import split_sentences

from .claude import (
    BATCH_DISCOUNT,
    QUESTION_MODEL,
    TEACHERS,
    Call,
    complete,
    cost,
    normalize_answer,
    parse_questions,
    question_params,
    target_key,
    teacher_params,
)
from .corpus import split_of
from .examples import examples_for
from .index import SourceIndex
from .prompt import REFUSAL

MIN_SNIPPET = 300  # gold chunks shorter than this rarely support 3 questions
REMOVED_RATE = 0.1  # share of answerable questions turned into "removed"
NEAR_MISS_RATE = 0.5  # share of gold chunks that also get their near-miss question
TARGET_EXAMPLES = 5000


def pick_gold(index: SourceIndex, source: str, n: int, rng: random.Random) -> list[int]:
    """One gold chunk from each of n random train-split documents."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(index.chunks):
        if len(c.snippet) >= MIN_SNIPPET and split_of(source, c.post) == "train":
            by_doc[c.post].append(i)
    docs = rng.sample(sorted(by_doc), min(n, len(by_doc)))
    return [rng.choice(by_doc[d]) for d in docs]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_pilot(
    corpus_root: Path,
    out_dir: Path,
    sources: list[str],
    per_source: int,
    *,
    seed: int = 0,
    client=None,
    model=None,
    workers: int = 8,
) -> tuple[list[dict], list[Call]]:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    rng = random.Random(seed)
    calls: list[Call] = []
    examples = []
    for name in sources:
        index = SourceIndex.build(corpus_root / name, model=model)
        for gi in pick_gold(index, name, per_source, rng):
            text, usage = complete(client, question_params(index.chunks[gi]))
            calls.append(Call("questions", QUESTION_MODEL, usage.input_tokens, usage.output_tokens))
            qs = parse_questions(text)
            removed = {j for j in range(len(qs.answerable)) if rng.random() < REMOVED_RATE}
            near = rng.random() < NEAR_MISS_RATE
            examples += examples_for(index, name, gi, qs, use_near_miss=near, removed=removed)

    def teach(job):
        ex, m = job
        text, usage = complete(client, teacher_params(m, ex.messages()))
        return ex.id, m, normalize_answer(text), Call("teacher", m, usage.input_tokens, usage.output_tokens)

    targets: dict[str, dict] = defaultdict(dict)
    with ThreadPoolExecutor(workers) as pool:
        for ex_id, m, answer, call in pool.map(teach, [(e, m) for e in examples for m in TEACHERS]):
            targets[ex_id][target_key(m)] = answer
            calls.append(call)

    rows = [{**e.to_dict(), **targets[e.id]} for e in examples]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "examples.jsonl", rows)
    write_jsonl(out_dir / "calls.jsonl", [asdict(c) for c in calls])
    (out_dir / "report.md").write_text(pilot_report(rows, calls))
    return rows, calls


def _refused(row: dict, model: str) -> bool:
    return row[target_key(model)] == REFUSAL


def _frac(rows: list[dict], pred) -> str:
    return f"{sum(1 for r in rows if pred(r))}/{len(rows)}"


def _teacher_line(rows: list[dict], model: str) -> str:
    def of(kind):
        return [r for r in rows if r["kind"] == kind]

    def answered(r):
        return not _refused(r, model)

    answers = [r[target_key(model)] for r in rows if answered(r)]
    over = sum(len(split_sentences(a)) > 2 for a in answers)
    mean = sum(map(len, answers)) / max(len(answers), 1)
    return (
        f"| {model} | {_frac(of('answerable'), lambda r: _refused(r, model))} "
        f"| {_frac(of('removed'), answered)} | {_frac(of('near-miss'), answered)} "
        f"| {over}/{len(answers)} | {mean:.0f} |"
    )


def _review_samples(rows: list[dict]) -> list[dict]:
    picks = {"answerable": 6, "removed": 2, "near-miss": 2}
    return [r for kind, n in picks.items() for r in [x for x in rows if x["kind"] == kind][:n]]


def pilot_report(rows: list[dict], calls: list[Call]) -> str:
    kinds = Counter(r["kind"] for r in rows)
    out = ["# Pilot report", ""]
    out.append(f"Examples: {len(rows)} (" + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")")
    ans = [r for r in rows if r["kind"] == "answerable"]
    if ans:
        out.append(
            f"Gold chunk retrieved unaided: {_frac(ans, lambda r: r['gold_retrieved'])}; "
            f"gold at rank 1: {_frac(ans, lambda r: r['gold_rank'] == 1)}"
        )

    out += [
        "", "## Teachers", "",
        "| teacher | answerable refused | removed answered | near-miss answered | answers over 2 sentences | mean answer chars |",
        "|---|---|---|---|---|---|",
    ]
    out += [_teacher_line(rows, m) for m in TEACHERS]
    agree = sum(_refused(r, TEACHERS[0]) == _refused(r, TEACHERS[1]) for r in rows)
    out += ["", f"Teachers agree on answer vs refuse: {agree}/{len(rows)}"]

    out += [
        "", "## Cost (pilot, standard rates)", "",
        "| role | model | calls | input tokens | output tokens | USD |",
        "|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str], list[Call]] = defaultdict(list)
    for c in calls:
        groups[(c.role, c.model)].append(c)
    for (role, model), cs in sorted(groups.items()):
        i = sum(c.input_tokens for c in cs)
        o = sum(c.output_tokens for c in cs)
        out.append(f"| {role} | {model} | {len(cs)} | {i:,} | {o:,} | {cost(model, i, o):.3f} |")

    def spent(role, model=None):
        return sum(
            cost(c.model, c.input_tokens, c.output_tokens)
            for c in calls
            if c.role == role and (model is None or c.model == model)
        )

    q_cost = spent("questions")
    out += ["", f"Projected for {TARGET_EXAMPLES:,} examples with the Batch API (50% off):", ""]
    for model in TEACHERS:
        per = (q_cost + spent("teacher", model)) / max(len(rows), 1)
        total = per * TARGET_EXAMPLES * BATCH_DISCOUNT
        out.append(f"- questions + teacher {model}: ${per:.4f} per example at standard rates -> ${total:.2f}")

    out += ["", "## Examples for review", ""]
    for r in _review_samples(rows):
        out.append(f"### {r['kind']}: {r['question']}")
        out.append("")
        for i, p in enumerate(r["passages"], 1):
            mark = " (gold)" if r["gold_rank"] == i else ""
            heading = f" — {p['heading']}" if p["heading"] else ""
            out.append(f"{i}. {p['title']}{heading}{mark}")
        if r["gold_rank"]:
            out += ["", "> " + r["passages"][r["gold_rank"] - 1]["snippet"]]
        out.append("")
        out += [f"- **{m}**: {r[target_key(m)]}" for m in TEACHERS]
        out.append("")
    return "\n".join(out) + "\n"
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `unset VIRTUAL_ENV; uv run --no-sync pytest tests/test_pilot.py -v`
Expected: 2 passed

- [ ] **Step 5: Add the CLI command**

Append to `src/small_rag_finetune/__init__.py`:

```python
@main.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("data/corpus"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/pilot"))
@click.option("--per-source", default=5, show_default=True, help="Documents per source.")
@click.option("--seed", default=0, show_default=True)
def pilot(corpus: Path, out: Path, per_source: int, seed: int) -> None:
    """Questions + both teachers for a few documents; writes a cost report."""
    from .pilot import run_pilot
    from .sources import SOURCES

    rows, calls = run_pilot(corpus, out, list(SOURCES), per_source, seed=seed)
    click.echo(f"{len(rows)} examples, {len(calls)} calls -> {out}/report.md")
```

- [ ] **Step 6: Run the whole suite, format, lint**

Run: `unset VIRTUAL_ENV; uv run --no-sync black src tests && uv run --no-sync ruff check src tests && uv run --no-sync pytest -v`
Expected: black/ruff clean; all tests pass (2 parity + 11 normalize + 4 corpus + 5 index + 8 claude + 6 examples + 2 pilot = 38).

- [ ] **Step 7: Commit**

```bash
git add src/small_rag_finetune/pilot.py src/small_rag_finetune/__init__.py tests/test_pilot.py
git commit -m "Pilot: questions and both teachers for a few documents, cost report"
```

---

### Task 10: Run the pilot (spends money: ask first)

Expected spend at standard rates, from rough token estimates: about $1-3 (20 question calls on Sonnet; about 70 examples × Opus and Sonnet teacher calls). The report states the measured figure.

- [ ] **Step 1: Ask the user to confirm the pilot spend**

Say what runs (20 documents, 5 per source; ~90 API calls) and the estimate above. Wait for a yes.

- [ ] **Step 2: Run**

Run: `cd ~/projects/small-rag-finetune && zsh -ic 'unset VIRTUAL_ENV; uv run --no-sync small-rag-finetune pilot'`
Expected: `NN examples, MMM calls -> data/pilot/report.md` with NN between 50 and 80.
If a request fails with a 400 on `output_config`, remove the `output_config` entries from `question_params`/`teacher_params`, note it in the report, and re-run.

- [ ] **Step 3: Read the report and the raw examples**

Run: `glow data/pilot/report.md` (or `bat data/pilot/report.md`)
Check by hand, and write findings at the end of `report.md` under `## Findings`:
- questions are self-contained and answerable from the gold passage;
- near-miss questions are really unanswered by the passages;
- teacher answers contain no "passage [n]" / "according to", are ≤ 2 sentences, and refuse exactly with `The blog does not seem to cover this.`;
- how Opus and Sonnet differ on the same examples (the table and 10 samples).

- [ ] **Step 4: Report to the user**

Give: examples per kind, gold retrieval rate, teacher table, measured cost per example and the projection for 5,000 (Opus vs Sonnet teacher, batch), 3-4 representative examples, and a recommendation for the teacher model and any prompt changes. Plan 2 starts after the user picks.

- [ ] **Step 5: Commit the findings (not the data)**

`data/` is gitignored, so copy the report into the docs:

```bash
cp data/pilot/report.md docs/superpowers/specs/2026-10-15-pilot-report.md
git add docs/superpowers/specs/2026-10-15-pilot-report.md
git commit -m "Pilot report: cost per example and teacher comparison"
```
