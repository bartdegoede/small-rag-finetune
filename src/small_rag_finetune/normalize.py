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
