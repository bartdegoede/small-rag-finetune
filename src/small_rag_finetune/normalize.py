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
    body = rst_to_markdown("\n".join(rest))
    return Doc(f"pep-{num:04d}", f"PEP {num}: {header['Title']}", body)
