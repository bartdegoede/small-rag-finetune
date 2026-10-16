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
