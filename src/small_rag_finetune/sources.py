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
        _git(
            "clone",
            "--quiet",
            "--filter=blob:none",
            "--no-checkout",
            src.repo,
            str(dest),
        )
    _git("-C", str(dest), "sparse-checkout", "set", src.sparse)
    _git("-C", str(dest), "checkout", "--quiet", src.commit)
    return dest
