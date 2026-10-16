from pathlib import Path

import click


@click.group()
def main() -> None:
    """Fine-tune small LMs to answer from retrieved passages, or refuse."""


@main.command()
@click.option("--answerable", default=750, show_default=True)
@click.option("--refusal", default=250, show_default=True)
@click.option("--valid", default=100, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/smoke"))
def build_data(answerable: int, refusal: int, valid: int, seed: int, out: Path) -> None:
    """Reshape neural-bridge's open RAG data into the ask-my-blog prompt."""
    from .data import build, write_splits

    examples = build(answerable, refusal, seed)
    write_splits(examples, out, valid)
    click.echo(f"{len(examples)} examples -> {out}/ ({valid} valid)")


@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=Path("data/sources"))
def fetch_sources(root: Path) -> None:
    """Check out every source at its pinned commit."""
    from .sources import SOURCES, fetch

    for src in SOURCES.values():
        click.echo(f"{src.name}: {fetch(src, root)}")


@main.command()
@click.option(
    "--sources", type=click.Path(path_type=Path), default=Path("data/sources")
)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/corpus"))
def build_corpus(sources: Path, out: Path) -> None:
    """Normalize every source into markdown posts the blog's chunker reads."""
    from .corpus import write_corpus
    from .sources import SOURCES

    for src in SOURCES.values():
        n = write_corpus(src, sources / src.name, out)
        click.echo(f"{src.name}: {n} documents -> {out / src.name}")
