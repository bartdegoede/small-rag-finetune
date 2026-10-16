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
