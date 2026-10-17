import json
from pathlib import Path

import click


@click.group()
def main() -> None:
    """Fine-tune small LMs to answer from retrieved passages, or refuse."""


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


@main.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("data/corpus"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/pilot"))
@click.option(
    "--per-source", default=5, show_default=True, help="Documents per source."
)
@click.option("--seed", default=0, show_default=True)
def pilot(corpus: Path, out: Path, per_source: int, seed: int) -> None:
    """Questions + both teachers for a few documents; writes a cost report."""
    from .pilot import run_pilot
    from .sources import SOURCES

    rows, calls = run_pilot(corpus, out, list(SOURCES), per_source, seed=seed)
    click.echo(f"{len(rows)} examples, {len(calls)} calls -> {out}/report.md")


@main.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("data/corpus"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/v1"))
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--style",
    type=click.Choice(["v1", "extractive"]),
    default="v1",
    show_default=True,
    help="Teacher answer style (claude.py: TEACHER_GUIDANCE, EXTRACTIVE_GUIDANCE).",
)
@click.option("--sample", type=int, default=None, help="Answer only N random examples.")
@click.option(
    "--questions-from",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Reuse another dataset's question batch (e.g. data/v1) instead of paying again.",
)
def generate(
    corpus: Path,
    out: Path,
    seed: int,
    style: str,
    sample: int | None,
    questions_from: Path | None,
) -> None:
    """Full dataset via the Message Batches API (resumable)."""
    import shutil

    from .claude import EXTRACTIVE_GUIDANCE, TEACHER_GUIDANCE
    from .generate import QUOTAS, run_generate
    from .sources import SOURCES

    if questions_from is not None:
        dest = out / "batches" / "questions" / "results.jsonl"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                questions_from / "batches" / "questions" / "results.jsonl", dest
            )
    guidance = EXTRACTIVE_GUIDANCE if style == "extractive" else TEACHER_GUIDANCE
    summary = run_generate(
        corpus, out, list(SOURCES), QUOTAS, seed=seed, guidance=guidance, sample=sample
    )
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.argument("log", type=click.Path(exists=True, path_type=Path))
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def best_checkpoint(log: Path, run_dir: Path) -> None:
    """Pick the lowest-validation-loss checkpoint into <run_dir>-best/.

    How v1 and v2 were picked; from v3 on, dev-judge picks on judged answers,
    because validation loss kept falling while the answers got worse."""
    from .checkpoints import best_checkpoint as pick
    from .checkpoints import val_losses

    it, out = pick(log, run_dir)
    click.echo(f"val losses: {val_losses(log.read_text())}")
    click.echo(f"best: iteration {it} -> {out}")


@main.command()
@click.argument("log", type=click.Path(exists=True, path_type=Path))
@click.argument("out", type=click.Path(path_type=Path))
@click.option(
    "--name", required=True, help="Model name for the title, e.g. SmolLM2-360M"
)
@click.option(
    "--epoch-iters", default=2264.5, show_default=True, help="Iterations per epoch"
)
def loss_chart(log: Path, out: Path, name: str, epoch_iters: float) -> None:
    """Draw a run's training and validation loss as a blog-style SVG."""
    from .charts import loss_chart as draw
    from .checkpoints import train_losses, val_losses

    text = log.read_text()
    uid = "viz-loss-" + "".join(c for c in name.lower() if c.isalnum())
    svg = draw(
        name, train_losses(text), val_losses(text, baseline=True), epoch_iters, uid
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg + "\n")
    click.echo(f"wrote {out}")


@main.command()
@click.option(
    "--src", type=click.Path(exists=True, path_type=Path), default=Path("data/v1")
)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/v2"))
def strip_hedges(src: Path, out: Path) -> None:
    """Copy a dataset without partial-answer hedges ("the blog does not ...")."""
    from .hedges import strip_split

    summary = {
        split: strip_split(src, out, split) for split in ("train", "valid", "test")
    }
    (out / "summary.json").write_text(
        json.dumps({"source": str(src), **summary}, indent=2)
    )
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.argument("data", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out", type=click.Path(path_type=Path))
@click.option("--split", default="test", show_default=True)
def dev_prompts(data: Path, out: Path, split: str) -> None:
    """Held-out prompts for picking checkpoints (input for js/generate.mjs)."""
    from .devjudge import dev_prompts as build

    rows = build(data, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    click.echo(f"wrote {len(rows)} prompts to {out}")


@main.command()
@click.argument("answers", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--prompts",
    type=click.Path(exists=True, path_type=Path),
    default=Path("runs/dev/prompts.jsonl"),
)
@click.option("--out", type=click.Path(path_type=Path), default=Path("runs/dev/judge"))
def dev_judge(answers: Path, prompts: Path, out: Path) -> None:
    """Grade one arm's answers to the dev prompts with Opus (Batch API)."""
    import anthropic

    from .devjudge import run_dev_judge

    rows = [json.loads(x) for x in prompts.read_text().splitlines() if x.strip()]
    summary = run_dev_judge(anthropic.Anthropic(), rows, answers, out)
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("data/corpus"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/v4-broad"))
@click.option("--seed", default=0, show_default=True)
@click.option("--sample", type=int, default=None, help="Answer only N random examples.")
def generate_broad(corpus: Path, out: Path, seed: int, sample: int | None) -> None:
    """v4's broad questions (topic and goal, per document) via Message Batches."""
    from .broad import run_broad
    from .sources import SOURCES

    summary = run_broad(corpus, out, list(SOURCES), seed=seed, sample=sample)
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.argument(
    "sources",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def combine(sources: tuple[Path, ...], out: Path) -> None:
    """Concatenate datasets (e.g. data/v3 data/v4-broad) into one."""
    from .broad import combine as run

    click.echo(json.dumps(run(list(sources), out)))


@main.command("hf-export")
@click.argument("out", type=click.Path(path_type=Path))
@click.option(
    "--root", type=click.Path(path_type=Path), default=Path("."), show_default=True
)
def hf_export(out: Path, root: Path) -> None:
    """Write the Hugging Face dataset (v3, broad, v1 configs + card) to OUT."""
    from .publish import export

    click.echo(json.dumps(export(root, out)))


@main.command("hf-stage-model")
@click.argument("fused", type=click.Path(exists=True, path_type=Path))
@click.argument("onnx", type=click.Path(exists=True, path_type=Path))
@click.argument("out", type=click.Path(path_type=Path))
def hf_stage_model(fused: Path, onnx: Path, out: Path) -> None:
    """Lay out a model repo: FUSED weights + ONNX export + card, in OUT."""
    from .publish import stage_model

    click.echo("\n".join(stage_model(fused, onnx, out)))


@main.command("eval-prompts")
@click.argument("out", type=click.Path(path_type=Path))
@click.option(
    "--corpus",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Folder of markdown posts with front matter (e.g. Hugo's content/post).",
)
@click.option(
    "--questions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help='JSONL: {"id", "question", "covered": true|false}.',
)
def eval_prompts(out: Path, corpus: Path, questions: Path) -> None:
    """Prompts for your own site's questions (input for generate.mjs, dev-judge)."""
    from .index import SourceIndex
    from .sitecheck import read_questions, site_prompts

    rows = site_prompts(SourceIndex.build(corpus), read_questions(questions))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    click.echo(f"{len(rows)} prompts -> {out}")


@main.command("post-charts")
@click.argument("out", type=click.Path(path_type=Path))
@click.option(
    "--judgements",
    type=click.Path(exists=True, path_type=Path),
    default=Path("runs/eval/judgements.jsonl"),
    show_default=True,
)
@click.option(
    "--v2-log",
    type=click.Path(exists=True, path_type=Path),
    default=Path("logs/v2-qwen3-0.6b.log"),
    show_default=True,
)
def post_charts(out: Path, judgements: Path, v2_log: Path) -> None:
    """The fine-tuning post's charts, from the blog-eval judgements and v2's log."""
    from collections import Counter

    from .charts import loss_vs_answers_chart, versions_chart
    from .checkpoints import val_losses

    counts: dict[str, Counter] = {}
    for line in judgements.read_text().splitlines():
        j = json.loads(line)
        neg = j["family"] == "negative"
        key = "invented" if neg and j["label"] != "faithful" else None
        if not neg:
            key = {"faithful": "faithful", "unsupported": "wrong"}.get(j["label"])
        counts.setdefault(j["arm"], Counter())[key] += 1

    def row(label: str, arm: str, shipped: bool = False) -> dict:
        c = counts[arm]
        return {
            "label": label,
            "shipped": shipped,
            **{k: c[k] for k in ("faithful", "wrong", "invented")},
        }

    rows = [
        row("Qwen3-0.6B", "genai-base-qwen3-0.6b-rp1"),
        row("v1: dense answers", "v1-qwen3-0.6b-rp1"),
        row("v2: without hedges", "v2-qwen3-0.6b-rp1"),
        row("v3: one short extract", "v3-qwen3-0.6b-rp1", shipped=True),
        row("v4: v3 + broad questions", "v4-qwen3-0.6b-rp1"),
    ]
    epoch = 2240
    val = {it / epoch: v for it, v in val_losses(v2_log.read_text()).items() if it > 1}
    faithful = {
        0.25: counts["v2-qwen3-0.6b-it560-rp1"]["faithful"],
        0.5: counts["v2-qwen3-0.6b-it1120-rp1"]["faithful"],
        1.0: counts["v2-qwen3-0.6b-rp1"]["faithful"],
    }
    base = counts["genai-base-qwen3-0.6b-rp1"]["faithful"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "versions.svg").write_text(versions_chart(rows) + "\n")
    (out / "loss-vs-answers.svg").write_text(
        loss_vs_answers_chart(val, faithful, base) + "\n"
    )
    click.echo(json.dumps({"rows": rows, "val": val, "faithful": faithful}))
