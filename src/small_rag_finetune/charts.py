"""Loss curves as theme-aware inline SVG, in the style of the blog's charts.

The SVG only carries `viz-*` classes; colours come from the blog's
`assets/css/extended/charts.css`, so the chart follows PaperMod's dark toggle.
Copy the output to the blog's `assets/svg/<post stem>/` and embed it with
`{{< svg "name.svg" "caption" >}}`. Hover = native SVG <title>.
"""

import math

W, H = 680, 320
LEFT, RIGHT, TOP, BOTTOM = 56, 24, 40, 52
STEPS = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def smooth(values: list[float], window: int = 9) -> list[float]:
    """Centred moving average; the window shrinks at the ends."""
    half = window // 2
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - half) : i + half + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def nice_ticks(lo: float, hi: float, max_ticks: int = 6) -> list[float]:
    """Round tick values covering [lo, hi] with at most max_ticks ticks."""
    step = next(s for s in STEPS if (hi - lo) / s <= max_ticks - 1)
    start = math.floor(lo / step + 1e-9) * step
    stop = math.ceil(hi / step - 1e-9) * step
    n = round((stop - start) / step)
    return [round(start + k * step, 4) for k in range(n + 1)]


def _path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points)
    )


def _epoch(it: int, epoch_iters: float) -> str:
    return f"{it / epoch_iters:.2f}".rstrip("0").rstrip(".")


def loss_chart(
    name: str,
    train: dict[int, float],
    val: dict[int, float],
    epoch_iters: float,
    uid: str = "viz-loss",
) -> str:
    """Training loss (raw, faint, plus smoothed) and validation loss per
    checkpoint; the lowest is ringed as the checkpoint used. `val` may include
    iteration 1, the untrained baseline: it is named in the legend, not
    plotted, because it would squash the rest of the chart."""
    its = sorted(train)
    before = val.get(1)
    vits = sorted(i for i in val if i > 1)
    best = min(vits, key=lambda i: val[i])
    # The log's last line can be an iteration past the final epoch boundary.
    epochs = math.ceil(max(its[-1], vits[-1]) / epoch_iters - 0.01)
    x_max = epochs * epoch_iters
    shown = [*train.values(), *(val[i] for i in vits)]
    ticks = nice_ticks(min(shown), max(shown))

    def xmap(it: float) -> float:
        return LEFT + it / x_max * (W - LEFT - RIGHT)

    def ymap(v: float) -> float:
        return TOP + (ticks[-1] - v) / (ticks[-1] - ticks[0]) * (H - TOP - BOTTOM)

    title = f"{name}: training and validation loss"
    start = f"from {before:.3f} before training " if before else ""
    desc = (
        f"Loss over {epochs} epochs of fine-tuning. Validation loss falls {start}"
        f"to a low of {val[best]:.3f} at epoch {_epoch(best, epoch_iters)}, the "
        f"checkpoint used, and ends at {val[vits[-1]]:.3f}. Training loss keeps "
        f"falling, most steeply where the second epoch starts."
    )
    out = [
        (
            f'<svg class="viz" viewBox="0 0 {W} {H}" role="img" '
            f'aria-labelledby="{uid}-t {uid}-d" xmlns="http://www.w3.org/2000/svg">'
        ),
        f'<title id="{uid}-t">{esc(title)}</title><desc id="{uid}-d">{esc(desc)}</desc>',
    ]
    # Legend.
    out.append(
        f'<line class="viz-line viz-s1" x1="{LEFT}" x2="{LEFT + 20}" y1="14" y2="14"/>'
    )
    out.append(
        f'<text class="viz-label" x="{LEFT + 28}" y="18">training (smoothed)</text>'
    )
    lx = LEFT + 190
    out.append(f'<circle class="viz-dot viz-s2" cx="{lx + 5}" cy="14" r="5"/>')
    vlabel = "validation"
    if before:
        vlabel += f' <tspan class="viz-sub">({before:.2f} before training)</tspan>'
    out.append(f'<text class="viz-label" x="{lx + 16}" y="18">{vlabel}</text>')
    lx += 250 if before else 110
    out.append(f'<circle class="viz-ring viz-s2" cx="{lx + 8}" cy="14" r="8"/>')
    out.append(f'<text class="viz-label" x="{lx + 22}" y="18">checkpoint used</text>')
    # Axes.
    for v in ticks:
        y = ymap(v)
        out.append(
            f'<line class="viz-grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="viz-tick" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{v:g}</text>'
        )
    mid = (ymap(ticks[0]) + ymap(ticks[-1])) / 2
    out.append(
        f'<text class="viz-axis" transform="translate(14 {mid:.1f}) rotate(-90)" '
        f'text-anchor="middle">Loss</text>'
    )
    for k in range(2 * epochs + 1):
        e = k / 2
        out.append(
            f'<text class="viz-tick" x="{xmap(e * epoch_iters):.1f}" y="{H - BOTTOM + 18}" '
            f'text-anchor="middle">{e:g}</text>'
        )
    out.append(
        f'<text class="viz-axis" x="{(LEFT + W - RIGHT) / 2}" y="{H - 12}" '
        f'text-anchor="middle">Epoch</text>'
    )
    out.append(
        f'<line class="viz-baseline" x1="{LEFT}" x2="{W - RIGHT}" '
        f'y1="{ymap(ticks[0]):.1f}" y2="{ymap(ticks[0]):.1f}"/>'
    )
    # Epoch boundaries: from here on the model sees each example again.
    for e in range(1, epochs):
        x = xmap(e * epoch_iters)
        out.append(
            f'<line class="viz-sep" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP}" y2="{H - BOTTOM}"/>'
        )
    if epochs > 1:
        out.append(
            f'<text class="viz-note" x="{xmap(epoch_iters) + 8:.1f}" y="{TOP + 14}">'
            f"second pass over the same examples</text>"
        )
    # Training loss: raw reports, faint, under a moving average.
    raw = [train[i] for i in its]
    out.append(
        f'<path class="viz-line viz-s1 viz-faint" d="{_path([(xmap(i), ymap(v)) for i, v in zip(its, raw)])}"/>'
    )
    out.append(
        f'<path class="viz-line viz-s1" d="{_path([(xmap(i), ymap(v)) for i, v in zip(its, smooth(raw))])}"/>'
    )
    # Validation loss per checkpoint.
    out.append(
        f'<path class="viz-line viz-s2" d="{_path([(xmap(i), ymap(val[i])) for i in vits])}"/>'
    )
    for i in vits:
        x, y = xmap(i), ymap(val[i])
        tip = f"Epoch {_epoch(i, epoch_iters)} (iteration {i}): validation loss {val[i]:.3f}"
        if i == best:
            tip += ", the lowest: this checkpoint is used"
            out.append(
                f'<circle class="viz-ring viz-s2" cx="{x:.1f}" cy="{y:.1f}" r="10"/>'
            )
        out.append(f"<g><title>{esc(tip)}</title>")
        out.append(f'<circle class="viz-hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>')
        out.append(
            f'<circle class="viz-dot viz-s2" cx="{x:.1f}" cy="{y:.1f}" r="5"/></g>'
        )
    bx, by = xmap(best), ymap(val[best])
    out.append(
        f'<text class="viz-label" x="{bx:.1f}" y="{by - 16:.1f}" text-anchor="middle">'
        f"{val[best]:.3f}</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def _svg_open(uid: str, h: int, title: str, desc: str) -> list[str]:
    return [
        (
            f'<svg class="viz" viewBox="0 0 {W} {h}" role="img" '
            f'aria-labelledby="{uid}-t {uid}-d" xmlns="http://www.w3.org/2000/svg">'
        ),
        f'<title id="{uid}-t">{esc(title)}</title><desc id="{uid}-d">{esc(desc)}</desc>',
    ]


def _dot(cls: str, x: float, y: float, tip: str) -> list[str]:
    return [
        f"<g><title>{esc(tip)}</title>",
        f'<circle class="viz-hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>',
        f'<circle class="viz-dot {cls}" cx="{x:.1f}" cy="{y:.1f}" r="6"/></g>',
    ]


def versions_chart(rows: list[dict], uid: str = "viz-versions") -> str:
    """One row per model version: faithful answers (of 30) and wrong ones
    (unsupported, of the same 30), on one 0-30 scale. rows: {label, faithful,
    wrong, invented, n, shipped}."""
    left, right, top, row_h = 210, 30, 52, 44
    h = top + row_h * len(rows) + 46

    def xmap(v: float) -> float:
        return left + v / 30 * (W - left - right)

    desc = "; ".join(
        f"{r['label']}: {r['faithful']} faithful, {r['wrong']} wrong" for r in rows
    )
    out = _svg_open(
        uid,
        h,
        "Faithful and wrong answers per fine-tune",
        f"Of the blog's 30 answerable questions. {desc}.",
    )
    out += [
        f'<circle class="viz-dot viz-s1" cx="{left + 5}" cy="14" r="5"/>',
        f'<text class="viz-label" x="{left + 16}" y="18">faithful</text>',
        f'<circle class="viz-dot viz-s2" cx="{left + 105}" cy="14" r="5"/>',
        (
            f'<text class="viz-label" x="{left + 116}" y="18">wrong (a claim the'
            " passages don't support)</text>"
        ),
    ]
    bottom = top + row_h * len(rows) - row_h / 2 + 14
    for v in range(0, 31, 5):
        x = xmap(v)
        out.append(
            f'<line class="viz-grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 14}" y2="{bottom:.1f}"/>'
        )
        out.append(
            f'<text class="viz-tick" x="{x:.1f}" y="{bottom + 18:.1f}" text-anchor="middle">{v}</text>'
        )
    out.append(
        f'<text class="viz-axis" x="{(left + W - right) / 2:.1f}" y="{h - 8}" '
        f'text-anchor="middle">Answers, of 30 the blog covers</text>'
    )
    for k, r in enumerate(rows):
        y = top + k * row_h
        out.append(
            f'<text class="viz-label" x="{left - 14}" y="{y + 5:.1f}" text-anchor="end">'
            f"{esc(r['label'])}</text>"
        )
        xa, xb = sorted([xmap(r["faithful"]), xmap(r["wrong"])])
        out.append(
            f'<line class="viz-grid" x1="{xa:.1f}" x2="{xb:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        extra = (
            f"; invented an answer for {r['invented']} of 15 questions the blog"
            " doesn't cover"
        )
        if r.get("shipped"):
            out.append(
                f'<circle class="viz-ring viz-s1" cx="{xmap(r["faithful"]):.1f}" '
                f'cy="{y:.1f}" r="11"/>'
            )
        out += _dot(
            "viz-s1",
            xmap(r["faithful"]),
            y,
            f"{r['label']}: {r['faithful']} of 30 faithful{extra}",
        )
        out += _dot(
            "viz-s2",
            xmap(r["wrong"]),
            y,
            f"{r['label']}: {r['wrong']} of 30 wrong{extra}",
        )
    out.append("</svg>")
    return "\n".join(out)


def loss_vs_answers_chart(
    val: dict[float, float],
    faithful: dict[float, int],
    base: int,
    uid: str = "viz-loss-answers",
) -> str:
    """Two panels over the same epochs: validation loss falling (top) while
    faithful answers on the blog fall too (bottom). Keys are epochs."""
    left, right, h = 64, 150, 420
    panels = [(40, 170), (230, 360)]  # (top, bottom) pixel rows

    def xmap(e: float) -> float:
        return left + e * (W - left - right)

    lo, hi = min(val.values()), max(val.values())
    pad = (hi - lo) * 0.3
    vt = (round(lo - pad, 2), round(hi + pad, 2))

    def y_loss(v: float) -> float:
        t, b = panels[0]
        return b - (v - vt[0]) / (vt[1] - vt[0]) * (b - t)

    def y_ans(v: float) -> float:
        t, b = panels[1]
        return b - v / 30 * (b - t)

    epochs = sorted(val)
    fe = sorted(faithful)
    desc = (
        "Validation loss per checkpoint: "
        + ", ".join(f"{val[e]:.3f}" for e in epochs)
        + ". Faithful answers on the blog (of 30): "
        + ", ".join(str(faithful[e]) for e in fe)
        + f"; {base} before fine-tuning."
    )
    out = _svg_open(
        uid, h, "Validation loss improves while the answers get worse", desc
    )
    # Panel titles and axes.
    out.append(
        f'<text class="viz-label" x="{left}" y="{panels[0][0] - 16}">Validation loss'
        ' <tspan class="viz-sub" dx="6">(lower is better)</tspan></text>'
    )
    out.append(
        f'<text class="viz-label" x="{left}" y="{panels[1][0] - 16}">Faithful answers'
        ' on the blog <tspan class="viz-sub" dx="6">(of 30, higher is better)</tspan></text>'
    )
    for v in nice_ticks(*vt, max_ticks=4):
        if vt[0] <= v <= vt[1]:
            y = y_loss(v)
            out.append(
                f'<line class="viz-grid" x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
            )
            out.append(
                f'<text class="viz-tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{v:g}</text>'
            )
    for v in (0, 10, 20, 30):
        y = y_ans(v)
        out.append(
            f'<line class="viz-grid" x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="viz-tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{v}</text>'
        )
    yb = y_ans(base)
    out.append(
        f'<line class="viz-sep" x1="{left}" x2="{W - right}" y1="{yb:.1f}" y2="{yb:.1f}"/>'
    )
    out.append(
        f'<text class="viz-note" x="{W - right + 8}" y="{yb + 4:.1f}">before fine-tuning: {base}</text>'
    )
    for e in (0.25, 0.5, 0.75, 1.0):
        out.append(
            f'<text class="viz-tick" x="{xmap(e):.1f}" y="{panels[1][1] + 20}" '
            f'text-anchor="middle">{e:g}</text>'
        )
    out.append(
        f'<text class="viz-axis" x="{(left + W - right) / 2:.1f}" y="{h - 10}" '
        f'text-anchor="middle">Epochs of training</text>'
    )
    out.append(
        f'<path class="viz-line viz-s1" d="{_path([(xmap(e), y_loss(val[e])) for e in epochs])}"/>'
    )
    for e in epochs:
        out += _dot(
            "viz-s1",
            xmap(e),
            y_loss(val[e]),
            f"Epoch {e:g}: validation loss {val[e]:.3f}",
        )
    out.append(
        f'<path class="viz-line viz-s2" d="{_path([(xmap(e), y_ans(faithful[e])) for e in fe])}"/>'
    )
    for e in fe:
        out += _dot(
            "viz-s2",
            xmap(e),
            y_ans(faithful[e]),
            f"Epoch {e:g}: {faithful[e]} of 30 answers faithful",
        )
    out.append("</svg>")
    return "\n".join(out)
