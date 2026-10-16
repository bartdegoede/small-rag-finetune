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
