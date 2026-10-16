import xml.etree.ElementTree as ET

from small_rag_finetune.charts import loss_chart, nice_ticks, smooth

TRAIN = {
    50: 1.2,
    100: 1.0,
    150: 1.1,
    200: 0.9,
    250: 0.95,
    300: 0.85,
    350: 0.9,
    400: 0.8,
}
VAL = {1: 1.7, 100: 1.0, 200: 0.9, 300: 0.92, 400: 0.95}


def test_smooth_keeps_length_and_averages():
    assert smooth([1.0, 2.0, 3.0], window=3) == [1.5, 2.0, 2.5]


def test_nice_ticks_cover_the_range():
    ticks = nice_ticks(0.83, 1.71)
    assert ticks[0] <= 0.83 and ticks[-1] >= 1.71
    assert len(ticks) <= 6


def test_loss_chart_is_valid_svg_and_rings_the_lowest_later_val():
    svg = loss_chart("Tiny", TRAIN, VAL, epoch_iters=200)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    tips = [t.text for t in root.iter(f"{ns}title")]
    assert "1.70 before training" in svg
    assert not any("iteration 1)" in t for t in tips)  # baseline is not plotted
    assert any("iteration 200" in t and "checkpoint is used" in t for t in tips)
    rings = [c for c in root.iter(f"{ns}circle") if "viz-ring" in c.get("class")]
    assert len(rings) == 2  # legend + the chosen checkpoint
