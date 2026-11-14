"""Animated results panel of the README: ``docs/media/results.svg``.

Three panels on one card, every number read from ``tools/readme_media/numbers.json`` (the out-of-fold evaluation of
the 707 training images and the public scores):

* (a) out-of-fold PQ of the three configurations as horizontal bars on an axis from 0.30 to 0.48, each with its public
  score, and the agreement of two annotators with each other as a dashed reference line;
* (b) the decomposition PQ = SQ x RQ of cross-fitting shuffle 0 and its counts of true positives, false positives and
  false negatives as one segmented bar;
* (c) the IoU and Dice of the matched pairs (IoU above 0.5) as quantile strips: whiskers from the 10 % to the 90 %
  quantile, a box from 25 % to 75 %, a line at the median and a dot at the mean.

On load the bars and strips grow from their baseline and the numbers fade in (played once, about 1.5 s, then frozen);
a slow shimmer runs along the bar of the final configuration. Every mark rests at its final value, so the figure is
complete without animation.

    python tools/readme_media/results_charts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from style import TextLayer, num, text_budget  # noqa: E402

NUMBERS_PATH = Path(__file__).resolve().parent / "numbers.json"
OUT_PATH = style.MEDIA / "results.svg"

WIDTH, HEIGHT = 1760, 780
MARGIN = 48
RIGHT = WIDTH - MARGIN
TITLE_PX, TEXT_PX, BIG_PX, OPERATOR_PX = 28, 26, 48, 40
EYEBROW_SPACING = 3
LINE_STEP = 30
CAP_TO_CENTRE = 9
"""Offset from the vertical centre of a 26 px line to its baseline."""

# --- panel (a): ablation bars ----------------------------------------------------------------------------------------
A_TITLE_Y = 74
A_LABEL_W = 576
A_X0, A_X1 = 648, 1420
A_MIN, A_MAX, A_TICK = 0.30, 0.48, 0.02
A_ROW_Y, A_ROW_STEP, BAR_H = 132, 64, 28
A_AXIS_Y = 296
A_TICK_LABEL_Y = 332
A_REF_TOP = 54
A_GRID_TOP = 96
CHIP_W, CHIP_H, CHIP_PAD = 220, 40, 14
BAR_ROUND = ' rx="3"'
"""Corner radius attribute of bars, segments and boxes."""
VALUE_GAP = 12
BREAK_OFFSETS = (-18, -11)
"""Positions (relative to the axis start) of the two strokes that mark the axis as not starting at zero."""

# --- lower panels ----------------------------------------------------------------------------------------------------
SEPARATOR_Y = 362
COLUMN_SPLIT = 784
LOWER_TITLE_Y, LOWER_SUB_Y = 404, 438
B_LEFT, B_RIGHT = MARGIN, COLUMN_SPLIT - 24
B_NUMBER_Y, B_NAME_Y = 516, 552
B_BAR_Y, B_LABEL_Y, B_SHARE_Y = 576, 638, 668
SEGMENT_GAP = 3
C_LEFT = COLUMN_SPLIT + 24
C_X0, C_X1 = 1008, 1680
C_MIN, C_MAX, C_TICK = 0.5, 1.0, 0.1
C_TICK_LABEL_Y, C_AXIS_Y = 476, 488
C_STRIP_Y, C_STRIP_STEP = 514, 90
LEADER_TOP, LEADER_BOTTOM, VALUE_DY = 18, 34, 58
VALUE_STEP = 100
FOOTER_Y = 726

# --- animation -------------------------------------------------------------------------------------------------------
GROW_S, FADE_S = 1.5, 0.6
EASE = "cubic-bezier(0.22, 0.8, 0.24, 1)"
SHIMMER_W, SHIMMER_S, SHIMMER_DELAY_S = 140, 4.8, 1.8


def wrap_phrases(text: str, size: float, room: float, bold: bool) -> list[str]:
    """Split ``text`` after its commas and pack the phrases greedily into lines that fit ``room``."""
    phrases = [p + "," for p in text.split(", ")]
    phrases[-1] = phrases[-1][:-1]
    lines = [phrases[0]]
    for p in phrases[1:]:
        candidate = f"{lines[-1]} {p}"
        if text_budget(candidate, size, bold=bold) <= room:
            lines[-1] = candidate
        else:
            lines.append(p)
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# shared marks
# ---------------------------------------------------------------------------------------------------------------------
def scale(v: float, lo: float, hi: float, x0: float, x1: float) -> float:
    """Position of value ``v`` on an axis that maps [lo, hi] to [x0, x1]."""
    return x0 + (v - lo) / (hi - lo) * (x1 - x0)


def ticks(lo: float, hi: float, step: float) -> list[float]:
    """Tick values from ``lo`` to ``hi`` inclusive."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.5, extra: str = "") -> str:
    """``<line>`` element."""
    return (f'<line x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}" stroke="{stroke}" '
            f'stroke-width="{num(width)}"{extra}/>\n')


def rect(x: float, y: float, w: float, h: float, fill: str, extra: str = "") -> str:
    """``<rect>`` element."""
    return f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" fill="{fill}"{extra}/>\n'


def x_ablation(v: float) -> float:
    """Position of a PQ value on the axis of panel (a)."""
    return scale(v, A_MIN, A_MAX, A_X0, A_X1)


def x_pairs(v: float) -> float:
    """Position of an IoU or Dice value on the axis of panel (c)."""
    return scale(v, C_MIN, C_MAX, C_X0, C_X1)


def title(layer: TextLayer, text: str, x: float, y: float, room: float) -> str:
    """Panel title."""
    return layer.text(text, x, y, TITLE_PX, room, style.TEXT, bold=True)


# ---------------------------------------------------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------------------------------------------------
def panel_ablation(layer: TextLayer, nums: dict) -> str:
    """(a) Out-of-fold PQ by configuration, public score chips and the annotator-agreement reference."""
    chip_x = RIGHT - CHIP_W
    ref = nums["annotator_agreement_pq"]
    xr = x_ablation(ref)
    parts = [title(layer, "Out-of-fold PQ by configuration", MARGIN, A_TITLE_Y, xr - 16 - MARGIN)]
    for t in ticks(A_MIN, A_MAX, A_TICK):
        x = x_ablation(t)
        parts.append(line(x, A_GRID_TOP, x, A_AXIS_Y, style.GRID, 1))
        parts.append(line(x, A_AXIS_Y, x, A_AXIS_Y + 8, style.PANEL_EDGE, 1.5))
        parts.append(layer.text(f"{t:.2f}", x, A_TICK_LABEL_Y, TEXT_PX, x_ablation(A_MIN + A_TICK) - A_X0 - 8,
                                style.MUTED, mono=True, anchor="middle"))
    parts.append(line(A_X0 - 26, A_AXIS_Y, A_X1, A_AXIS_Y, style.PANEL_EDGE, 1.5))
    for dx in BREAK_OFFSETS:
        parts.append(line(A_X0 + dx - 3, A_AXIS_Y + 7, A_X0 + dx + 3, A_AXIS_Y - 7, style.MUTED, 1.5))
    parts.append(layer.text(f"axis: PQ from {A_MIN:.2f} to {A_MAX:.2f}", MARGIN, A_TICK_LABEL_Y, TEXT_PX,
                            A_X0 - 26 - 16 - MARGIN, style.MUTED))
    parts.append(line(xr, A_REF_TOP, xr, A_AXIS_Y, style.SLATE, 2, ' stroke-dasharray="6 6"'))
    parts.append(layer.text(f"two annotators against each other: {ref:.3f}", xr + 12, A_TITLE_Y, TEXT_PX,
                            RIGHT - xr - 12, style.MUTED))
    entries = nums["ablation"]
    for i, entry in enumerate(entries):
        final = i == len(entries) - 1
        colour = style.GREEN if final else style.CYAN
        cy = A_ROW_Y + i * A_ROW_STEP
        lines = wrap_phrases(entry["label"], TEXT_PX, A_LABEL_W - 24, final)
        for k, text in enumerate(lines):
            y = cy + CAP_TO_CENTRE - (len(lines) - 1) * LINE_STEP / 2 + k * LINE_STEP
            parts.append(layer.text(text, MARGIN, y, TEXT_PX, A_LABEL_W - 24, style.TEXT, bold=final))
        xv = x_ablation(entry["oof_pq"])
        parts.append(rect(A_X0, cy - BAR_H / 2, xv - A_X0, BAR_H, colour,
                          f' fill-opacity="0.85"{BAR_ROUND} class="grow oa d{i}"'))
        if final:
            parts.append(f'<g clip-path="url(#final-bar)"><rect x="{num(A_X0 - SHIMMER_W)}" y="{num(cy - BAR_H / 2)}" '
                         f'width="{SHIMMER_W}" height="{BAR_H}" fill="url(#shimmer)" class="shimmer"/></g>\n')
        parts.append(layer.text(f"{entry['oof_pq']:.4f}", xv + VALUE_GAP, cy + CAP_TO_CENTRE, TEXT_PX,
                                chip_x - 16 - xv - VALUE_GAP, style.GREEN if final else style.TEXT, mono=True,
                                css_class="fade d3"))
        parts.append(rect(chip_x, cy - CHIP_H / 2, CHIP_W, CHIP_H, style.BG_SOFT,
                          f' rx="{CHIP_H / 2:g}" stroke="{style.GREEN if final else style.PANEL_EDGE}" '
                          'stroke-width="1.5" class="fade d3"'))
        parts.append(layer.text(f"public {entry['public']:.2f}", chip_x + CHIP_PAD, cy + CAP_TO_CENTRE, TEXT_PX,
                                CHIP_W - 2 * CHIP_PAD, style.GREEN if final else style.MUTED, mono=True,
                                css_class="fade d3"))
    return "".join(parts)


def final_bar_clip(nums: dict) -> str:
    """Clip path of the final configuration's bar (keeps the shimmer inside it)."""
    i = len(nums["ablation"]) - 1
    xv = x_ablation(nums["ablation"][i]["oof_pq"])
    cy = A_ROW_Y + i * A_ROW_STEP
    bar = rect(A_X0, cy - BAR_H / 2, xv - A_X0, BAR_H, style.TEXT, BAR_ROUND).strip()
    return f'<clipPath id="final-bar">{bar}</clipPath>\n'


def panel_decomposition(layer: TextLayer, nums: dict) -> str:
    """(b) PQ = SQ x RQ for shuffle 0 and the segmented bar of TP, FP and FN."""
    s0 = nums["shuffle0"]
    width = B_RIGHT - B_LEFT
    number_w = text_budget("0.0000", BIG_PX, mono=True)
    op_w = (width - 3 * number_w) / 2
    parts = [title(layer, "PQ = SQ × RQ", B_LEFT, LOWER_TITLE_Y, width),
             layer.text("cross-fitting shuffle 0", B_LEFT, LOWER_SUB_Y, TEXT_PX, width, style.MUTED)]
    terms = [("SQ", s0["sq"], style.TEXT), ("RQ", s0["rq"], style.TEXT), ("PQ", s0["pq"], style.SUN)]
    for i, (name, value, colour) in enumerate(terms):
        x = B_LEFT + i * (number_w + op_w)
        parts.append(layer.text(f"{value:.4f}", x, B_NUMBER_Y, BIG_PX, number_w, colour, mono=True,
                                css_class=f"fade d{i}"))
        parts.append(layer.text(name, x, B_NAME_Y, TEXT_PX, number_w, style.MUTED, mono=True,
                                spacing=EYEBROW_SPACING))
        if i < 2:
            parts.append(layer.text("×" if i == 0 else "=", x + number_w + op_w / 2, B_NUMBER_Y - 4, OPERATOR_PX,
                                    op_w, style.MUTED, mono=True, anchor="middle"))
    # (name, count, colour of the segment, colour of its label)
    counts = [("TP", s0["tp"], style.GREEN, style.GREEN), ("FP", s0["fp"], style.ROSE, style.ROSE),
              ("FN", s0["fn"], style.SLATE, style.MUTED)]
    total = sum(c[1] for c in counts)
    usable = width - SEGMENT_GAP * (len(counts) - 1)
    x, spans = B_LEFT, []
    for _, count, colour, _ in counts:
        w = usable * count / total
        parts.append(rect(x, B_BAR_Y, w, BAR_H, colour, f' fill-opacity="0.85"{BAR_ROUND} class="grow ob d1"'))
        spans.append((x, w))
        x += w + SEGMENT_GAP
    labels = [f"{name} {count:,}" for name, count, _, _ in counts]
    shares = [f"{100 * count / total:.1f} %" for _, count, _, _ in counts]
    for row_y, texts, fills in [(B_LABEL_Y, labels, [c[3] for c in counts]),
                                (B_SHARE_Y, shares, [style.MUTED] * len(counts))]:
        needs = [text_budget(t, TEXT_PX, mono=True) for t in texts]
        mid = spans[1][0] + spans[1][1] / 2
        rooms = [mid - needs[1] / 2 - 12 - B_LEFT,
                 2 * min(mid - B_LEFT - needs[0] - 12, B_RIGHT - needs[2] - 12 - mid),
                 B_RIGHT - (mid + needs[1] / 2) - 12]
        anchors = [(B_LEFT, "start"), (mid, "middle"), (B_RIGHT, "end")]
        for text, fill, room, (ax, anchor) in zip(texts, fills, rooms, anchors):
            parts.append(layer.text(text, ax, row_y, TEXT_PX, room, fill, mono=True, anchor=anchor,
                                    css_class="fade d3"))
    return "".join(parts)


def quantile_strip(layer: TextLayer, name: str, q: dict, cy: float, index: int) -> str:
    """(c) One strip: whiskers 10-90 %, box 25-75 %, median line, mean dot, the five values under it."""
    keys = ("q10", "q25", "q50", "q75", "q90")
    xs = {k: x_pairs(q[k]) for k in keys}
    half = BAR_H / 2
    group = [line(xs["q10"], cy, xs["q90"], cy, style.GREEN, 2),
             line(xs["q10"], cy - 9, xs["q10"], cy + 9, style.GREEN, 2),
             line(xs["q90"], cy - 9, xs["q90"], cy + 9, style.GREEN, 2),
             rect(xs["q25"], cy - half, xs["q75"] - xs["q25"], BAR_H, style.GREEN,
                  f' fill-opacity="0.22" stroke="{style.GREEN}" stroke-width="2"{BAR_ROUND}'),
             line(xs["q50"], cy - half, xs["q50"], cy + half, style.TEXT, 2.5)]
    parts = [f'<g class="grow oc d{index + 1}">\n' + "".join(group) + "</g>\n",
             f'<circle cx="{num(x_pairs(q["mean"]))}" cy="{num(cy)}" r="6" fill="{style.TEXT}" '
             f'stroke="{style.GREEN}" stroke-width="2" class="fade d3"/>\n',
             layer.text(name, C_LEFT, cy + CAP_TO_CENTRE, TEXT_PX, C_X0 - 16 - C_LEFT, style.TEXT, bold=True),
             layer.text(f"mean {q['mean']:.3f}", C_LEFT, cy + VALUE_DY, TEXT_PX, C_X0 - 16 - C_LEFT, style.TEXT,
                        mono=True, css_class="fade d3")]
    half_value = text_budget("0.000", TEXT_PX, mono=True) / 2
    first = min(max(xs["q50"] - 2 * VALUE_STEP, C_X0 + half_value), RIGHT - half_value - 4 * VALUE_STEP)
    for i, k in enumerate(keys):
        vx = first + i * VALUE_STEP
        parts.append(line(xs[k], cy + LEADER_TOP, vx, cy + LEADER_BOTTOM, style.MUTED, 1,
                          ' stroke-opacity="0.6" class="fade d3"'))
        parts.append(layer.text(f"{q[k]:.3f}", vx, cy + VALUE_DY, TEXT_PX, VALUE_STEP - 8, style.MUTED, mono=True,
                                anchor="middle", css_class="fade d3"))
    return "".join(parts)


def panel_pairs(layer: TextLayer, nums: dict) -> str:
    """(c) IoU and Dice of the matched pairs as quantile strips on a common axis from 0.5 to 1.0."""
    pairs = nums["matched_pairs"]
    width = RIGHT - C_LEFT
    parts = [title(layer, f"Matched pairs (IoU above 0.5), n = {pairs['n']:,}", C_LEFT, LOWER_TITLE_Y, width),
             layer.text("box 25–75 %, whiskers 10–90 %, median line, mean dot", C_LEFT, LOWER_SUB_Y, TEXT_PX, width,
                        style.MUTED)]
    last_y = C_STRIP_Y + C_STRIP_STEP + BAR_H / 2
    step_px = scale(C_MIN + C_TICK, C_MIN, C_MAX, C_X0, C_X1) - C_X0
    for t in ticks(C_MIN, C_MAX, C_TICK):
        x = scale(t, C_MIN, C_MAX, C_X0, C_X1)
        parts.append(line(x, C_AXIS_Y + 6, x, last_y, style.GRID, 1))
        parts.append(line(x, C_AXIS_Y, x, C_AXIS_Y + 6, style.PANEL_EDGE, 1.5))
        parts.append(layer.text(f"{t:.1f}", x, C_TICK_LABEL_Y, TEXT_PX, step_px - 8, style.MUTED, mono=True,
                                anchor="middle"))
    parts.append(line(C_X0, C_AXIS_Y, C_X1, C_AXIS_Y, style.PANEL_EDGE, 1.5))
    for i, (name, key) in enumerate([("IoU", "iou"), ("Dice", "dice")]):
        parts.append(quantile_strip(layer, name, pairs[key], C_STRIP_Y + i * C_STRIP_STEP, i))
    return "".join(parts)


# ---------------------------------------------------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------------------------------------------------
def css(nums: dict) -> str:
    """The one style block: grow and fade once on load, an endless shimmer, the reduced-motion rule."""
    shimmer_path = x_ablation(nums["ablation"][-1]["oof_pq"]) - A_X0 + SHIMMER_W
    rules = [
        f".grow {{ transform-box: view-box; animation: grow {GROW_S}s {EASE} both; }}",
        "@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }",
        f".oa {{ transform-origin: {A_X0}px 0px; }}",
        f".ob {{ transform-origin: {B_LEFT}px 0px; }}",
        f".oc {{ transform-origin: {C_X0}px 0px; }}",
        f".fade {{ animation: fade {FADE_S}s ease-out both; }}",
        "@keyframes fade { from { opacity: 0; } to { opacity: 1; } }",
        ".d0 { animation-delay: 0.1s; }",
        ".d1 { animation-delay: 0.25s; }",
        ".d2 { animation-delay: 0.4s; }",
        ".d3 { animation-delay: 1.1s; }",
        f".shimmer {{ animation: shimmer {SHIMMER_S}s ease-in-out {SHIMMER_DELAY_S}s infinite; }}",
        f"@keyframes shimmer {{ 0% {{ transform: translateX(0px); }} "
        f"55%, 100% {{ transform: translateX({num(shimmer_path)}px); }} }}",
        style.REDUCED_MOTION_CSS,
    ]
    return "<style>\n" + "\n".join(rules) + "\n</style>\n"


def description(nums: dict) -> str:
    """Text alternative of the figure."""
    abl = "; ".join(f"{e['label']}: {e['oof_pq']:.4f} (public {e['public']:.2f})" for e in nums["ablation"])
    s0, mp = nums["shuffle0"], nums["matched_pairs"]
    return (f"Out-of-fold PQ by configuration: {abl}; two annotators against each other: "
            f"{nums['annotator_agreement_pq']:.3f}. Shuffle 0: SQ {s0['sq']:.4f} x RQ {s0['rq']:.4f} = PQ "
            f"{s0['pq']:.4f}; TP {s0['tp']}, FP {s0['fp']}, FN {s0['fn']}. Matched pairs (n = {mp['n']}): IoU median "
            f"{mp['iou']['q50']:.3f}, mean {mp['iou']['mean']:.3f}; Dice median {mp['dice']['q50']:.3f}, mean "
            f"{mp['dice']['mean']:.3f}.")


def build_svg(nums: dict, layer: TextLayer) -> str:
    """The complete SVG document."""
    counts = nums["counts"]
    stops = style.gradient_stops([(0, style.TEXT, 0), (0.5, style.TEXT, 0.45), (1, style.TEXT, 0)])
    gradient = f'<linearGradient id="shimmer" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient>\n'
    parts = [style.svg_open(WIDTH, HEIGHT, "Results", escape(description(nums))),
             style.svg_defs(gradient + final_bar_clip(nums)), css(nums),
             style.svg_background(WIDTH, HEIGHT),
             panel_ablation(layer, nums),
             line(MARGIN, SEPARATOR_Y, RIGHT, SEPARATOR_Y, style.PANEL_EDGE, 1),
             line(COLUMN_SPLIT, SEPARATOR_Y + 22, COLUMN_SPLIT, B_SHARE_Y + 8, style.PANEL_EDGE, 1),
             panel_decomposition(layer, nums),
             panel_pairs(layer, nums),
             layer.text(f"{counts['training_images']} training images, every image scored by models that never saw "
                        f"it; {counts['image_annotator_pairs']:,} image-annotator pairs", MARGIN, FOOTER_Y, TEXT_PX,
                        RIGHT - MARGIN, style.MUTED),
             "</svg>\n"]
    return "".join(parts)


def main() -> None:
    """Read the numbers, build the figure, check the text budgets and write the file."""
    nums = json.loads(NUMBERS_PATH.read_text(encoding="utf-8"))
    layer = TextLayer()
    svg = build_svg(nums, layer)
    layer.report()
    size = style.write_text(OUT_PATH, svg)
    print(f"wrote {OUT_PATH.relative_to(style.REPO)}: {size} bytes")


if __name__ == "__main__":
    main()
