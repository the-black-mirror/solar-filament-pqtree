"""Animated figure: one filament region of a test image becomes a component tree, and the dynamic programme keeps
the best set of non-overlapping candidates in it.

Left, the probability map of the dense networks around the region; the outlines of the components appear threshold by
threshold (0.2 ... 0.9). Right, the same components as the nodes of the component tree (x = threshold, node radius
growing with the square root of the component area, violet edges to the parent at the level below). Every node then
receives its predicted matched IoU, and finally the kept candidates turn green on both sides.

Computed from the stored artefacts of the repository only:

* candidates of the test images with all 52 model inputs: ``pqtree.pipeline.build_test_table``;
* scores: the stored hurdle LightGBM models (``artifacts/lgbm_hit.txt``, ``artifacts/lgbm_iou.txt``), applied as in
  ``pqtree.pipeline.default_submission``; kept set: ``pqtree.selection.select`` with ``config.TAU``;
* component label images: ``artifacts/test_components.npz``; probability maps: ``artifacts/test_probs_unet.npz``.

The tree links a candidate to the component that contains it one level below through the ``parent`` column of the
candidate table (see ``pqtree.candidates``).

Choice of the example, printed when the script runs: among the level-0 components (roots) of all test images, those
whose subtree has 7 to 16 candidates, spans at least 4 levels, contains a candidate with two or more children and has at
least two kept candidates; of these, the root with the largest area whose crop (1.2 times the longer side of its
bounding box) fits a 640 px square.

Usage::

    python tools/readme_media/component_tree.py [--frames DIR]

``--frames`` also writes key frames as PNG (and the final state at half size) into DIR.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import contourpy
import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from pqtree import lgbm_scorer, selection  # noqa: E402
from pqtree.artifacts import fetch_artifact  # noqa: E402
from pqtree.config import LEVELS, TAU  # noqa: E402
from pqtree.pipeline import build_test_table  # noqa: E402

OUT = style.MEDIA / "component_tree.gif"

# --- choice of the example -----------------------------------------------------------------------------------------
MIN_NODES, MAX_NODES = 7, 16
"""Allowed number of candidates in the subtree of the example root."""
MIN_DEPTH = 4
"""Minimum number of levels spanned by the subtree."""
MIN_KEPT = 2
"""Minimum number of kept candidates in the subtree."""
CROP_FACTOR = 1.2
"""Side of the map crop relative to the longer side of the root's bounding box."""
MAX_CROP = 640
"""Largest allowed crop side (image pixels)."""

# --- canvas and grid (figure pixels) -------------------------------------------------------------------------------
WIDTH, HEIGHT = 1520, 840
MARGIN = 48
DPI = 100
PT = 72 / DPI
"""Points per figure pixel (matplotlib sizes fonts and lines in points)."""
HEADER_Y = 66
RULE_TOP_Y = 104
EYEBROW_Y = 142
MAP_X, MAP_Y, MAP_PX = MARGIN, 168, 520
"""Top-left corner and side of the map panel."""
MAP_RADIUS = 14
TREE_X = 632
"""Left edge of the tree panel."""
COL_X0, COL_X1 = 680, 1424
"""Centres of the first (threshold 0.2) and last (0.9) tree column."""
SCALE_LABEL_Y = 196
TRACK_Y = 228
GUIDE_Y0, GUIDE_Y1 = 250, 604
NODE_Y0, NODE_Y1 = 286, 556
"""Vertical band of the tree nodes."""
MAX_ROW_GAP = 120
SUM_Y = (650, 686)
RULE_BOTTOM_Y = 716
CAPTION_Y = 762
R_MIN, R_MAX = 6.0, 22.0
"""Node radius of the smallest candidate and of the root (radius grows with the square root of the area)."""
KEPT_RING = 8
"""Gap between a kept node and its green ring."""
LABEL_GAP = 28
"""Distance from the bottom of a node to the centre of its score label (clears the ring of a kept node)."""
EDGE_SAMPLES = 48
"""Points per tree edge (an edge grows by revealing them in order)."""
STEP_BAR = (36, 10)
"""Length and gap of the four step markers right of the caption."""

# --- type ----------------------------------------------------------------------------------------------------------
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
TEXT_PX = 26
CAPTION_PX = 28
MONO_ADVANCE = 0.602
"""Advance width of a DejaVu Sans Mono glyph in em."""
TRACKING_PX = 4
"""Extra letter spacing of the upper-case eyebrow labels."""

# --- colour use ----------------------------------------------------------------------------------------------------
SCORE_FULL = 0.75
"""Score at which the cyan node fill reaches full strength."""
GREEN_FILL_ALPHA = 0.38
MAP_DIM = 0.35
"""How much the map darkens when the kept masks are shown."""
BLOOM = 0.6
"""Strength of the glow added around high probabilities (screen blend of a blurred copy)."""
BLOOM_SIGMA = 9
"""Width (figure pixels) of the glow."""

# --- timeline (seconds) --------------------------------------------------------------------------------------------
FRAME_MS = 80
"""Frame duration (12.5 frames per second)."""
CAPTION_FADE = 0.3
"""Duration of a caption fade; the captions change while nothing else moves, so that a frame changes in one place."""
MAP_IN = (0.3, 1.3)
TREE_START = 1.9
LEVEL_DT = 0.5
"""Time per threshold level in step 2."""
MARKER = (0.0, 0.3)
EDGE = (0.1, 0.3)
NODE = (0.15, 0.45)
OUTLINE = (0.1, 0.45)
"""Offsets (from the start of a level) of the marker move and of the appearance of edges, nodes and outlines."""
INDICATOR_OUT = (5.9, 6.2)
SCORE_START = 6.8
SCORE_STAGGER = 0.08
SCORE_DT = 0.4
DECIDE_START = 8.4
DECIDE_DT = 0.8
SUM_IN = (9.2, 9.6)
FINAL = SUM_IN[1]
HOLD = 2.5
FADE_OUT = 0.4
TOTAL = FINAL + HOLD + FADE_OUT
CAPTIONS = (("1", "probability map of the dense networks"),
            ("2", "components at 8 thresholds nest into a tree"),
            ("3", "each candidate gets a predicted matched IoU"),
            ("4", f"keep the best non-overlapping set with score above {TAU:.2f}"))
STEP_STARTS = (0.0, TREE_START, SCORE_START, DECIDE_START)
"""The caption of a step is fully shown when its action starts."""
KEY_TIMES = (1.0, 2.25, 3.3, 4.45, 5.95, 7.1, 7.9, 8.75, FINAL + 0.5, TOTAL - 0.2)
"""Times of the key frames written with ``--frames``."""


@dataclass
class Node:
    """One candidate of the example subtree with its place in the drawing."""
    level: int
    cid: int
    parent: int
    area: float
    score: float
    kept: bool
    cy: float
    children: list["Node"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    r: float = 0.0
    outline: list[np.ndarray] = field(default_factory=list)
    """Contour polylines in figure pixels."""
    alpha: np.ndarray | None = None
    """Soft mask on the map panel grid (kept candidates only)."""


@dataclass
class Scene:
    """Everything a frame needs: the map image, the nodes and the texts computed from the data."""
    stem: str
    nodes: list[Node]
    map_rgb: np.ndarray
    sum_lines: tuple[str, str]


# --- data ----------------------------------------------------------------------------------------------------------
def scored_test_table() -> pd.DataFrame:
    """Test candidates with the hurdle-model score and the kept flag, exactly as in the default submission path."""
    table = build_test_table()
    model = lgbm_scorer.HurdleModel.load(fetch_artifact(lgbm_scorer.HIT_FILE), fetch_artifact(lgbm_scorer.IOU_FILE))
    score = model.predict(table)
    kept = set(selection.select(table, score, TAU))
    return table.assign(score=score, kept=[(s, lv, c) in kept for s, lv, c in zip(table.stem, table.level, table.cid)])


def with_roots(table: pd.DataFrame) -> pd.DataFrame:
    """Add the column ``root``: the level-0 component id at the top of each candidate's chain of parents."""
    root_of: dict = {}
    for row in table.sort_values("level", kind="stable").itertuples():
        own = (row.stem, row.level, row.cid)
        root_of[own] = row.cid if row.level == 0 else root_of[(row.stem, row.level - 1, row.parent)]
    return table.assign(root=[root_of[k] for k in zip(table.stem, table.level, table.cid)])


def eligible_roots(table: pd.DataFrame) -> pd.DataFrame:
    """One row per root that satisfies the rule of the module docstring, largest area first."""
    rows = []
    for (stem, root), sub in table.groupby(["stem", "root"], sort=True):
        n_children = sub[sub.level > 0].groupby(["level", "parent"]).size()
        top = sub[sub.level == 0].iloc[0]
        crop = int(np.ceil(CROP_FACTOR * max(top.bw, top.bh)))
        ok = (MIN_NODES <= len(sub) <= MAX_NODES and sub.level.nunique() >= MIN_DEPTH
              and (n_children >= 2).any() and sub.kept.sum() >= MIN_KEPT and crop <= MAX_CROP)
        if ok:
            rows.append(dict(stem=stem, root=int(root), nodes=len(sub), levels=sub.level.nunique(),
                             kept=int(sub.kept.sum()), area=float(top.area), crop=crop))
    return pd.DataFrame(rows).sort_values(["area", "stem"], ascending=[False, True], ignore_index=True)


def crop_window(labels0: np.ndarray, root: int, side: int) -> tuple[int, int, int]:
    """Top-left corner and side of a square crop centred on the bounding box of a level-0 component."""
    ys, xs = np.nonzero(labels0 == root)
    cy, cx = (ys.min() + ys.max() + 1) / 2, (xs.min() + xs.max() + 1) / 2
    h, w = labels0.shape
    y0 = int(np.clip(round(cy - side / 2), 0, h - side))
    x0 = int(np.clip(round(cx - side / 2), 0, w - side))
    return y0, x0, side


# --- geometry of the drawing ---------------------------------------------------------------------------------------
def outline_polylines(mask: np.ndarray, scale: float) -> list[np.ndarray]:
    """Sub-pixel contour lines of a binary crop mask, in figure pixels of the map panel."""
    pad = 2
    soft = cv2.GaussianBlur(np.pad(mask.astype(np.float32), pad), (0, 0), 0.8)
    lines = contourpy.contour_generator(z=soft, line_type="Separate").lines(0.5)
    return [np.column_stack([MAP_X + (ln[:, 0] - pad + 0.5) * scale, MAP_Y + (ln[:, 1] - pad + 0.5) * scale])
            for ln in lines if len(ln) > 2]


def soft_mask(mask: np.ndarray) -> np.ndarray:
    """Crop mask resampled to the map panel with a narrow anti-aliased edge (values 0 ... 1)."""
    up = cv2.resize(mask.astype(np.float32), (MAP_PX, MAP_PX), interpolation=cv2.INTER_LINEAR)
    return np.clip((cv2.GaussianBlur(up, (0, 0), 1.0) - 0.5) * 3 + 0.5, 0, 1)


def glowing_map(prob_crop: np.ndarray) -> np.ndarray:
    """Probability crop in the probability colour map, resampled to the panel, with a soft bloom on high values."""
    p = cv2.resize(prob_crop.astype(np.float32) / 255, (MAP_PX, MAP_PX), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    rgb = style.probability_colormap()(p)[..., :3]
    bloom = cv2.GaussianBlur((rgb * p[..., None] ** 2).astype(np.float32), (0, 0), BLOOM_SIGMA)
    return 1 - (1 - rgb) * (1 - BLOOM * bloom)


def build_nodes(sub: pd.DataFrame, labels: Callable[[int], np.ndarray], window: tuple[int, int, int]) -> list[Node]:
    """Nodes of the subtree with their outlines on the map; children are linked through ``parent`` and ordered by
    their vertical position in the image."""
    y0, x0, side = window
    scale = MAP_PX / side
    nodes = {}
    for row in sub.sort_values(["level", "cid"]).itertuples():
        mask = labels(row.level)[y0:y0 + side, x0:x0 + side] == row.cid
        node = Node(int(row.level), int(row.cid), int(row.parent), float(row.area), float(row.score), bool(row.kept),
                    float(row.cy), outline=outline_polylines(mask, scale))
        if node.kept:
            node.alpha = soft_mask(mask)
        nodes[(node.level, node.cid)] = node
        if node.level > 0:
            nodes[(node.level - 1, node.parent)].children.append(node)
    for node in nodes.values():
        node.children.sort(key=lambda c: c.cy)
    return list(nodes.values())


def column_x(level: int) -> float:
    """Centre of the tree column of a threshold level."""
    return COL_X0 + (COL_X1 - COL_X0) * level / (len(LEVELS) - 1)


def layout_tree(nodes: list[Node]) -> None:
    """Tidy layout: leaves on equally spaced rows in depth-first order, every parent at the mean of its children.

    Depth-first leaf order keeps every subtree on a contiguous band of rows, so no two edges cross.
    """
    root = next(n for n in nodes if n.level == 0)
    leaves: list[Node] = []

    def collect(node: Node) -> None:
        if not node.children:
            leaves.append(node)
        for child in node.children:
            collect(child)

    def place(node: Node) -> float:
        if node.children:
            node.y = float(np.mean([place(c) for c in node.children]))
        return node.y

    collect(root)
    gap = min(MAX_ROW_GAP, (NODE_Y1 - NODE_Y0) / max(len(leaves) - 1, 1))
    top = (NODE_Y0 + NODE_Y1) / 2 - gap * (len(leaves) - 1) / 2
    for i, leaf in enumerate(leaves):
        leaf.y = top + i * gap
    place(root)
    for node in nodes:
        node.x = column_x(node.level)
        node.r = R_MIN + (R_MAX - R_MIN) * np.sqrt(node.area / root.area)


def edge_points(parent: Node, child: Node) -> np.ndarray:
    """Cubic S-curve from the right side of a parent to the left side of a child (horizontal end tangents)."""
    x0, y0, x1, y1 = parent.x + parent.r, parent.y, child.x - child.r, child.y
    s = np.linspace(0, 1, EDGE_SAMPLES)[:, None]
    p0, p1, p2, p3 = (np.array(p, float) for p in ((x0, y0), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), (x1, y1)))
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s ** 2 * p2 + s ** 3 * p3


def objective_lines(nodes: list[Node]) -> tuple[str, str]:
    """The sum the programme maximises for this tree, for the kept set and for the root alone."""
    kept = sorted((n for n in nodes if n.kept), key=lambda n: (n.level, n.cid))
    root = next(n for n in nodes if n.level == 0)
    terms = " + ".join(f"({n.score:.2f} − {TAU:.2f})" for n in kept)
    gain = sum(n.score - TAU for n in kept)
    return (f"kept set    {terms} = {gain:.2f}",
            f"root alone  ({root.score:.2f} − {TAU:.2f}) = {root.score - TAU:.2f}")


# --- timeline ------------------------------------------------------------------------------------------------------
def ramp(t: float, start: float, end: float) -> float:
    """Smoothstep from 0 at ``start`` to 1 at ``end``."""
    u = float(np.clip((t - start) / (end - start), 0, 1))
    return u * u * (3 - 2 * u)


def level_start(level: int) -> float:
    """Time at which the threshold indicator starts moving to a level."""
    return TREE_START + level * LEVEL_DT


def level_ramp(t: float, level: int, offsets: tuple[float, float]) -> float:
    """Smoothstep of one event of a level, ``offsets`` measured from the start of the level."""
    return ramp(t, level_start(level) + offsets[0], level_start(level) + offsets[1])


def score_ramp(t: float, level: int) -> float:
    """Appearance of the scores of one level in step 3 (levels follow each other with a small delay)."""
    start = SCORE_START + level * SCORE_STAGGER
    return ramp(t, start, start + SCORE_DT)


def frame_times() -> np.ndarray:
    """Start time of every frame of the loop."""
    return np.arange(0, TOTAL, FRAME_MS / 1000)


# --- drawing primitives (figure pixels) ----------------------------------------------------------------------------
def put_text(ax, x: float, y: float, s: str, colour: str, size_px: float = TEXT_PX, family: str = MONO,
             ha: str = "left", alpha: float = 1.0, z: int = 3) -> None:
    """Text centred vertically on ``y``."""
    ax.text(x, y, s, color=colour, alpha=alpha, fontsize=size_px * PT, family=family, ha=ha, va="center_baseline",
            zorder=z)


def spaced_text(ax, x: float, y: float, s: str, colour: str, align: str = "left") -> None:
    """Upper-case mono label with extra letter spacing (one glyph at a time on a fixed advance)."""
    step = MONO_ADVANCE * TEXT_PX + TRACKING_PX
    start = x - (step * len(s) - TRACKING_PX) if align == "right" else x
    for i, ch in enumerate(s):
        put_text(ax, start + i * step, y, ch, colour)


def rounded_box(x: float, y: float, w: float, h: float, radius: float, **kwargs) -> FancyBboxPatch:
    """Rounded rectangle patch."""
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", **kwargs)


def circle(x: float, y: float, r: float, fc: str, ec: str = "none", lw: float = 0.0, alpha: float = 1.0,
           z: int = 3) -> Circle:
    """Circle patch (line width in pixels)."""
    return Circle((x, y), r, fc=fc, ec=ec, lw=lw * PT, alpha=alpha, zorder=z)


# --- frame layers --------------------------------------------------------------------------------------------------
def draw_static(ax, scene: Scene) -> None:
    """Card, rules, header and panel labels: identical in every frame."""
    ax.add_patch(rounded_box(1, 1, WIDTH - 2, HEIGHT - 2, style.RADIUS, fc=style.BG, ec=style.PANEL_EDGE, lw=2 * PT))
    spaced_text(ax, MARGIN, HEADER_Y, "FROM PROBABILITY MAP TO INSTANCES", style.SUN)
    spaced_text(ax, WIDTH - MARGIN, HEADER_Y, f"TEST IMAGE {scene.stem}", style.MUTED, align="right")
    for y in (RULE_TOP_Y, RULE_BOTTOM_Y):
        ax.plot([MARGIN, WIDTH - MARGIN], [y, y], color=style.GRID, lw=1.5 * PT)
    spaced_text(ax, MAP_X, EYEBROW_Y, "PROBABILITY MAP", style.MUTED)
    spaced_text(ax, TREE_X, EYEBROW_Y, "COMPONENT TREE", style.MUTED)
    spaced_text(ax, WIDTH - MARGIN, EYEBROW_Y, f"{len(scene.nodes)} CANDIDATES", style.MUTED, align="right")
    for level in range(len(LEVELS)):
        x = column_x(level)
        ax.plot([x, x], [GUIDE_Y0, GUIDE_Y1], color=style.GRID, lw=1.2 * PT, dashes=(2, 4))
    ax.plot([COL_X0, COL_X1], [TRACK_Y, TRACK_Y], color=style.PANEL_EDGE, lw=1.5 * PT)


def draw_map(ax, scene: Scene, t: float, fade: float) -> None:
    """Map panel: the glowing probability map, then the kept masks in translucent green on the dimmed map."""
    bg = np.array(to_rgb(style.BG))
    decide = ramp(t, DECIDE_START, DECIDE_START + DECIDE_DT)
    shown = ramp(t, *MAP_IN) * fade
    rgb = bg + (scene.map_rgb * (1 - MAP_DIM * decide) - bg) * shown
    green = np.array(to_rgb(style.GREEN))
    for node in scene.nodes:
        if node.kept:
            a = (GREEN_FILL_ALPHA * decide * fade * node.alpha)[..., None]
            rgb = rgb * (1 - a) + green * a
    frame = rounded_box(MAP_X, MAP_Y, MAP_PX, MAP_PX, MAP_RADIUS, fc="none", ec=style.PANEL_EDGE, lw=1.5 * PT,
                        zorder=3)
    image = ax.imshow(rgb, extent=(MAP_X, MAP_X + MAP_PX, MAP_Y + MAP_PX, MAP_Y), interpolation="nearest", zorder=1)
    ax.add_patch(frame)
    image.set_clip_path(frame)


def draw_outlines(ax, scene: Scene, t: float, fade: float) -> None:
    """Component outlines on the map in the colour of their threshold; in the last step only the kept ones remain."""
    decide = ramp(t, DECIDE_START, DECIDE_START + DECIDE_DT)
    for node in scene.nodes:
        show = level_ramp(t, node.level, OUTLINE) * fade
        level_colour = style.LEVEL_COLOURS[node.level]
        if node.kept:
            colour, alpha, width = style.blend(level_colour, style.GREEN, decide), show, 1.8 + decide
        else:
            colour, alpha, width = level_colour, show * (1 - decide), 1.8
        if alpha <= 0:
            continue
        for line in node.outline:
            ax.plot(line[:, 0], line[:, 1], color=colour, alpha=0.18 * alpha, lw=6 * PT, zorder=4)
            ax.plot(line[:, 0], line[:, 1], color=colour, alpha=alpha, lw=width * PT, zorder=5)


def draw_readout(ax, scene: Scene, t: float, fade: float) -> None:
    """Plate on the map: the current threshold during step 2, the number of kept candidates in step 4."""
    level = int(np.clip((t - TREE_START - OUTLINE[0]) // LEVEL_DT, 0, len(LEVELS) - 1))
    threshold = level_ramp(t, 0, OUTLINE) * (1 - ramp(t, *INDICATOR_OUT))
    kept = ramp(t, DECIDE_START, DECIDE_START + DECIDE_DT) * fade
    for alpha, label, colour in ((threshold, f"t ≥ {LEVELS[level]:.1f}", style.LEVEL_COLOURS[level]),
                                 (kept, f"KEPT {sum(n.kept for n in scene.nodes)} OF {len(scene.nodes)}", style.GREEN)):
        if alpha <= 0:
            continue
        width = MONO_ADVANCE * TEXT_PX * len(label) + 32
        ax.add_patch(rounded_box(MAP_X + 16, MAP_Y + 16, width, 44, 10, fc=style.BG, ec=style.PANEL_EDGE,
                                 lw=1.2 * PT, alpha=0.85 * alpha, zorder=6))
        put_text(ax, MAP_X + 32, MAP_Y + 38, label, colour, alpha=alpha, z=7)


def draw_scale(ax, t: float, fade: float) -> None:
    """Threshold scale above the tree: labels light up in their level colour and a marker climbs along the track."""
    for level, colour in enumerate(style.LEVEL_COLOURS):
        lit = level_ramp(t, level, OUTLINE) * fade
        shade = style.blend(style.blend(style.BG, colour, 0.35), colour, lit)
        put_text(ax, column_x(level), SCALE_LABEL_Y, f"{LEVELS[level]:.1f}", shade, ha="center")
        ax.add_patch(circle(column_x(level), TRACK_Y, 4, shade))
    shown = level_ramp(t, 0, MARKER) * (1 - ramp(t, *INDICATOR_OUT))
    if shown <= 0:
        return
    position = sum(level_ramp(t, lv, MARKER) for lv in range(1, len(LEVELS)))
    lower = int(np.floor(position))
    upper = min(lower + 1, len(LEVELS) - 1)
    colour = style.blend(style.LEVEL_COLOURS[lower], style.LEVEL_COLOURS[upper], position - lower)
    x = column_x(0) + (column_x(1) - column_x(0)) * position
    ax.add_patch(circle(x, TRACK_Y, 16, colour, alpha=0.18 * shown))
    ax.add_patch(circle(x, TRACK_Y, 8, colour, alpha=shown))


def node_colours(node: Node, t: float) -> tuple[str, str, str]:
    """Ring, fill and label colour of a node at time ``t``."""
    decide = ramp(t, DECIDE_START, DECIDE_START + DECIDE_DT)
    cyan = style.blend(style.PANEL, style.CYAN, 0.15 + 0.85 * min(node.score / SCORE_FULL, 1.0))
    fill = style.blend(style.BG, cyan, score_ramp(t, node.level))
    level_colour = style.LEVEL_COLOURS[node.level]
    if node.kept:
        return (style.blend(level_colour, style.GREEN, decide), style.blend(fill, style.GREEN, decide),
                style.blend(style.CYAN, style.GREEN, decide))
    rejected = style.blend(style.BG, style.SLATE, 0.45)
    return (style.blend(level_colour, style.SLATE, decide), style.blend(fill, rejected, decide),
            style.blend(style.CYAN, style.SLATE, decide))


def draw_tree(ax, scene: Scene, t: float, fade: float) -> None:
    """Edges, nodes, score labels and the green rings of the kept candidates."""
    decide = ramp(t, DECIDE_START, DECIDE_START + DECIDE_DT)
    edge_alpha = fade * (1 - 0.45 * decide)
    for node in scene.nodes:
        for child in node.children:
            grow = level_ramp(t, child.level, EDGE)
            if grow <= 0:
                continue
            pts = edge_points(node, child)[: max(2, int(round(grow * EDGE_SAMPLES)))]
            ax.plot(pts[:, 0], pts[:, 1], color=style.VIOLET, lw=5 * PT, alpha=0.15 * edge_alpha, zorder=2)
            ax.plot(pts[:, 0], pts[:, 1], color=style.VIOLET, lw=1.8 * PT, alpha=edge_alpha, zorder=2)
    for node in scene.nodes:
        grow = level_ramp(t, node.level, NODE) * fade
        if grow <= 0:
            continue
        ring, fill, label = node_colours(node, t)
        r = node.r * (0.4 + 0.6 * grow)
        ax.add_patch(circle(node.x, node.y, r + 5, ring, alpha=0.14 * grow, z=3))
        ax.add_patch(circle(node.x, node.y, r, fill, ring, 2.0, alpha=grow, z=4))
        if node.kept and decide > 0:
            ax.add_patch(circle(node.x, node.y, r + KEPT_RING, "none", style.GREEN, 2.0, alpha=decide * fade, z=4))
        scored = score_ramp(t, node.level) * fade
        if scored > 0:
            put_text(ax, node.x, node.y + node.r + LABEL_GAP, f"{node.score:.2f}", label, ha="center", alpha=scored)


def draw_objective(ax, scene: Scene, t: float, fade: float) -> None:
    """The objective of the programme for the kept set and, for comparison, for the root alone."""
    shown = ramp(t, *SUM_IN) * fade
    if shown <= 0:
        return
    put_text(ax, TREE_X, SUM_Y[0], scene.sum_lines[0], style.GREEN, alpha=shown)
    put_text(ax, TREE_X, SUM_Y[1], scene.sum_lines[1], style.SLATE, alpha=shown)


def draw_caption(ax, t: float) -> None:
    """Step caption (fades out and in between steps) and the four step markers on the right."""
    current = 0
    for i, (number, words) in enumerate(CAPTIONS):
        start = max(STEP_STARTS[i] - CAPTION_FADE, 0.0)
        end = STEP_STARTS[i + 1] - CAPTION_FADE if i + 1 < len(CAPTIONS) else TOTAL
        alpha = ramp(t, start, start + CAPTION_FADE) * (1 - ramp(t, end - CAPTION_FADE, end))
        current = i if t >= start else current
        if alpha <= 0:
            continue
        put_text(ax, MARGIN, CAPTION_Y, number, style.SUN, CAPTION_PX, alpha=alpha)
        put_text(ax, MARGIN + 40, CAPTION_Y, words, style.TEXT, CAPTION_PX, family=SANS, alpha=alpha)
    length, gap = STEP_BAR
    for i in range(len(CAPTIONS)):
        x = WIDTH - MARGIN - (len(CAPTIONS) - i) * (length + gap) + gap
        colour = style.SUN if i == current else (style.MUTED if i < current else style.PANEL_EDGE)
        ax.plot([x, x + length], [CAPTION_Y - 8, CAPTION_Y - 8], color=colour, lw=4 * PT, solid_capstyle="round")


def render(scene: Scene, t: float) -> np.ndarray:
    """One frame of the loop as an RGB array."""
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.patch.set_facecolor(style.BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.set_axis_off()
    fade = 1 - ramp(t, TOTAL - FADE_OUT, TOTAL)
    draw_static(ax, scene)
    draw_map(ax, scene, t, fade)
    draw_outlines(ax, scene, t, fade)
    draw_readout(ax, scene, t, fade)
    draw_scale(ax, t, fade)
    draw_tree(ax, scene, t, fade)
    draw_objective(ax, scene, t, fade)
    draw_caption(ax, t)
    rgb = style.figure_to_rgb(fig)
    plt.close(fig)
    return rgb


# --- output --------------------------------------------------------------------------------------------------------
def merge_still_frames(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Merge runs of identical frames into one frame with the summed duration."""
    kept, durations = [frames[0]], [FRAME_MS]
    for frame in frames[1:]:
        if np.array_equal(frame, kept[-1]):
            durations[-1] += FRAME_MS
        else:
            kept.append(frame)
            durations.append(FRAME_MS)
    return kept, durations


def build_scene() -> Scene:
    """Choose the example by the rule of the module docstring and prepare its drawing; prints the choice."""
    table = with_roots(scored_test_table())
    candidates = eligible_roots(table)
    print(f"eligible roots: {len(candidates)}")
    print(candidates.to_string(index=False))
    pick = candidates.iloc[0]
    sub = table[(table.stem == pick.stem) & (table.root == pick.root)]
    labels = selection.stored_label_images(fetch_artifact("test_components.npz"))(pick.stem)
    window = crop_window(labels(0), int(pick.root), int(pick.crop))
    y0, x0, side = window
    prob = np.load(fetch_artifact("test_probs_unet.npz"))[pick.stem][y0:y0 + side, x0:x0 + side]
    nodes = build_nodes(sub, labels, window)
    layout_tree(nodes)
    scene = Scene(str(pick.stem), nodes, glowing_map(prob), objective_lines(nodes))
    print(f"example: stem {scene.stem}, root {pick.root}, {len(nodes)} nodes, crop {side} px at (y {y0}, x {x0})")
    for node in nodes:
        print(f"  level {node.level} cid {node.cid:4d} parent {node.parent:4d} area {node.area:7.0f} "
              f"score {node.score:.4f}{'  KEPT' if node.kept else ''}")
    print("\n".join(scene.sum_lines))
    return scene


def write_key_frames(scene: Scene, folder: Path) -> None:
    """Key frames as PNG, and the final state at half size (the size on the README page)."""
    folder.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(KEY_TIMES):
        Image.fromarray(render(scene, t)).save(folder / f"key_{i:02d}_{t:05.2f}s.png")
    final = Image.fromarray(render(scene, FINAL + 0.5))
    final.resize((WIDTH // 2, HEIGHT // 2), Image.LANCZOS).save(folder / "final_half_size.png")


def main() -> None:
    """Build the scene, render the loop and write the GIF."""
    parser = argparse.ArgumentParser(description="Write docs/media/component_tree.gif from the stored artefacts.")
    parser.add_argument("--frames", type=Path, default=None, help="folder for key frames as PNG")
    args = parser.parse_args()
    style.apply_matplotlib_style()
    scene = build_scene()
    frames, durations = merge_still_frames([render(scene, t) for t in frame_times()])
    size = style.save_gif(frames, OUT, durations)
    print(f"wrote {OUT.relative_to(style.REPO)}: {len(frames)} frames, {sum(durations) / 1000:.2f} s, {size} bytes")
    if args.frames is not None:
        write_key_frames(scene, args.frames)


if __name__ == "__main__":
    main()
