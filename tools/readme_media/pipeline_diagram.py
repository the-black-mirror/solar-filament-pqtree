"""Animated pipeline diagram of the README: ``docs/media/pipeline.svg``.

Six stage cards, connected in the order in which the package runs them, between an input chip (the full-disk H-alpha
image) and an output chip (the run-length-encoded instance masks). The small pictures in the two chips and in four of
the cards are computed from one component tree of the test set, picked by a fixed rule (:func:`choose_tree`):

* input chip: the 2048 x 2048 frame with the solar disk, drawn like the one of the hero banner, and the position of the
  crop that the other pictures show;
* stage 1: the stored test probability map on that crop, averaged over a 30 x 8 grid (``test_probs_unet.npz``);
* stages 2 and 6: the component tree itself (``candidates_test.tbl.npz``), coloured by threshold level in stage 2 and by
  the decision of the stored hurdle model followed by the antichain selection at ``config.TAU`` in stage 6;
* stage 4: the outline of the best-matching instance of each detector family around the largest kept candidate
  (``detector_instances_test.csv.gz``, ``test_components.npz``);
* output chip: the masks of ``reference/submission.csv`` on the same 30 x 8 grid, drawn as runs.

Stages 3 and 5 are schematic (a stack of four crop channels feeding network layers; two decision trees joined by a
multiplication sign). Crops of a tree that is taller than wide are turned by 90 degrees clockwise so that they fill the
wide picture boxes; the marker in the input chip shows the crop where it lies in the image.

The animation is CSS only: dashes flow along the connectors, a highlight travels from tile to tile in data order and
the pictures breathe slightly. Text never moves, and every element is complete in the resting state.

    python tools/readme_media/pipeline_diagram.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from style import TextLayer, num  # noqa: E402
from pqtree import dense_maps, lgbm_scorer, pipeline, selection  # noqa: E402
from pqtree.artifacts import fetch_artifact  # noqa: E402
from pqtree.config import (CANDIDATE_FEATURES, CNN_CONFIG, DISK_RADIUS, FAMILIES, LEVELS, MODEL_FEATURES,  # noqa: E402
                           N_FOLDS, TAU, UNCERTAINTY_FEATURES, H, W)

OUT_PATH = style.MEDIA / "pipeline.svg"

# --- canvas and grid -------------------------------------------------------------------------------------------------
WIDTH, HEIGHT = 1760, 640
MARGIN = 48
GAP = 28
"""Horizontal gap between two tiles of a row (the visible part of a connector)."""
ROW_GAP = 26
CHIP_W = 281
"""Width of the input and output chips (set by the widest chip line, 'instance masks,')."""
CARD_W = (WIDTH - 2 * MARGIN - CHIP_W - 3 * GAP) / 3
ROW_H = (HEIGHT - 2 * MARGIN - ROW_GAP) / 2
ROW_Y = (MARGIN, MARGIN + ROW_H + ROW_GAP)
TILE_RADIUS = 14
BOTTOM_PAD = 16
"""Smallest distance between the last text baseline of a tile and its lower edge."""

# --- type and positions inside a tile (offsets from its top-left corner) ---------------------------------------------
EYEBROW_PX, NAME_PX, FACT_PX = 26, 28, 26
EYEBROW_SPACING = 3
PAD, CHIP_PAD = 20, 14
EYEBROW_Y, NAME_Y, FACT_Y = 40, 100, 140
LINE_STEP, FACT_STEP = 30, 36
PICTO_W, PICTO_H, PICTO_TOP = 208, 52, 14
PICTO_GAP = 16
"""Free space between the eyebrow and the picture of a card."""
ICON_TOP = 58
DISK_ICON_H, RUNS_ICON_H = 104, 80
"""Heights of the input-chip icon (two text lines below it) and of the output-chip icon (three lines)."""
DISK_ICON_GLOW = 6.0
"""Width of the glow outside the limb of the input-chip disk (the hero's glow scaled to the smaller disk)."""
DISK_ICON_GRATICULE_STEP = 5.0
"""Sampling step (degrees) of the graticule arcs of the input-chip disk."""
ICON_TO_LINE = 41
"""Distance from the lower edge of a chip icon to the first text baseline."""

# --- pictures --------------------------------------------------------------------------------------------------------
GRID_COLS, GRID_ROWS = 30, 8
"""Cells of the probability grid (stage 1) and of the output runs (output chip)."""
CROP_FILL = 1.12
"""The crop is this much larger than the bounding box of the tree's root component."""
MIN_CELL_OPACITY = 0.07
GLOW_PROBABILITY = 0.5
TREE_NODE_R, KEPT_NODE_R = 4.5, 5.5
MAX_TREE_ROW_STEP = 24.0
OUTLINE_SIMPLIFY_PX = 1.0
OUTLINE_PAD = 0.15
DETECTOR_OPACITY = (0.95, 0.75, 0.55, 0.4)
"""Stroke opacity of the outlines of the four detector families (FAMILIES order)."""

# --- animation -------------------------------------------------------------------------------------------------------
HIGHLIGHT_STEP_S = 1.3
"""Time between the highlights of two consecutive tiles; eight tiles give one pass of 10.4 s."""
FLOW_PERIOD_PX, FLOW_DASH_PX, FLOW_S = 16, 6, 1.2
BREATHE_S, BREATHE_PHASE_S = 6.0, 0.75

Picture = Callable[[float, float], str]
"""Draws a picture into the box whose top-left corner is given."""


@dataclass(frozen=True)
class Stage:
    """One card: name, colour of its eyebrow and highlight, and its two facts (each a tuple of display lines)."""
    name: str
    accent: str
    facts: tuple[tuple[str, ...], ...]


STAGES = (
    Stage("Dense maps", style.CYAN, (("10 U-Nets × 8 views", "at 2048 px"),
                                     ("probability, view", "uncertainty, family gap"))),
    Stage("Component tree", style.VIOLET, (("8 thresholds,", "0.2 to 0.9"), ("33 shape and intensity", "descriptors"))),
    Stage("CNN scorer", style.CYAN, (("ResNet18 on", "4-channel crops"), ("5-fold cross-fitted",))),
    Stage("Detector evidence", style.CYAN, (("4 detector families", "× 5 folds"),
                                            ("IoU, confidence,", "coverage, count"))),
    Stage("Hurdle LightGBM", style.CYAN, (("P(hit) × matched IoU",), ("52 inputs",))),
    Stage("Antichain selection", style.GREEN, (("exact dynamic programme", "on the tree"),
                                               ("keep where", "score > τ = 0.34"))),
)
INPUT_LINES = (("H-alpha image,", False), ("2048 × 2048", True))
OUTPUT_LINES = (("instance masks,", False), ("run-length", False), ("encoded", False))
"""Lines of the two chips as (text, set in mono)."""


def check_facts() -> None:
    """Raise if a number printed on the cards disagrees with the package configuration."""
    checks = {
        "image side": W == H == 2048,
        "views": len(dense_maps.D4) == 8,
        "thresholds": len(LEVELS) == 8 and (LEVELS[0], LEVELS[-1]) == (0.2, 0.9),
        "descriptors": len(CANDIDATE_FEATURES) + len(UNCERTAINTY_FEATURES) == 33,
        "cnn": CNN_CONFIG["backbone"] == "resnet18" and CNN_CONFIG["folds"] == 5,
        "detectors": len(FAMILIES) == 4 and N_FOLDS == 5,
        "inputs": len(MODEL_FEATURES) == 52,
        "tau": TAU == 0.34,
    }
    wrong = [k for k, ok in checks.items() if not ok]
    if wrong:
        raise ValueError(f"card facts disagree with pqtree.config: {wrong}")


# ---------------------------------------------------------------------------------------------------------------------
# data: one component tree of the test set
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class TreeExample:
    """Everything the pictures show about the chosen tree."""
    stem: str
    nodes: pd.DataFrame
    crop: tuple[int, int, int, int]
    prob_cells: np.ndarray
    output_cells: np.ndarray
    outlines: list[np.ndarray]


def scored_test_candidates() -> pd.DataFrame:
    """Test candidates with the stored hurdle model's score, the selection decision and the root of their tree."""
    table = pipeline.build_test_table()
    model = lgbm_scorer.HurdleModel.load(fetch_artifact(lgbm_scorer.HIT_FILE), fetch_artifact(lgbm_scorer.IOU_FILE))
    score = model.predict(table)
    chosen = set(selection.select(table, score, TAU))
    table = table.assign(score=score, kept=[k in chosen for k in zip(table.stem, table.level, table.cid)])
    table = table.sort_values(["stem", "level", "cid"]).reset_index(drop=True)
    root: dict = {}
    for stem, level, cid, parent in zip(table.stem, table.level, table.cid, table.parent):
        root[(stem, level, cid)] = cid if level == 0 else root[(stem, level - 1, parent)]
    return table.assign(root=[root[k] for k in zip(table.stem, table.level, table.cid)])


def tree_summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per tree: number of nodes, deepest level, leaves, kept nodes and kept nodes that have children."""
    deeper = table[table.level > 0]
    has_child = set(zip(deeper.stem, deeper.level - 1, deeper.parent))
    leaf = np.array([k not in has_child for k in zip(table.stem, table.level, table.cid)])
    t = table.assign(leaf=leaf, kept_inner=table.kept.to_numpy() & ~leaf)
    return t.groupby(["stem", "root"]).agg(n=("cid", "size"), depth=("level", "max"), leaves=("leaf", "sum"),
                                           kept=("kept", "sum"), kept_inner=("kept_inner", "sum")).reset_index()


def choose_tree(table: pd.DataFrame) -> tuple[str, int]:
    """First tree (by image and root id) that spans all eight levels, has two or three leaves, at most 14 nodes and at
    least two kept nodes, one of which has children: a small tree on which the antichain selection is visible."""
    s = tree_summary(table)
    ok = s[(s.depth == len(LEVELS) - 1) & s.leaves.between(2, 3) & (s.n <= 14) & (s.kept >= 2) & (s.kept_inner >= 1)]
    if ok.empty:
        raise ValueError("no tree satisfies the selection rule")
    first = ok.sort_values(["stem", "root"]).iloc[0]
    return str(first.stem), int(first.root)


def crop_around(ys: np.ndarray, xs: np.ndarray, turn: bool) -> tuple[int, int, int, int]:
    """Crop (x0, y0, x1, y1) around a component with the aspect of the GRID_COLS x GRID_ROWS grid after turning."""
    long_ext, short_ext = (np.ptp(ys) + 1, np.ptp(xs) + 1) if turn else (np.ptp(xs) + 1, np.ptp(ys) + 1)
    cell = max(long_ext * CROP_FILL / GRID_COLS, short_ext * CROP_FILL / GRID_ROWS)
    ext_x, ext_y = (GRID_ROWS * cell, GRID_COLS * cell) if turn else (GRID_COLS * cell, GRID_ROWS * cell)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x0, y0 = int(round(cx - ext_x / 2)), int(round(cy - ext_y / 2))
    return max(x0, 0), max(y0, 0), min(x0 + int(round(ext_x)), W), min(y0 + int(round(ext_y)), H)


def is_turned(crop: tuple[int, int, int, int]) -> bool:
    """A crop that is taller than wide is shown turned by 90 degrees clockwise."""
    x0, y0, x1, y1 = crop
    return (y1 - y0) > (x1 - x0)


def to_view(a: np.ndarray, turn: bool) -> np.ndarray:
    """Array as it is shown: turned by 90 degrees clockwise if ``turn``."""
    return np.ascontiguousarray(np.rot90(a, -1) if turn else a)


def grid_means(a: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Mean of ``a`` over each cell of the GRID_ROWS x GRID_COLS grid laid over the shown crop."""
    x0, y0, x1, y1 = crop
    view = to_view(a[y0:y1, x0:x1].astype(np.float32), is_turned(crop))
    return cv2.resize(view, (GRID_COLS, GRID_ROWS), interpolation=cv2.INTER_AREA)


def rle_of(text: str) -> dict:
    """COCO run-length dict of an H x W mask from its counts string."""
    return {"size": [H, W], "counts": text.encode("ascii")}


def outline(mask: np.ndarray, crop: tuple[int, int, int, int], turn: bool) -> np.ndarray:
    """Simplified outer contour (N x 2, shown pixels of the crop) of the largest part of ``mask`` inside ``crop``."""
    x0, y0, x1, y1 = crop
    view = to_view(mask[y0:y1, x0:x1].astype(np.uint8), turn)
    contours, _ = cv2.findContours(view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(largest, OUTLINE_SIMPLIFY_PX, True)[:, 0, :].astype(np.float64)


def detector_outlines(stem: str, target: np.ndarray, turn: bool) -> list[np.ndarray]:
    """Outline of the instance with the highest IoU with ``target``, for each detector family (FAMILIES order)."""
    inst = pd.read_csv(fetch_artifact("detector_instances_test.csv.gz"))
    inst = inst[inst.stem == stem]
    target_rle = mask_utils.encode(np.asfortranarray(target.astype(np.uint8)))
    ys, xs = np.nonzero(target)
    pad_y, pad_x = int(np.ptp(ys) * OUTLINE_PAD) + 2, int(np.ptp(xs) * OUTLINE_PAD) + 2
    crop = (max(int(xs.min()) - pad_x, 0), max(int(ys.min()) - pad_y, 0), min(int(xs.max()) + pad_x + 1, W),
            min(int(ys.max()) + pad_y + 1, H))
    found = []
    for family in FAMILIES:
        rles = [rle_of(t) for t in inst[inst.family == family].rle]
        ious = mask_utils.iou(rles, [target_rle], [0])[:, 0]
        best = int(np.argmax(ious))
        print(f"  {family}: {len(rles)} instances, best IoU with the kept candidate {ious[best]:.3f}")
        found.append(outline(mask_utils.decode(rles[best]).astype(bool), crop, turn))
    return found


def load_example() -> TreeExample:
    """Choose the tree and compute the probability grid, the output runs and the detector outlines on its crop."""
    table = scored_test_candidates()
    stem, root = choose_tree(table)
    nodes = table[(table.stem == stem) & (table.root == root)][["level", "cid", "parent", "area", "score", "kept"]]
    labels = selection.stored_label_images(fetch_artifact("test_components.npz"))(stem)
    ys, xs = np.nonzero(labels(0) == root)
    crop = crop_around(ys, xs, bool(np.ptp(ys) > np.ptp(xs)))
    prob = np.load(fetch_artifact("test_probs_unet.npz"))[stem] / 255.0
    sub = pd.read_csv(style.REPO / "reference" / "submission.csv")
    kept_union = np.zeros((H, W), dtype=np.float32)
    for counts in sub[sub.filament_id.str.rsplit("_", n=1).str[0] == stem].segmentation_rle:
        kept_union[mask_utils.decode(rle_of(counts)).astype(bool)] = 1.0
    largest = nodes[nodes.kept].sort_values("area", ascending=False).iloc[0]
    target = labels(int(largest.level)) == int(largest.cid)
    print(f"tree {stem} root {root}: {len(nodes)} nodes, {int(nodes.kept.sum())} kept, crop {crop}, "
          f"turned {is_turned(crop)}")
    print(nodes.to_string(index=False))
    return TreeExample(stem, nodes.reset_index(drop=True), crop, grid_means(prob, crop),
                       grid_means(kept_union, crop) >= 0.5, detector_outlines(stem, target, is_turned(crop)))


# ---------------------------------------------------------------------------------------------------------------------
# pictures (each drawn into a box whose top-left corner is bx, by)
# ---------------------------------------------------------------------------------------------------------------------
def probability_grid(cells: np.ndarray, bx: float, by: float) -> str:
    """Stage 1: one square per grid cell, CYAN with an opacity that follows the mean probability; bright cells glow."""
    rows, cols = cells.shape
    pitch = min(PICTO_W / cols, PICTO_H / rows)
    x0 = bx + PICTO_W - cols * pitch
    base, glow = [], []
    for (r, c), p in np.ndenumerate(cells):
        cell = (f'<rect x="{num(x0 + c * pitch)}" y="{num(by + r * pitch)}" width="{num(pitch - 1.2)}" '
                f'height="{num(pitch - 1.2)}" rx="1" fill="{style.CYAN}" ')
        base.append(cell + f'fill-opacity="{MIN_CELL_OPACITY + (1 - MIN_CELL_OPACITY) * float(p):.2f}"/>')
        if p >= GLOW_PROBABILITY:
            glow.append(cell + 'fill-opacity="0.8"/>')
    return f'<g filter="url(#glow-soft)" opacity="0.55">{"".join(glow)}</g>\n' + "\n".join(base) + "\n"


def tree_positions(nodes: pd.DataFrame, bx: float, by: float) -> dict[tuple[int, int], tuple[float, float]]:
    """Node centres (level, cid) -> (x, y): x by threshold level, y by leaf order (larger children first)."""
    children: dict = {}
    for r in nodes.sort_values(["area", "cid"], ascending=[False, True]).itertuples():
        if r.level > 0:
            children.setdefault((int(r.level) - 1, int(r.parent)), []).append((int(r.level), int(r.cid)))
    rows: dict = {}
    leaves: list = []

    def place(key: tuple[int, int]) -> float:
        """Row of node ``key``: the next free row for a leaf, the mean row of its children otherwise."""
        kids = children.get(key, [])
        if kids:
            rows[key] = float(np.mean([place(k) for k in kids]))
        else:
            rows[key] = float(len(leaves))
            leaves.append(key)
        return rows[key]

    place((0, int(nodes[nodes.level == 0].cid.iloc[0])))
    step_x = (PICTO_W - 2 * KEPT_NODE_R - 2) / (len(LEVELS) - 1)
    step_y = min(MAX_TREE_ROW_STEP, (PICTO_H - 2 * KEPT_NODE_R - 4) / max(len(leaves) - 1, 1))
    centre = (len(leaves) - 1) / 2
    x0, mid = bx + KEPT_NODE_R + 1, by + PICTO_H / 2
    return {k: (x0 + k[0] * step_x, mid + (v - centre) * step_y) for k, v in rows.items()}


def tree_picture(nodes: pd.DataFrame, bx: float, by: float, by_decision: bool) -> str:
    """Stages 2 and 6: the tree with VIOLET edges; nodes in LEVEL_COLOURS, or GREEN (kept) and SLATE (rejected)."""
    pos = tree_positions(nodes, bx, by)
    parts = []
    for r in nodes.itertuples():
        if r.level > 0:
            (x0, y0), (x1, y1) = pos[(int(r.level) - 1, int(r.parent))], pos[(int(r.level), int(r.cid))]
            parts.append(f'<line x1="{num(x0)}" y1="{num(y0)}" x2="{num(x1)}" y2="{num(y1)}" stroke="{style.VIOLET}" '
                         'stroke-opacity="0.7" stroke-width="1.6"/>')
    for r in nodes.itertuples():
        x, y = pos[(int(r.level), int(r.cid))]
        if not by_decision:
            parts.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{TREE_NODE_R}" '
                         f'fill="{style.LEVEL_COLOURS[int(r.level)]}"/>')
        elif r.kept:
            parts.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{KEPT_NODE_R + 4}" fill="{style.GREEN}" '
                         'opacity="0.5" filter="url(#glow-soft)"/>')
            parts.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{KEPT_NODE_R}" fill="{style.GREEN}"/>')
        else:
            parts.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{TREE_NODE_R - 0.5}" fill="{style.PANEL}" '
                         f'stroke="{style.SLATE}" stroke-width="1.6"/>')
    return "\n".join(parts) + "\n"


def chevron(x: float, y: float, horizontal: bool, colour: str, size: float = 5.0) -> str:
    """Open arrowhead whose tip is at (x, y), pointing right (horizontal) or down."""
    pts = ([(x - size - 1, y - size), (x, y), (x - size - 1, y + size)] if horizontal
           else [(x - size, y - size - 1), (x, y), (x + size, y - size - 1)])
    return (f'<path d="{style.path_from_points(pts, closed=False)}" fill="none" stroke="{colour}" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/>\n')


def cnn_picture(bx: float, by: float) -> str:
    """Stage 3 (schematic): four crop channels feeding three network layers that end in one score."""
    side, step, mid = 40, 4, by + PICTO_H / 2
    parts = []
    for i in range(4):
        parts.append(f'<rect x="{num(bx + step * (3 - i))}" y="{num(by + step * i)}" width="{side}" height="{side}" '
                     f'rx="4" fill="{style.BG_SOFT}" stroke="{style.CYAN}" stroke-opacity="{0.3 + 0.2 * i:.1f}" '
                     'stroke-width="1.5"/>')
    parts.append(f'<line x1="{num(bx + 60)}" y1="{num(mid)}" x2="{num(bx + 76)}" y2="{num(mid)}" '
                 f'stroke="{style.MUTED}" stroke-width="1.6"/>')
    parts.append(chevron(bx + 77, mid, True, style.MUTED, 4))
    for j, height in enumerate((52, 38, 24)):
        parts.append(f'<rect x="{num(bx + 88 + j * 24)}" y="{num(mid - height / 2)}" width="12" height="{height}" '
                     f'rx="3" fill="{style.VIOLET}" fill-opacity="0.18" stroke="{style.VIOLET}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{num(bx + 154)}" y1="{num(mid)}" x2="{num(bx + 172)}" y2="{num(mid)}" '
                 f'stroke="{style.MUTED}" stroke-width="1.6"/>')
    parts.append(chevron(bx + 173, mid, True, style.MUTED, 4))
    parts.append(f'<circle cx="{num(bx + 192)}" cy="{num(mid)}" r="12" fill="{style.CYAN}" opacity="0.5" '
                 'filter="url(#glow-soft)"/>')
    parts.append(f'<circle cx="{num(bx + 192)}" cy="{num(mid)}" r="7" fill="{style.CYAN}"/>')
    return "\n".join(parts) + "\n"


def outlines_picture(outlines: Sequence[np.ndarray], bx: float, by: float) -> str:
    """Stage 4: the outlines of the detector instances, scaled together into the box, one opacity per family."""
    pts = np.vstack(outlines)
    lo, span = pts.min(axis=0), np.ptp(pts, axis=0) + 1
    s = 0.94 * min(PICTO_W / span[0], PICTO_H / span[1])
    off = np.array([bx + (PICTO_W - s * span[0]) / 2, by + (PICTO_H - s * span[1]) / 2]) - s * lo
    return "".join(f'<path d="{style.path_from_points(off + s * o)}" fill="{style.CYAN}" fill-opacity="0.04" '
                   f'stroke="{style.CYAN}" stroke-opacity="{op}" stroke-width="1.6" stroke-linejoin="round"/>\n'
                   for o, op in zip(outlines, DETECTOR_OPACITY))


def hurdle_picture(bx: float, by: float) -> str:
    """Stage 5 (schematic): two depth-2 decision trees (hit probability, matched IoU) joined by a times sign."""
    parts = []
    for cx in (bx + 40, bx + 168):
        inner = [(cx, by + 6), (cx - 18, by + 26), (cx + 18, by + 26)]
        leaves = [(cx - 27, by + 46), (cx - 9, by + 46), (cx + 9, by + 46), (cx + 27, by + 46)]
        for a, b in [(0, 1), (0, 2)]:
            parts.append(f'<line x1="{num(inner[a][0])}" y1="{num(inner[a][1])}" x2="{num(inner[b][0])}" '
                         f'y2="{num(inner[b][1])}" stroke="{style.VIOLET}" stroke-opacity="0.7" stroke-width="1.6"/>')
        for i, (lx, ly) in enumerate(leaves):
            px, py = inner[1 + i // 2]
            parts.append(f'<line x1="{num(px)}" y1="{num(py)}" x2="{num(lx)}" y2="{num(ly)}" stroke="{style.VIOLET}" '
                         'stroke-opacity="0.7" stroke-width="1.6"/>')
        parts += [f'<circle cx="{num(x)}" cy="{num(y)}" r="4" fill="{style.PANEL}" stroke="{style.VIOLET}" '
                  'stroke-width="1.6"/>' for x, y in inner]
        parts += [f'<circle cx="{num(x)}" cy="{num(y)}" r="4" fill="{style.CYAN}"/>' for x, y in leaves]
    cx, cy, arm = bx + 104, by + PICTO_H / 2, 7
    parts.append(f'<path d="M {num(cx - arm)} {num(cy - arm)} L {num(cx + arm)} {num(cy + arm)} M {num(cx + arm)} '
                 f'{num(cy - arm)} L {num(cx - arm)} {num(cy + arm)}" stroke="{style.TEXT}" stroke-width="2" '
                 'stroke-linecap="round"/>')
    return "\n".join(parts) + "\n"


def disk_icon(crop: tuple[int, int, int, int], bx: float, by: float) -> str:
    """Input chip: the image frame, the solar disk and a marker on the crop that the card pictures show.

    The disk is drawn like the one of the hero banner (dark disk, soft glow, faint graticule, thin orange limb); at this
    size the graticule is shown with the solar equator through the centre. The two gradients of the disk are defined
    here, so that the icon is self-contained.
    """
    s = DISK_ICON_H / W
    x0, y0, x1, y1 = crop
    cx, cy, r = bx + DISK_ICON_H / 2, by + DISK_ICON_H / 2, DISK_RADIUS * s
    mx, my = bx + (x0 + x1) / 2 * s, by + (y0 + y1) / 2 * s
    centre = f'cx="{num(cx)}" cy="{num(cy)}"'
    halo = style.gradient_stops(style.halo_stops(r / (r + DISK_ICON_GLOW)))
    fill = style.gradient_stops(style.DISK_FILL_STOPS)
    graticule = style.graticule_path((cx, cy), r, b0=0.0, step=DISK_ICON_GRATICULE_STEP)
    return (f'<rect x="{num(bx)}" y="{num(by)}" width="{DISK_ICON_H}" height="{DISK_ICON_H}" rx="4" fill="{style.BG}" '
            f'stroke="{style.PANEL_EDGE}" stroke-width="1.2"/>\n'
            f'<defs><radialGradient id="disk-halo">{halo}</radialGradient>'
            f'<radialGradient id="disk-fill">{fill}</radialGradient></defs>\n'
            f'<circle {centre} r="{num(r + DISK_ICON_GLOW)}" fill="url(#disk-halo)"/>\n'
            f'<circle {centre} r="{num(r)}" fill="url(#disk-fill)"/>\n'
            f'<path d="{graticule}" fill="none" stroke="{style.PANEL_EDGE}" stroke-width="1"/>\n'
            f'<circle {centre} r="{num(r)}" fill="none" stroke="{style.SUN}" stroke-width="1.2" '
            'stroke-opacity="0.9"/>\n'
            f'<rect x="{num(bx + x0 * s)}" y="{num(by + y0 * s)}" width="{num((x1 - x0) * s)}" '
            f'height="{num((y1 - y0) * s)}" fill="none" stroke="{style.CYAN}" stroke-width="1.4"/>\n'
            f'<circle cx="{num(mx)}" cy="{num(my)}" r="10" fill="none" stroke="{style.TEXT}" stroke-opacity="0.75" '
            'stroke-width="1.2"/>\n')


def runs_icon(cells: np.ndarray, bx: float, by: float, width: float) -> str:
    """Output chip: each grid row of the kept masks drawn as runs (GREEN) on a faint SLATE base line."""
    rows, cols = cells.shape
    px, py = width / cols, RUNS_ICON_H / rows
    base, runs = [], []
    for r in range(rows):
        y = by + r * py
        base.append(f'<line x1="{num(bx)}" y1="{num(y + py / 2)}" x2="{num(bx + width)}" y2="{num(y + py / 2)}" '
                    f'stroke="{style.SLATE}" stroke-opacity="0.45" stroke-width="1"/>')
        edges = np.flatnonzero(np.diff(np.r_[0, cells[r].astype(int), 0]))
        for j0, j1 in zip(edges[::2], edges[1::2]):
            runs.append(f'<rect x="{num(bx + j0 * px)}" y="{num(y + 2)}" width="{num((j1 - j0) * px - 1.5)}" '
                        f'height="{num(py - 4)}" rx="2" fill="{style.GREEN}"/>')
    return ("\n".join(base) + f'\n<g filter="url(#glow-soft)" opacity="0.5">{"".join(runs)}</g>\n'
            + "\n".join(runs) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# tiles, connectors, animation
# ---------------------------------------------------------------------------------------------------------------------
def tile_layout() -> list[tuple[float, float, float]]:
    """(x, y, width) of the eight tiles in data order: input chip, six cards, output chip."""
    row1 = [MARGIN + CHIP_W + GAP + k * (CARD_W + GAP) for k in range(3)]
    row2 = [MARGIN + k * (CARD_W + GAP) for k in range(3)]
    tiles = ([(MARGIN, ROW_Y[0], CHIP_W)] + [(x, ROW_Y[0], CARD_W) for x in row1]
             + [(x, ROW_Y[1], CARD_W) for x in row2] + [(row2[-1] + CARD_W + GAP, ROW_Y[1], CHIP_W)])
    if abs(tiles[3][0] + CARD_W - (WIDTH - MARGIN)) > 1e-6 or abs(tiles[7][0] + CHIP_W - (WIDTH - MARGIN)) > 1e-6:
        raise ValueError("tiles do not fill the rows")
    return tiles


def tile_rect(x: float, y: float, w: float, fill: str, stroke: str, extra: str = "") -> str:
    """Rounded rectangle of a tile."""
    return (f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(ROW_H)}" rx="{TILE_RADIUS}" fill="{fill}" '
            f'stroke="{stroke}"{extra}/>\n')


def card(layer: TextLayer, x: float, y: float, number: int, stage: Stage, picture: Picture) -> str:
    """Stage card: eyebrow, picture, name and facts."""
    bx, by = x + CARD_W - PAD - PICTO_W, y + PICTO_TOP
    parts = [tile_rect(x, y, CARD_W, style.PANEL, style.PANEL_EDGE, ' stroke-width="1.5"'),
             layer.text(f"STAGE {number}", x + PAD, y + EYEBROW_Y, EYEBROW_PX, bx - PICTO_GAP - (x + PAD),
                        stage.accent, mono=True, spacing=EYEBROW_SPACING),
             f'<g class="breathe b{number}">\n{picture(bx, by)}</g>\n',
             layer.text(stage.name, x + PAD, y + NAME_Y, NAME_PX, CARD_W - 2 * PAD, style.TEXT, bold=True)]
    baseline = y + FACT_Y
    for fact in stage.facts:
        for line in fact:
            parts.append(layer.text(line, x + PAD, baseline, FACT_PX, CARD_W - 2 * PAD, style.MUTED))
            baseline += LINE_STEP
        baseline += FACT_STEP - LINE_STEP
    if baseline - FACT_STEP > y + ROW_H - BOTTOM_PAD:
        raise ValueError(f"facts of stage {number} do not fit the card")
    return "".join(parts)


def chip(layer: TextLayer, x: float, y: float, index: int, label: str, accent: str, icon: tuple[str, float],
         lines: Sequence[tuple[str, bool]]) -> str:
    """Input or output chip: eyebrow, icon (SVG and height) and the text lines below the icon."""
    line_y = y + ICON_TOP + icon[1] + ICON_TO_LINE
    parts = [tile_rect(x, y, CHIP_W, style.BG_SOFT, accent, ' stroke-opacity="0.55" stroke-width="1.5"'),
             layer.text(label, x + CHIP_PAD, y + EYEBROW_Y, EYEBROW_PX, CHIP_W - 2 * CHIP_PAD, accent, mono=True,
                        spacing=EYEBROW_SPACING),
             f'<g class="breathe b{index}">\n{icon[0]}</g>\n']
    for i, (text, mono) in enumerate(lines):
        parts.append(layer.text(text, x + CHIP_PAD, line_y + i * LINE_STEP, FACT_PX, CHIP_W - 2 * CHIP_PAD,
                                style.MUTED if mono else style.TEXT, mono=mono))
    if line_y + (len(lines) - 1) * LINE_STEP > y + ROW_H - BOTTOM_PAD:
        raise ValueError(f"lines of chip {label} do not fit")
    return "".join(parts)


def connectors(tiles: Sequence[tuple[float, float, float]]) -> tuple[list[str], str]:
    """Path data of the connectors between consecutive tiles (in data direction) and their arrowheads."""
    paths, heads, radius = [], [], 6.0
    for (xa, ya, wa), (xb, yb, wb) in zip(tiles, tiles[1:]):
        if ya == yb:
            mid = ya + ROW_H / 2
            paths.append(f"M {num(xa + wa)} {num(mid)} H {num(xb - 1)}")
            heads.append(chevron(xb - 1, mid, True, style.CYAN))
        else:
            sx, sy, ex, ey = xa + wa / 2, ya + ROW_H, xb + wb / 2, yb
            my = sy + ROW_GAP / 2
            paths.append(f"M {num(sx)} {num(sy)} V {num(my - radius)} Q {num(sx)} {num(my)} {num(sx - radius)} "
                         f"{num(my)} H {num(ex + radius)} Q {num(ex)} {num(my)} {num(ex)} {num(my + radius)} "
                         f"V {num(ey - 1)}")
            heads.append(chevron(ex, ey - 1, False, style.CYAN))
    return paths, "".join(heads)


def css(n_tiles: int) -> str:
    """The one style block: flowing dashes, travelling highlight, breathing pictures, reduced-motion rule."""
    cycle = n_tiles * HIGHLIGHT_STEP_S
    rules = [
        f".flow {{ stroke-dasharray: {FLOW_DASH_PX} {FLOW_PERIOD_PX - FLOW_DASH_PX}; "
        f"animation: flow {FLOW_S}s linear infinite; }}",
        f"@keyframes flow {{ to {{ stroke-dashoffset: -{FLOW_PERIOD_PX}; }} }}",
        f".hl {{ animation: highlight {num(cycle)}s ease-in-out infinite; }}",
        "@keyframes highlight { 0% { opacity: 0; } 4% { opacity: 1; } 12% { opacity: 0; } 100% { opacity: 0; } }",
        f".breathe {{ animation: breathe {num(BREATHE_S)}s ease-in-out infinite; }}",
        "@keyframes breathe { 0%, 100% { opacity: 1; } 50% { opacity: 0.8; } }",
    ]
    rules += [f".h{i} {{ animation-delay: {round(i * HIGHLIGHT_STEP_S, 2):g}s; }}" for i in range(n_tiles)]
    rules += [f".b{i} {{ animation-delay: {-i * BREATHE_PHASE_S:g}s; }}" for i in range(n_tiles)]
    rules.append(style.REDUCED_MOTION_CSS)
    return "<style>\n" + "\n".join(rules) + "\n</style>\n"


def description() -> str:
    """Text alternative of the figure."""
    stages = "; ".join(f"stage {i} {s.name}: " + ", ".join(" ".join(f) for f in s.facts)
                       for i, s in enumerate(STAGES, 1))
    return (f"From an H-alpha image (2048 x 2048) to instance masks (run-length encoded) in six stages: {stages}. "
            "The pictures show one component tree of the test set.")


def build_svg(example: TreeExample, layer: TextLayer) -> str:
    """The complete SVG document."""
    tiles = tile_layout()
    paths, heads = connectors(tiles)
    pictures: list[Picture] = [
        lambda bx, by: probability_grid(example.prob_cells, bx, by),
        lambda bx, by: tree_picture(example.nodes, bx, by, by_decision=False),
        cnn_picture,
        lambda bx, by: outlines_picture(example.outlines, bx, by),
        hurdle_picture,
        lambda bx, by: tree_picture(example.nodes, bx, by, by_decision=True),
    ]
    accents = [style.SUN] + [s.accent for s in STAGES] + [style.GREEN]
    parts = [style.svg_open(WIDTH, HEIGHT, "Pipeline", escape(description())), style.svg_defs(), css(len(tiles)),
             style.svg_background(WIDTH, HEIGHT)]
    parts += [f'<path d="{d}" fill="none" stroke="{style.PANEL_EDGE}" stroke-width="2"/>\n' for d in paths]
    parts += [f'<path class="flow" d="{d}" fill="none" stroke="{style.CYAN}" stroke-opacity="0.9" stroke-width="2"/>\n'
              for d in paths]
    x, y, _ = tiles[0]
    disk = disk_icon(example.crop, x + CHIP_PAD, y + ICON_TOP)
    parts.append(chip(layer, x, y, 0, "INPUT", style.SUN, (disk, DISK_ICON_H), INPUT_LINES))
    for number, ((x, y, _), stage, picture) in enumerate(zip(tiles[1:7], STAGES, pictures), 1):
        parts.append(card(layer, x, y, number, stage, picture))
    x, y, _ = tiles[7]
    runs = runs_icon(example.output_cells, x + CHIP_PAD, y + ICON_TOP, CHIP_W - 2 * CHIP_PAD)
    parts.append(chip(layer, x, y, 7, "OUTPUT", style.GREEN, (runs, RUNS_ICON_H), OUTPUT_LINES))
    parts.append(heads)
    parts += [tile_rect(x, y, w, "none", accent, f' stroke-width="2.5" opacity="0" filter="url(#glow)" '
                                                 f'class="hl h{i}"')
              for i, ((x, y, w), accent) in enumerate(zip(tiles, accents))]
    parts.append("</svg>\n")
    return "".join(parts)


def main() -> None:
    """Compute the pictures, build the figure, check the text budgets and write the file."""
    check_facts()
    example = load_example()
    layer = TextLayer()
    svg = build_svg(example, layer)
    layer.report()
    size = style.write_text(OUT_PATH, svg)
    print(f"wrote {OUT_PATH.relative_to(style.REPO)}: {size} bytes")


if __name__ == "__main__":
    main()
