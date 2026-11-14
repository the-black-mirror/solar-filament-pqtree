"""Animated figure ``docs/media/threshold_sweep.gif``: the one free parameter of the method, the selection threshold.

Left: the dense probability map of one test image inside the solar disk, with the instances that the tree selection
keeps at the current tau outlined; instances that drop out as tau rises turn grey and fade. Right: the out-of-fold PQ
of the 707 training images (mean of the three cross-fitting shuffles, and shuffle 0) and the number of kept instances
per test image, both as functions of tau, with the range of tau that scored 0.41 on the public leaderboard.

Sources (all in the repository):

* test side: the candidate table of the test images (``pqtree.pipeline.build_test_table``) scored with the stored
  hurdle LightGBM model, kept sets from ``pqtree.selection.select``; masks from ``artifacts/test_components.npz``; the
  probability map from ``artifacts/test_probs_unet.npz``;
* training side: ``artifacts/candidates_oof.tbl.npz`` joined with ``artifacts/oof_scores.csv.gz``, scored with
  ``pqtree.evaluation.target_based_pq`` and the annotated-mask count of ``artifacts/ngt_train.json``.

The image is chosen by a fixed rule (:func:`choose_image`). Before drawing, the curves are checked against the values
printed in README.md and the kept count at ``config.TAU`` against the rows of ``reference/submission.csv``.

    python tools/readme_media/threshold_sweep.py [--series PATH] [--stills DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import IdentityTransform, blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from pqtree import evaluation, lgbm_scorer, pipeline, selection  # noqa: E402
from pqtree.artifacts import fetch_artifact, load_table  # noqa: E402
from pqtree.config import DISK_RADIUS, H, KEY, TAU, W  # noqa: E402

OUT = style.MEDIA / "threshold_sweep.gif"

# --- data -------------------------------------------------------------------------------------------------------------
TAU_GRID: tuple[float, ...] = tuple(round(t / 100, 2) for t in range(16, 51))
"""Thresholds of the sweep (0.16 to 0.50 in steps of 0.01)."""
SHUFFLES = (0, 1, 2)
"""Cross-fitting shuffles of the out-of-fold scores."""
N_PROCESSES = 2
"""Processes for the selections of the sweep."""
PUBLIC_BAND = (0.28, 0.40)
"""Range of tau that scored 0.41 on the public leaderboard (README.md, 'The selection threshold')."""
EXPECTED_SHUFFLE0 = {0.20: 0.4645, 0.24: 0.4658, 0.26: 0.4668, 0.30: 0.4646, 0.34: 0.4608, 0.38: 0.4569}
"""Out-of-fold PQ of shuffle 0 printed in README.md (section Results)."""
EXPECTED_MEAN = {0.26: 0.4663, 0.34: 0.4625}
"""Three-shuffle mean PQ printed in README.md."""
CHECK_TOLERANCE = 0.0002
MIN_KEPT_FOR_CHOICE = 10
"""The figure image keeps at least this many instances at ``config.TAU``."""
DISK_CENTRE = (W / 2, H / 2)
"""Centre (x, y) of the solar disk in image pixels."""

# --- frame geometry (pixels, origin top left) -------------------------------------------------------------------
FRAME_W, FRAME_H = 1520, 800
DPI = 100
MARGIN = 48
EYEBROW_Y, TITLE_Y = 62, 108
CAPTION_Y = 172
ROW1_Y, ROW2_Y = 700, 738
LEFT_X0, LEFT_X1 = MARGIN, 588
DISK_PX = 480
"""Side of the square that shows the disk crop."""
DISK_TOP = 196
CROP_HALF = 1000
"""Half side (image pixels) of the square crop around the disk centre."""
AX_LEFT, AX_RIGHT, AX_TOP, AX_BOTTOM = 760, 1416, 214, 636
FONT_PX = 26
TITLE_PX = 36
TRACKING_PX = 3
MONO_ADVANCE = 1233 / 2048
"""Advance width of DejaVu Sans Mono in em."""
LEGEND_GAP = 70
LEGEND_SWATCH = (40, 12)
"""Start and end of a legend line, measured leftwards from its label."""

# --- chart scales -----------------------------------------------------------------------------------------------------
PQ_LIMITS = (0.36, 0.483)
PQ_TICKS = (0.42, 0.44, 0.46)
KEPT_LIMITS = (0.0, 18.0)
KEPT_TICKS = (2, 4, 6)
"""The PQ curve lives in the upper part of the chart and the kept count in the lower part, so they never cross."""
TAU_TICKS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
TOP_LABEL_ROW = 0.955
BAND_LABEL_ROWS = (0.13, 0.05)
"""Heights (fractions of the chart) of the labels at the top and of the two lines of the band label."""
TAU_LINE_FROM = 0.18
"""The line at ``config.TAU`` starts above the band label."""
LABEL_GAP = 0.006
"""Horizontal gap (in tau) between a line and its label."""
SHUFFLE0_LABEL_TAU, SHUFFLE0_LABEL_DROP = 0.44, 0.009
"""The shuffle-0 line is labelled below and left of its value at this tau (where it departs most from the mean)."""

# --- motion -----------------------------------------------------------------------------------------------------------
FRAME_MS = 80
SEGMENTS = ((0.16, 0.16, 6), (0.16, 0.50, 56), (0.50, 0.50, 4), (0.50, 0.34, 30), (0.34, 0.34, 32), (0.34, 0.16, 16))
"""(tau from, tau to, frames) of the loop; moving segments use smoothstep easing."""
FADE_FRAMES = 6
"""Frames an instance needs to fade out (or in) when its kept status changes."""
BAND_EASE = 0.03
"""Distance in tau over which the band label gains emphasis as the marker approaches the band."""

# --- drawing ----------------------------------------------------------------------------------------------------------
DISK_FILL = style.blend(style.BG, style.PANEL, 0.55)
PROB_ALPHA_GAIN = 2.5
BLOOM_SIGMA, BLOOM_GAIN = 6.0, 0.55
LIMB_WIDTH, LIMB_GLOW, LIMB_GLOW_SIGMA = 1.4, 0.35, 4.0
SUPERSAMPLE = 3
LINE_PX = 2.2
GLOW_SIGMA, GLOW_GAIN = 3.0, 0.45
FILL_ALPHA = 0.16
STILL_FRAMES = (0, 14, 28, 29, 40, 52, 61, 66, 78, 95, 105, 128, 136)
"""Frames written by ``--stills``."""


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def _oof_pq(table: pd.DataFrame, n_gt: int, job: tuple[int, float]) -> float:
    """Pooled PQ from the stored targets for one (shuffle, tau)."""
    seed, tau = job
    chosen = selection.select(table, table[f"score_seed{seed}"].values, tau)
    return evaluation.target_based_pq(table, chosen, n_gt)


def out_of_fold_curves(taus: Sequence[float]) -> list[list[float]]:
    """PQ(tau) of the out-of-fold training predictions, one list per shuffle.

    Same computation as ``three_shuffle_means`` in ``tools/report_statistics.py``.
    """
    cands = load_table(fetch_artifact("candidates_oof.tbl.npz"))
    scores = pd.read_csv(fetch_artifact("oof_scores.csv.gz"), float_precision="round_trip")
    n_gt = sum(json.loads(fetch_artifact("ngt_train.json").read_text()).values())
    table = cands.merge(scores, on=KEY, how="left")
    table = table[KEY + ["parent", "u", "nann"] + [f"score_seed{s}" for s in SHUFFLES]]
    jobs = [(s, t) for s in SHUFFLES for t in taus]
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        values = list(pool.map(partial(_oof_pq, table, n_gt), jobs))
    return [values[i * len(taus):(i + 1) * len(taus)] for i in range(len(SHUFFLES))]


def test_kept_sets(taus: Sequence[float]) -> tuple[pd.DataFrame, list[list[tuple[str, int, int]]]]:
    """Test candidate table (key, parent, centroid) and the kept candidates at every tau."""
    te = pipeline.build_test_table()
    model = lgbm_scorer.HurdleModel.load(fetch_artifact(lgbm_scorer.HIT_FILE), fetch_artifact(lgbm_scorer.IOU_FILE))
    score = model.predict(te)
    slim = te[KEY + ["parent", "cx", "cy"]].copy()
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        kept = list(pool.map(partial(selection.select, slim, score), taus))
    return slim, kept


def choose_image(table: pd.DataFrame, kept: Sequence[list[tuple[str, int, int]]], taus: Sequence[float]) -> str:
    """Test image of the figure.

    Rule: among the images that keep at least ``MIN_KEPT_FOR_CHOICE`` instances at ``config.TAU`` and whose kept
    instances lie in all four quadrants of the disk, take the one with the most candidates whose kept status changes
    over the sweep (kept at some tau, not kept at another); ties go to more kept instances at ``config.TAU``, then to
    the smaller stem.
    """
    centre = table.set_index(KEY)[["cx", "cy"]]
    per_stem: dict[str, list[set]] = {}
    for i, chosen in enumerate(kept):
        for stem, level, cid in chosen:
            per_stem.setdefault(stem, [set() for _ in taus])[i].add((level, cid))
    i_tau = taus.index(TAU)
    ranked = []
    for stem, sets in per_stem.items():
        at_tau = sets[i_tau]
        if len(at_tau) < MIN_KEPT_FOR_CHOICE:
            continue
        xy = centre.loc[[(stem, lv, c) for lv, c in at_tau]].to_numpy()
        quadrants = {(x > DISK_CENTRE[0], y > DISK_CENTRE[1]) for x, y in xy}
        if len(quadrants) < 4:
            continue
        changing = len(set().union(*sets)) - len(set.intersection(*sets))
        ranked.append((-changing, -len(at_tau), stem))
    return min(ranked)[2]


def compute_series() -> dict:
    """Everything the figure shows, as plain JSON-serialisable values."""
    taus = list(TAU_GRID)
    pq = out_of_fold_curves(taus)
    table, kept = test_kept_sets(taus)
    stem = choose_image(table, kept, taus)
    n_images = int(table.stem.nunique())
    with_kept = {s for s, _, _ in kept[taus.index(TAU)]}
    return dict(taus=taus, pq_shuffles=pq, pq_mean=np.mean(pq, axis=0).tolist(),
                kept_total=[len(k) for k in kept], n_test_images=n_images, fallback_images=n_images - len(with_kept),
                image=stem, image_kept=[sorted([lv, c] for s, lv, c in k if s == stem) for k in kept])


def load_or_compute_series(path: Path | None) -> dict:
    """Series from ``path`` if it exists, else computed (and written to ``path`` if given)."""
    if path is not None and path.exists():
        return json.loads(path.read_text())
    series = compute_series()
    if path is not None:
        style.write_text(path, json.dumps(series))
    return series


def check_series(series: dict) -> list[str]:
    """Compare the curves with README.md and the kept count with the reference submission; raise on a mismatch."""
    taus = series["taus"]
    report = []
    for name, curve, expected in (("shuffle 0", series["pq_shuffles"][0], EXPECTED_SHUFFLE0),
                                  ("three-shuffle mean", series["pq_mean"], EXPECTED_MEAN)):
        for tau, want in expected.items():
            got = curve[taus.index(tau)]
            if abs(got - want) > CHECK_TOLERANCE:
                raise ValueError(f"{name} PQ at tau {tau:.2f} is {got:.5f}, README.md says {want:.4f}")
            report.append(f"{name:>18} tau {tau:.2f}: {got:.5f} (README {want:.4f})")
    rows = len(pd.read_csv(style.REPO / "reference" / "submission.csv"))
    kept = series["kept_total"][taus.index(TAU)]
    if kept + series["fallback_images"] != rows:
        raise ValueError(f"{kept} kept + {series['fallback_images']} fallback rows != {rows} submission rows")
    report.append(f"kept at tau {TAU:.2f}: {kept} + {series['fallback_images']} images without a kept candidate "
                  f"(one fallback row each) = {rows} rows of reference/submission.csv")
    return report


# ---------------------------------------------------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------------------------------------------------
def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    """Cubic ease-in-out on [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def tau_schedule() -> np.ndarray:
    """Tau of every frame of the loop."""
    out = []
    for a, b, n in SEGMENTS:
        out += [a + (b - a) * smoothstep((i + 1) / n) for i in range(n)]
    return np.array(out)


def grid_index(tau: float) -> int:
    """Index of the grid threshold nearest to ``tau``."""
    return int(np.clip(round((tau - TAU_GRID[0]) * 100), 0, len(TAU_GRID) - 1))


def instance_states(schedule: np.ndarray, kept_by_tau: Sequence[set], keys: Sequence[tuple]) -> np.ndarray:
    """Visibility (0 = gone, 1 = kept) of every instance in every frame; each change takes ``FADE_FRAMES`` frames.

    The loop is run twice and the second pass is returned, so that the last frame leads into the first without a jump.
    """
    state = np.array([k in kept_by_tau[grid_index(schedule[0])] for k in keys], float)
    out = np.zeros((len(schedule), len(keys)))
    for _ in range(2):
        for f, tau in enumerate(schedule):
            target = np.array([k in kept_by_tau[grid_index(tau)] for k in keys], float)
            state += np.clip(target - state, -1 / FADE_FRAMES, 1 / FADE_FRAMES)
            out[f] = state
    return out


def band_emphasis(tau: float) -> float:
    """1 inside the public-score band, easing to 0 within ``BAND_EASE`` outside it."""
    d = max(PUBLIC_BAND[0] - tau, tau - PUBLIC_BAND[1], 0.0)
    return float(smoothstep(1 - d / BAND_EASE))


# ---------------------------------------------------------------------------------------------------------------------
# left panel: the disk
# ---------------------------------------------------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """'#rrggbb' as float RGB in [0, 1]."""
    return np.array(style.hex_to_rgb(colour), float) / 255


def screen(base: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Screen blend (adds light without clipping)."""
    return 1 - (1 - base) * (1 - np.clip(light, 0, 1))


def crop_to_display(a: np.ndarray, size: int) -> np.ndarray:
    """Square crop around the disk centre, resized (area average) to ``size`` x ``size``."""
    cy, cx = int(DISK_CENTRE[1]), int(DISK_CENTRE[0])
    crop = a[cy - CROP_HALF:cy + CROP_HALF, cx - CROP_HALF:cx + CROP_HALF].astype(np.float32)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def disk_base(prob: np.ndarray) -> np.ndarray:
    """Probability map inside the disk with a soft bloom, a thin solar limb and a dark surround (float RGB)."""
    p = crop_to_display(prob, DISK_PX) / 255
    colours = style.probability_colormap()(p)[..., :3]
    alpha = np.clip(p * PROB_ALPHA_GAIN, 0, 1)[..., None]
    img = rgb(DISK_FILL) * (1 - alpha) + colours * alpha
    img = screen(img, cv2.GaussianBlur(colours * alpha, (0, 0), BLOOM_SIGMA) * BLOOM_GAIN)
    yy, xx = np.mgrid[0:DISK_PX, 0:DISK_PX]
    r = np.hypot(yy - (DISK_PX - 1) / 2, xx - (DISK_PX - 1) / 2)
    radius = DISK_RADIUS * DISK_PX / (2 * CROP_HALF)
    inside = np.clip(radius - r + 0.5, 0, 1)[..., None]
    img = rgb(style.BG) * (1 - inside) + img * inside
    ring = np.clip(1 - np.abs(r - radius) / LIMB_WIDTH, 0, 1)[..., None]
    img = screen(img, rgb(style.SUN) * cv2.GaussianBlur(ring, (0, 0), LIMB_GLOW_SIGMA)[..., None] * LIMB_GLOW)
    return img * (1 - ring) + rgb(style.SUN) * ring


def outline_layers(labels: Callable[[int], np.ndarray], key: tuple[int, int]) -> np.ndarray:
    """Line, glow and fill layers (3, DISK_PX, DISK_PX) of one candidate mask at display size."""
    level, cid = key
    mask = (labels(level) == cid).astype(np.float32)
    big = crop_to_display(mask, DISK_PX * SUPERSAMPLE)
    contours, _ = cv2.findContours((big > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    canvas = np.zeros(big.shape, np.uint8)
    cv2.drawContours(canvas, contours, -1, 255, thickness=round(LINE_PX * SUPERSAMPLE), lineType=cv2.LINE_AA)
    line = cv2.resize(canvas.astype(np.float32) / 255, (DISK_PX, DISK_PX), interpolation=cv2.INTER_AREA)
    glow = cv2.GaussianBlur(line, (0, 0), GLOW_SIGMA)
    fill = cv2.resize(big, (DISK_PX, DISK_PX), interpolation=cv2.INTER_AREA)
    return np.stack([line, glow, fill])


def instance_colour(state: float) -> tuple[np.ndarray, float]:
    """Colour and opacity of an instance: green when kept, turning grey first and then fading as it leaves."""
    mix = float(smoothstep((state - 0.7) / 0.3))
    return rgb(style.blend(style.SLATE, style.GREEN, mix)), float(smoothstep(state / 0.7))


def compose_disk(base: np.ndarray, layers: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Disk image of one frame (uint8 RGB): leaving instances below, kept ones on top."""
    img = base.copy()
    for i in np.argsort(states, kind="stable"):
        if states[i] <= 0:
            continue
        colour, opacity = instance_colour(states[i])
        line, glow, fill = (layers[i, j][..., None] for j in range(3))
        img = img * (1 - opacity * FILL_ALPHA * fill) + colour * opacity * FILL_ALPHA * fill
        img = screen(img, colour * opacity * GLOW_GAIN * glow)
        img = img * (1 - opacity * line) + colour * opacity * line
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------------------------------
def pt(px: float) -> float:
    """Font size in points for a size in figure pixels."""
    return px * 72 / DPI


def mono_width(n_chars: int, size_px: float = FONT_PX) -> float:
    """Width in pixels of ``n_chars`` characters of the mono font."""
    return n_chars * MONO_ADVANCE * size_px


class Frame:
    """The matplotlib figure of the animation and the artists that change from frame to frame."""

    def __init__(self, series: dict, base_disk: np.ndarray) -> None:
        style.apply_matplotlib_style()
        import matplotlib.pyplot as plt
        self.fig = plt.figure(figsize=(FRAME_W / DPI, FRAME_H / DPI), dpi=DPI)
        self.pixels = IdentityTransform()
        self.series = series
        card = FancyBboxPatch((1, 1), FRAME_W - 2, FRAME_H - 2, boxstyle=f"round,pad=0,rounding_size={style.RADIUS}",
                              transform=self.pixels, facecolor=style.BG, edgecolor=style.PANEL_EDGE, lw=2, zorder=-1)
        self.fig.patches.append(card)
        self._header()
        self.disk = self._disk_axes(base_disk)
        self.counter_image, self.counter_all = self._counters()
        self._chart()
        self.readout = self._readout()

    # -- text helpers ------------------------------------------------------------------------------------------------
    def text(self, x: float, y: float, s: str, colour: str = style.TEXT, size: float = FONT_PX, mono: bool = True,
             ha: str = "left", **kw):
        """Text at pixel position (x, y from the top), vertically centred."""
        return self.fig.text(x, FRAME_H - y, s, color=colour, fontsize=pt(size), ha=ha, va="center_baseline",
                             family="DejaVu Sans Mono" if mono else "DejaVu Sans", transform=self.pixels, **kw)

    def tracked(self, x: float, y: float, s: str, colour: str, ha: str = "left") -> None:
        """Upper-case letter-spaced mono label (one artist per character)."""
        step = mono_width(1) + TRACKING_PX
        if ha == "right":
            x -= len(s) * step - TRACKING_PX
        for i, ch in enumerate(s):
            if ch != " ":
                self.text(x + i * step, y, ch, colour)

    # -- static parts ------------------------------------------------------------------------------------------------
    def _header(self) -> None:
        """Eyebrow, title and the legend of the outlines."""
        self.tracked(MARGIN, EYEBROW_Y, "ONE PARAMETER: THE SELECTION THRESHOLD TAU", style.SUN)
        self.text(MARGIN, TITLE_Y, "A candidate is kept only if its predicted IoU exceeds tau",
                  size=TITLE_PX, mono=False)
        right = FRAME_W - MARGIN
        dropped, kept = "dropped as tau rises", "kept"
        x_dropped = right - mono_width(len(dropped))
        x_kept = x_dropped - LEGEND_GAP - mono_width(len(kept))
        for x, label, colour in ((x_kept, kept, style.GREEN), (x_dropped, dropped, style.SLATE)):
            swatch = (x - LEGEND_SWATCH[0], x - LEGEND_SWATCH[1])
            self.fig.add_artist(Line2D(swatch, [FRAME_H - EYEBROW_Y] * 2, color=colour, lw=3, transform=self.pixels))
            self.text(x, EYEBROW_Y, label, style.MUTED)

    def _disk_axes(self, base: np.ndarray):
        """Axes that show the disk image 1:1 and the caption above it."""
        x0 = (LEFT_X0 + LEFT_X1 - DISK_PX) / 2
        ax = self.fig.add_axes((x0 / FRAME_W, 1 - (DISK_TOP + DISK_PX) / FRAME_H, DISK_PX / FRAME_W, DISK_PX / FRAME_H))
        ax.set_axis_off()
        image = ax.imshow(np.round(base * 255).astype(np.uint8), interpolation="nearest")
        self.tracked(LEFT_X0, CAPTION_Y, "PROBABILITY MAP", style.CYAN)
        stem = self.series["image"]
        self.text(LEFT_X1, CAPTION_Y, f"test {stem[:4]}-{stem[4:6]}-{stem[6:8]}", style.MUTED, ha="right")
        return image

    def _counters(self):
        """Two counter rows under the disk; the numbers are right-aligned in one column."""
        n = self.series["n_test_images"]
        self.text(LEFT_X0, ROW1_Y, "kept in this image:", style.MUTED)
        self.text(LEFT_X0, ROW2_Y, f"kept in all {n} test images:", style.MUTED)
        return (self.text(LEFT_X1, ROW1_Y, "", style.GREEN, ha="right"),
                self.text(LEFT_X1, ROW2_Y, "", style.TEXT, ha="right"))

    def _chart(self) -> None:
        """Out-of-fold PQ (left axis, upper half) and kept instances per test image (right axis, lower half)."""
        s = self.series
        taus = np.array(s["taus"])
        per_image = np.array(s["kept_total"]) / s["n_test_images"]
        rect = (AX_LEFT / FRAME_W, 1 - AX_BOTTOM / FRAME_H,
                (AX_RIGHT - AX_LEFT) / FRAME_W, (AX_BOTTOM - AX_TOP) / FRAME_H)
        ax = self.fig.add_axes(rect)
        ax2 = ax.twinx()
        for a in (ax, ax2):
            a.set_facecolor("none")
            a.set_xlim(TAU_GRID[0], TAU_GRID[-1])
            for side in ("top", "right", "left"):
                a.spines[side].set_visible(False)
            a.spines["bottom"].set_color(style.PANEL_EDGE)
            a.tick_params(length=0, pad=12, labelsize=pt(FONT_PX), labelfontfamily="DejaVu Sans Mono")
        ax.set_ylim(*PQ_LIMITS)
        ax2.set_ylim(*KEPT_LIMITS)
        ax.set_xticks(TAU_TICKS, [f"{t:.2f}" for t in TAU_TICKS], color=style.MUTED)
        ax.set_yticks(PQ_TICKS, [f"{t:.2f}" for t in PQ_TICKS], color=style.blend(style.CYAN, style.MUTED, 0.35))
        ax2.set_yticks(KEPT_TICKS, [str(t) for t in KEPT_TICKS], color=style.blend(style.VIOLET, style.MUTED, 0.25))
        for t in PQ_TICKS:
            ax.axhline(t, color=style.GRID, lw=1.2, zorder=0)
        self.band = ax.axvspan(*PUBLIC_BAND, color=style.SUN, lw=0, alpha=0.08, zorder=0)
        x_data_y_axes = blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([TAU, TAU], [TAU_LINE_FROM, 1], color=style.SUN, lw=1.6, alpha=0.9, zorder=1, transform=x_data_y_axes)
        label = dict(fontsize=pt(FONT_PX), va="center_baseline", transform=x_data_y_axes)
        self.band_label = [ax.text(np.mean(PUBLIC_BAND), y, line, color=style.SUN, ha="center", **label)
                           for y, line in zip(BAND_LABEL_ROWS, ("public score 0.41", "across this range"))]
        ax.text(TAU + LABEL_GAP, TOP_LABEL_ROW, f"tau = {TAU:.2f} used", color=style.SUN, ha="left", **label)
        ax.text(TAU_GRID[0] + LABEL_GAP, TOP_LABEL_ROW, "mean of 3 shuffles", color=style.CYAN, ha="left", **label)
        mean, s0 = np.array(s["pq_mean"]), np.array(s["pq_shuffles"][0])
        ax.text(SHUFFLE0_LABEL_TAU, float(np.interp(SHUFFLE0_LABEL_TAU, taus, s0)) - SHUFFLE0_LABEL_DROP, "shuffle 0",
                color=style.CYAN, alpha=0.6, ha="right", va="center_baseline", fontsize=pt(FONT_PX))
        ax.plot(taus, s0, color=style.CYAN, lw=1.4, alpha=0.45, zorder=2)
        ax.plot(taus, mean, color=style.CYAN, lw=9, alpha=0.12, zorder=2)
        ax.plot(taus, mean, color=style.CYAN, lw=3, zorder=3)
        ax2.plot(taus, per_image, color=style.VIOLET, lw=9, alpha=0.12, zorder=2)
        ax2.plot(taus, per_image, color=style.VIOLET, lw=3, zorder=3)
        self.tracked(AX_LEFT, CAPTION_Y, "OUT-OF-FOLD PQ", style.CYAN)
        self.tracked(AX_RIGHT, CAPTION_Y, "KEPT PER TEST IMAGE", style.VIOLET, ha="right")
        self.text((AX_LEFT + AX_RIGHT) / 2, ROW1_Y, "selection threshold tau", style.MUTED, mono=False, ha="center")
        self.cursor = ax.axvline(TAU_GRID[0], color=style.TEXT, lw=1.2, alpha=0.35, zorder=4)
        self.dots = []
        for a, colour in ((ax, style.CYAN), (ax2, style.VIOLET)):
            halo, = a.plot([], [], "o", ms=24, color=colour, alpha=0.2, mew=0, zorder=5)
            dot, = a.plot([], [], "o", ms=11, color=colour, mec=style.BG, mew=2, zorder=6)
            self.dots.append((halo, dot))
        self.curves = (mean, per_image)

    def _readout(self) -> list:
        """Read-out row 'tau 0.27  PQ 0.4660  5.9 per image' with fixed character positions."""
        parts = (("tau ", style.MUTED), ("0.00", style.TEXT), ("   PQ ", style.MUTED), ("0.0000", style.CYAN),
                 ("   ", style.MUTED), ("0.0", style.VIOLET), (" per image", style.MUTED))
        artists, column = [], 0
        for s, colour in parts:
            artists.append(self.text(AX_LEFT + mono_width(column), ROW2_Y, s, colour))
            column += len(s)
        return artists

    # -- per frame ---------------------------------------------------------------------------------------------------
    def render(self, tau: float, disk: np.ndarray, kept_here: int) -> np.ndarray:
        """Set the moving parts for ``tau`` and return the frame as an RGB array."""
        s = self.series
        k = grid_index(tau)
        self.disk.set_data(disk)
        self.counter_image.set_text(f"{kept_here}")
        self.counter_all.set_text(f"{s['kept_total'][k]}")
        self.cursor.set_xdata([tau, tau])
        for (halo, dot), curve in zip(self.dots, self.curves):
            y = float(np.interp(tau, s["taus"], curve))
            halo.set_data([tau], [y])
            dot.set_data([tau], [y])
        self.readout[1].set_text(f"{s['taus'][k]:.2f}")
        self.readout[3].set_text(f"{s['pq_mean'][k]:.4f}")
        self.readout[5].set_text(f"{s['kept_total'][k] / s['n_test_images']:.1f}")
        e = band_emphasis(tau)
        self.band.set_alpha(0.06 + 0.06 * e)
        for label in self.band_label:
            label.set_alpha(0.6 + 0.4 * e)
        return style.figure_to_rgb(self.fig)


# ---------------------------------------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------------------------------------
def build_frames(series: dict) -> list[np.ndarray]:
    """All frames of the loop."""
    stem = series["image"]
    prob = np.load(fetch_artifact("test_probs_unet.npz"))[stem]
    labels = selection.stored_label_images(fetch_artifact("test_components.npz"))(stem)
    kept_by_tau = [{tuple(k) for k in ks} for ks in series["image_kept"]]
    keys = sorted(set().union(*kept_by_tau))
    layers = np.stack([outline_layers(labels, key) for key in keys])
    base = disk_base(prob)
    schedule = tau_schedule()
    states = instance_states(schedule, kept_by_tau, keys)
    frame = Frame(series, base)
    return [frame.render(tau, compose_disk(base, layers, states[f]), len(kept_by_tau[grid_index(tau)]))
            for f, tau in enumerate(schedule)]


def merge_repeats(frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Merge runs of identical frames into one frame with the summed duration."""
    out, durations = [], []
    for f in frames:
        if out and np.array_equal(out[-1], f):
            durations[-1] += FRAME_MS
        else:
            out.append(f)
            durations.append(FRAME_MS)
    return out, durations


def write_stills(frames: Sequence[np.ndarray], folder: Path) -> None:
    """Key frames as PNG, and one of them at half size (the size on the README page)."""
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    for i in STILL_FRAMES:
        Image.fromarray(frames[i]).save(folder / f"frame_{i:03d}.png")
    half = Image.fromarray(frames[STILL_FRAMES[-1]])
    half.resize((FRAME_W // 2, FRAME_H // 2), Image.Resampling.LANCZOS).save(folder / "frame_half.png")


def main() -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--series", type=Path, default=None, help="JSON file to reuse (or store) the computed curves")
    ap.add_argument("--stills", type=Path, default=None, help="folder for key frames as PNG")
    args = ap.parse_args()
    series = load_or_compute_series(args.series)
    for line in check_series(series):
        print(line)
    print(f"image {series['image']}: kept {len(series['image_kept'][TAU_GRID.index(TAU)])} at tau {TAU:.2f}")
    frames = build_frames(series)
    if args.stills is not None:
        write_stills(frames, args.stills)
    unique, durations = merge_repeats(frames)
    size = style.save_gif(unique, OUT, durations)
    print(f"wrote {OUT.relative_to(style.REPO)}: {len(frames)} frames ({len(unique)} distinct), {size} bytes")


if __name__ == "__main__":
    main()
