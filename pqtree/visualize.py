"""Small figures for the notebook: dense map, candidates at several thresholds, selected instances, annotator outlines,
a gallery of test images and a full-disk view.

All functions take 2048 x 2048 arrays and, except :func:`full_disk_figure`, draw a square crop
(``box = (y0, y1, x0, x1)``). The H-alpha image is optional in the single-crop figures; without it the probability map is
used as background. The gallery and the full-disk view need the image.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

CREDIT = "H-alpha images: GONG, NSO. MAGFiLO (Ahmadzadeh et al., Sci. Data 11, 1031, 2024), CC BY-NC 4.0, non-commercial use."


COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]


def box_around(mask: np.ndarray, size: int = 512, shape: tuple[int, int] = (2048, 2048)) -> tuple[int, int, int, int]:
    """Square crop of side ``size`` centred on the centroid of ``mask``, moved to lie inside the image."""
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - size // 2, 0, shape[0] - size))
    x0 = int(np.clip(cx - size // 2, 0, shape[1] - size))
    return y0, y0 + size, x0, x0 + size


def _credit(fig, img) -> None:
    """Credit line under pictures that contain crops of competition images."""
    if img is not None:
        fig.text(0.5, 0.005, CREDIT, ha="center", va="bottom", fontsize=5.5)
        fig.subplots_adjust(bottom=0.08)


def _crop(a: np.ndarray, box) -> np.ndarray:
    y0, y1, x0, x1 = box
    return a[y0:y1, x0:x1].copy()


def _outline(ax, mask_crop: np.ndarray, color: str, lw: float = 1.2) -> None:
    if mask_crop.any() and not mask_crop.all():
        ax.contour(mask_crop.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _background(ax, img: Optional[np.ndarray], prob: np.ndarray, box) -> None:
    ax.imshow(_crop(img, box) if img is not None else _crop(prob, box), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def dense_map_figure(prob: np.ndarray, box, img: Optional[np.ndarray] = None):
    """Image crop next to the probability map crop."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    _background(axes[0], img, prob, box)
    axes[0].set_title("H-alpha image" if img is not None else "probability map")
    axes[1].imshow(_crop(prob, box), cmap="magma", vmin=0, vmax=255)
    axes[1].set_title("probability of filament")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.tight_layout()
    _credit(fig, img)
    return fig


def candidate_levels_figure(prob: np.ndarray, labels: Sequence[np.ndarray], levels: Sequence[int], thresholds: Sequence[float],
                            box, img: Optional[np.ndarray] = None):
    """Connected components at the chosen tree levels, each component outlined in its own colour."""
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4))
    for ax, lv, t in zip(np.atleast_1d(axes), levels, thresholds):
        _background(ax, img, prob, box)
        lab = _crop(labels[lv], box)
        ids = [i for i in np.unique(lab) if i != 0]
        for j, i in enumerate(ids):
            _outline(ax, lab == i, COLORS[j % len(COLORS)])
        ax.set_title(f"threshold {t:.1f}: {len(ids)} components in view")
    fig.tight_layout()
    _credit(fig, img)
    return fig


def instances_figure(prob: np.ndarray, masks: Sequence[np.ndarray], box, img: Optional[np.ndarray] = None, title: str = ""):
    """Selected instances (full-size boolean masks) outlined on the crop."""
    fig, ax = plt.subplots(figsize=(5, 5))
    _background(ax, img, prob, box)
    for j, m in enumerate(masks):
        _outline(ax, _crop(m, box), COLORS[j % len(COLORS)], 1.5)
    ax.set_title(title)
    fig.tight_layout()
    _credit(fig, img)
    return fig


def annotator_figure(prob: np.ndarray, masks: Sequence[np.ndarray], annotator_masks: Sequence[Sequence[np.ndarray]], box,
                     img: Optional[np.ndarray] = None, title: str = ""):
    """Selected instances (solid red) and the outlines of each annotator (dashed, one colour per annotator)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    _background(ax, img, prob, box)
    for m in masks:
        _outline(ax, _crop(m, box), "#e41a1c", 1.8)
    palette = ["#00b4d8", "#ffd60a", "#80ed99"]
    for a, group in enumerate(annotator_masks):
        for m in group:
            c = _crop(m, box)
            if c.any() and not c.all():
                ax.contour(c.astype(float), levels=[0.5], colors=[palette[a % 3]], linewidths=1.2, linestyles="dashed")
    ax.set_title(title + "  red: selected, dashed: annotators")
    fig.tight_layout()
    _credit(fig, img)
    return fig


def gallery_figure(panels: Sequence[tuple[str, np.ndarray, Sequence[np.ndarray], tuple[int, int, int, int]]], ncols: int = 4):
    """Grid of square crops of H-alpha images with every selected instance outlined in its own colour.

    Each panel is ``(image id, grey image, full-size boolean masks of all selected instances of that image, box)``; the
    title gives the image id and the number of selected instances in the whole image (also those outside the crop).
    """
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.75 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (stem, img, masks, box) in zip(axes, panels):
        _background(ax, img, img, box)
        for j, m in enumerate(masks):
            _outline(ax, _crop(m, box), COLORS[j % len(COLORS)], 1.2)
        ax.set_title(f"{stem}: {len(masks)} instance{'' if len(masks) == 1 else 's'}", fontsize=8.5)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout(h_pad=2.0)
    _credit(fig, panels[0][1] if panels else None)
    return fig


def full_disk_figure(img: np.ndarray, masks: Sequence[np.ndarray], title: str = ""):
    """One whole image with every selected instance (full-size boolean masks) outlined in its own colour."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    for j, m in enumerate(masks):
        _outline(ax, m, COLORS[j % len(COLORS)], 1.0)
    ax.set_title(title)
    fig.tight_layout()
    _credit(fig, img)
    return fig
