"""Shared look of the README figures: colours, fonts, SVG building blocks and the GIF writer.

Every figure in ``docs/media`` is drawn by one script of this folder from the stored artefacts of the repository
(model outputs only: probability maps, component images, candidate tables, the reference submission). The scripts
share this module so that the figures use one palette and one type scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
"""Root of the repository."""
MEDIA = REPO / "docs" / "media"
"""Folder the figures are written to."""

# --- palette ----------------------------------------------------------------------------------------------------------
BG = "#070b18"
"""Background of every figure (deep navy)."""
BG_SOFT = "#0c1226"
"""Second background tone, used for gradients."""
PANEL = "#111a33"
"""Fill of cards and chart panels."""
PANEL_EDGE = "#22305c"
"""Outline of cards, axes and grid lines."""
GRID = "#18223f"
"""Faint grid lines."""
TEXT = "#e8ecf8"
"""Main text."""
MUTED = "#93a0c0"
"""Secondary text and axis labels."""
SUN = "#ff8a3d"
"""Solar orange: the disk, titles, highlights that refer to the Sun."""
SUN_LIGHT = "#ffc46b"
"""Light end of the solar gradient."""
CYAN = "#38e1ff"
"""Model evidence: probability maps and scores."""
VIOLET = "#8b7bff"
"""Structure: tree edges and thresholds."""
GREEN = "#3dffa2"
"""Kept candidates and positive outcomes."""
SLATE = "#5b6785"
"""Rejected candidates and reference values."""
ROSE = "#ff5d8f"
"""Sparingly used warning / contrast colour."""

LEVEL_COLOURS = ("#2b4a9e", "#2f6fd0", "#2aa3e6", "#38e1ff", "#7df0e0", "#c8f58a", "#ffc46b", "#ff8a3d")
"""One colour per threshold level of the component tree (level 0, the lowest threshold, first)."""

FONT_SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_MONO = "ui-monospace, 'SFMono-Regular', 'Cascadia Mono', Menlo, Consolas, monospace"
"""Font stacks of the SVG figures: system fonts only, nothing is loaded from the network."""

RADIUS = 22
"""Corner radius of the outer card of every figure (SVG user units)."""
MIN_FONT_PX = 26
"""Smallest font size (figure pixels) for a figure that is 1760 px wide; the README shows it at about half size."""


# --- colour helpers ---------------------------------------------------------------------------------------------------
def hex_to_rgb(colour: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b) with values 0..255."""
    c = colour.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def blend(c0: str, c1: str, t: float) -> str:
    """Linear mix of two '#rrggbb' colours, t = 0 gives c0 and t = 1 gives c1."""
    a, b = hex_to_rgb(c0), hex_to_rgb(c1)
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))


def probability_colormap():
    """Colour map for probability maps: background navy -> blue -> cyan -> near white."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("probability", [BG, "#12305f", "#1d6fb8", CYAN, "#eaffff"], N=256)


def apply_matplotlib_style() -> None:
    """Dark style for the raster figures (call once before creating a figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": PANEL_EDGE,
        "axes.labelcolor": MUTED, "axes.titlecolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
        "ytick.color": MUTED, "grid.color": GRID, "font.family": "DejaVu Sans", "font.size": 13,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2, "lines.solid_capstyle": "round",
    })


def figure_to_rgb(fig) -> np.ndarray:
    """Render a matplotlib figure and return it as an (H, W, 3) uint8 array."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


# --- GIF --------------------------------------------------------------------------------------------------------------
TRANSPARENT_INDEX = 255
"""Palette index that the GIF writer reserves for the transparent corners of the card."""


def outside_card(width: int, height: int, radius: int = RADIUS + 1) -> np.ndarray:
    """Boolean (H, W) mask of the pixels outside a rounded card that fills the frame (the four corner wedges)."""
    y, x = np.mgrid[0:height, 0:width]
    dx = np.maximum(np.maximum(radius - x, x - (width - 1 - radius)), 0)
    dy = np.maximum(np.maximum(radius - y, y - (height - 1 - radius)), 0)
    return dx * dx + dy * dy > radius * radius


def save_gif(frames: Sequence[np.ndarray], path: str | Path, durations_ms: int | Sequence[int] = 80,
             round_corners: bool = True) -> int:
    """Write RGB frames as a looping GIF with one palette for all frames (no flicker, no dithering noise).

    ``durations_ms`` is one value for every frame or one value per frame. With ``round_corners`` the pixels outside
    the rounded card are transparent, so that the card has round corners on a light page as well. Returns the size
    of the file in bytes.
    """
    from PIL import Image
    images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    width, height = images[0].size
    sample = images[:: max(1, len(images) // 12)]
    sheet = Image.new("RGB", (width, height * len(sample)))
    for i, im in enumerate(sample):
        sheet.paste(im, (0, i * height))
    palette = sheet.quantize(colors=TRANSPARENT_INDEX, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantised = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    options: dict = {}
    if round_corners:
        corners = outside_card(width, height)
        for i, im in enumerate(quantised):
            indices = np.array(im)
            indices[corners] = TRANSPARENT_INDEX
            frame = Image.fromarray(indices)
            frame.putpalette(im.getpalette())
            quantised[i] = frame
        options = dict(transparency=TRANSPARENT_INDEX)
    if isinstance(durations_ms, int):
        durations_ms = [durations_ms] * len(quantised)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(path, save_all=True, append_images=quantised[1:], duration=list(durations_ms), loop=0,
                      optimize=False, disposal=1, **options)
    return path.stat().st_size


# --- SVG --------------------------------------------------------------------------------------------------------------
def svg_open(width: int, height: int, title: str, description: str) -> str:
    """Opening tag of a self-contained SVG figure with a title and a description for screen readers."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT_SANS}">\n'
            f'<title id="t">{title}</title>\n<desc id="d">{description}</desc>\n')


def _gradient(kind: str, ident: str, geometry: str, stops: Sequence[tuple[str, str]]) -> str:
    """One gradient definition: ``kind`` is 'linear' or 'radial', ``stops`` are (offset, colour) pairs."""
    inner = "".join(f'<stop offset="{offset}" stop-color="{colour}"/>' for offset, colour in stops)
    return f'<{kind}Gradient id="{ident}" {geometry}>{inner}</{kind}Gradient>\n'


def svg_defs(extra: str = "") -> str:
    """Definitions shared by the SVG figures: soft glow filters, the background and the solar gradients."""
    region = 'x="-50%" y="-50%" width="200%" height="200%"'
    merge = '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
    return ("<defs>\n"
            + _gradient("linear", "bg", 'x1="0" y1="0" x2="1" y2="1"', [("0", BG_SOFT), ("1", BG)])
            + _gradient("radial", "sun", 'cx="0.42" cy="0.38" r="0.75"',
                        [("0", SUN_LIGHT), ("0.55", SUN), ("1", "#b3401a")])
            + _gradient("linear", "accent", 'x1="0" y1="0" x2="1" y2="0"', [("0", SUN), ("1", CYAN)])
            + f'<filter id="glow" {region}><feGaussianBlur stdDeviation="6" result="b"/>{merge}</filter>\n'
            + f'<filter id="glow-soft" {region}><feGaussianBlur stdDeviation="14"/></filter>\n'
            + f"{extra}</defs>\n")


def svg_background(width: int, height: int) -> str:
    """Rounded dark card that carries the figure, so that it reads the same on light and dark pages."""
    return (f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="{RADIUS}" fill="url(#bg)" '
            f'stroke="{PANEL_EDGE}" stroke-width="2"/>\n')


REDUCED_MOTION_CSS = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"
"""Style rule every animated SVG includes, so that viewers who ask for less motion get a still figure."""


def path_from_points(points: Iterable[tuple[float, float]], closed: bool = True, digits: int = 1) -> str:
    """SVG path data ('M x y L x y ... Z') through the given points."""
    pts = [f"{x:.{digits}f} {y:.{digits}f}" for x, y in points]
    return "M " + " L ".join(pts) + (" Z" if closed else "")


def write_text(path: str | Path, text: str) -> int:
    """Write a text file with Unix line ends; returns its size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return len(data)
