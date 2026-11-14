"""Shared look of the README figures: colours, fonts, SVG building blocks, text budgets, the solar disk, the GIF writer.

Every figure in ``docs/media`` is drawn by one script of this folder from the stored artefacts of the repository
(probability maps, component images, candidate tables, the reference submission) and the evaluation numbers
of ``numbers.json``. The scripts share this module so that the figures use one palette and one type scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.sax.saxutils import escape

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


GradientStop = tuple[float, str, float]
"""(offset, colour, opacity) of one gradient stop."""


def gradient_stops(stops: Iterable[GradientStop]) -> str:
    """``<stop>`` elements of a gradient; the opacity attribute is written only below 1."""
    return "".join(f'<stop offset="{o:.3g}" stop-color="{c}"' + (f' stop-opacity="{a:g}"/>' if a < 1 else "/>")
                   for o, c, a in stops)


def _gradient(kind: str, ident: str, geometry: str, stops: Iterable[GradientStop]) -> str:
    """One gradient definition: ``kind`` is 'linear' or 'radial'."""
    return f'<{kind}Gradient id="{ident}" {geometry}>{gradient_stops(stops)}</{kind}Gradient>\n'


def svg_defs(extra: str = "") -> str:
    """Definitions shared by the SVG figures: soft glow filters, the background and the solar gradients."""
    region = 'x="-50%" y="-50%" width="200%" height="200%"'
    merge = '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
    return ("<defs>\n"
            + _gradient("linear", "bg", 'x1="0" y1="0" x2="1" y2="1"', [(0, BG_SOFT, 1), (1, BG, 1)])
            + _gradient("radial", "sun", 'cx="0.42" cy="0.38" r="0.75"',
                        [(0, SUN_LIGHT, 1), (0.55, SUN, 1), (1, "#b3401a", 1)])
            + _gradient("linear", "accent", 'x1="0" y1="0" x2="1" y2="0"', [(0, SUN, 1), (1, CYAN, 1)])
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


# --- text with width budgets ------------------------------------------------------------------------------------------
# An SVG figure cannot measure its text: the viewer picks the font from the stacks above. Every label therefore gets a
# budgeted width from its number of characters, and the scripts check that the budget fits the room the layout leaves.
WIDTH_FACTOR = {(False, False): 0.60, (False, True): 0.64, (True, False): 0.62, (True, True): 0.62}
"""Budgeted advance width of one character in units of the font size, keyed by (mono, bold)."""
TEXT_SLACK = 1.08
"""Factor on every budget, a margin for system fonts that run wider than the factors above."""


def num(value: float) -> str:
    """SVG number with at most one decimal and no trailing '.0' (12.0 -> '12', 12.34 -> '12.3')."""
    s = f"{value:.1f}"
    return s[:-2] if s.endswith(".0") else s


def text_budget(text: str, size: float, *, mono: bool = False, bold: bool = False, spacing: float = 0.0) -> float:
    """Width that ``text`` may need: characters x (font size x width factor + letter-spacing), times the slack."""
    return len(text) * (size * WIDTH_FACTOR[(mono, bold)] + spacing) * TEXT_SLACK


@dataclass(frozen=True)
class Label:
    """One line of SVG text and the width the layout leaves for it.

    ``room`` is measured from the anchor point: to the right of a 'start' anchor, centred on a 'middle' anchor, to the
    left of an 'end' anchor. ``css_class`` names the animation classes of the element, if any.
    """

    text: str
    x: float
    y: float
    size: float
    room: float
    fill: str
    mono: bool = False
    bold: bool = False
    spacing: float = 0.0
    anchor: str = "start"
    css_class: str = ""

    @property
    def budget(self) -> float:
        """Budgeted width of the text, slack included (see :func:`text_budget`)."""
        return text_budget(self.text, self.size, mono=self.mono, bold=self.bold, spacing=self.spacing)

    @property
    def fits(self) -> bool:
        """True if the budget does not exceed the room."""
        return self.budget <= self.room


@dataclass
class TextLayer:
    """Collects the text of one figure: writes the ``<text>`` element of every label and keeps the label.

    ``coord`` formats the x and y attributes. After the figure is built, :meth:`report` prints the budget table and
    raises if a label is wider than its room.
    """

    coord: Callable[[float], str] = num
    labels: list[Label] = field(default_factory=list)

    def add(self, label: Label) -> str:
        """Record ``label`` and return its ``<text>`` element; raises if the font is smaller than ``MIN_FONT_PX``."""
        if label.size < MIN_FONT_PX:
            raise ValueError(f"font size {label.size} below {MIN_FONT_PX}: {label.text!r}")
        self.labels.append(label)
        attrs = f' font-family="{FONT_MONO}"' if label.mono else ""
        attrs += ' font-weight="700"' if label.bold else ""
        attrs += f' letter-spacing="{num(label.spacing)}"' if label.spacing else ""
        attrs += f' text-anchor="{label.anchor}"' if label.anchor != "start" else ""
        attrs += f' class="{label.css_class}"' if label.css_class else ""
        return (f'<text x="{self.coord(label.x)}" y="{self.coord(label.y)}" font-size="{num(label.size)}" '
                f'fill="{label.fill}"{attrs}>{escape(label.text)}</text>\n')

    def text(self, text: str, x: float, y: float, size: float, room: float, fill: str, *, mono: bool = False,
             bold: bool = False, spacing: float = 0.0, anchor: str = "start", css_class: str = "") -> str:
        """Shorthand for ``add(Label(...))`` with the same arguments."""
        return self.add(Label(text, x, y, size, room, fill, mono=mono, bold=bold, spacing=spacing, anchor=anchor,
                              css_class=css_class))

    def report(self) -> None:
        """Print one row per label (text, font size, budget, room) and raise ``ValueError`` if a label overflows."""
        width = max([len("text")] + [len(label.text) for label in self.labels])
        print(f"{'text':<{width}} {'px':>4} {'budget':>8} {'room':>8}")
        for label in self.labels:
            flag = "" if label.fits else "  OVERFLOW"
            print(f"{label.text:<{width}} {label.size:>4.0f} {label.budget:>8.1f} {label.room:>8.1f}{flag}")
        over = [label.text for label in self.labels if not label.fits]
        if over:
            raise ValueError(f"text wider than its room: {over}")


# --- solar disk -------------------------------------------------------------------------------------------------------
# The Sun is drawn as an instrument view: a dark disk with a faint orange rim, a soft orange glow outside the limb, a
# faint graticule of the solar sphere and a thin orange limb line on top.
DISK_FILL_STOPS: tuple[GradientStop, ...] = ((0, PANEL, 1), (0.82, BG_SOFT, 1), (1, blend(BG_SOFT, SUN, 0.12), 1))
"""Radial gradient of the dark disk: panel blue in the centre, darker towards the limb, a faint orange rim."""
GRATICULE_DEGREES = (-60, -30, 0, 30, 60)
"""Heliographic longitudes of the drawn meridians and latitudes of the drawn parallels (degrees)."""


def halo_stops(limb: float) -> list[GradientStop]:
    """Radial gradient of the glow around the disk; ``limb`` is the disk radius divided by the glow radius."""
    return [(limb, SUN, 0.24), ((limb + 1) / 2, SUN, 0.06), (1, SUN, 0)]


def sphere_to_plane(lat: np.ndarray, lon: np.ndarray, b0: float, centre: tuple[float, float],
                    radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic projection of heliographic (lat, lon) in degrees onto a disk of ``radius`` around ``centre``.

    ``b0`` is the heliographic latitude of the disk centre; solar north is up. Returns the points (n, 2), rounded to
    0.1, and a boolean array that is True on the visible hemisphere.
    """
    la, lo, b = np.radians(lat), np.radians(lon), math.radians(b0)
    x = np.cos(la) * np.sin(lo)
    up = np.sin(la) * math.cos(b) - np.cos(la) * math.sin(b) * np.cos(lo)
    depth = np.sin(la) * math.sin(b) + np.cos(la) * math.cos(b) * np.cos(lo)
    cx, cy = centre
    points = np.round(np.stack([cx + radius * x, cy - radius * up], axis=1), 1)
    return points, depth > 0


def visible_runs(points: np.ndarray, visible: np.ndarray) -> list[np.ndarray]:
    """Consecutive visible stretches (at least two points) of a sampled curve."""
    runs, start = [], None
    for i, v in enumerate(list(visible) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 2:
                runs.append(points[start:i])
            start = None
    return runs


def graticule_path(centre: tuple[float, float], radius: float, b0: float, step: float,
                   lines: Sequence[float] = GRATICULE_DEGREES) -> str:
    """Path data of the meridians and parallels at ``lines`` degrees on the visible hemisphere of a disk.

    The arcs are sampled every ``step`` degrees; ``b0`` is the heliographic latitude of the disk centre.
    """
    lat = np.arange(-90.0, 90.0 + step, step)
    lon = np.arange(-180.0, 180.0 + step, step)
    runs = []
    for meridian in lines:
        runs += visible_runs(*sphere_to_plane(lat, np.full_like(lat, meridian), b0, centre, radius))
    for parallel in lines:
        runs += visible_runs(*sphere_to_plane(np.full_like(lon, parallel), lon, b0, centre, radius))
    return " ".join(path_from_points(r, closed=False) for r in runs)
