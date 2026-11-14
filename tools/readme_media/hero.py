"""Hero banner of the README and the animated section divider.

``docs/media/hero.svg`` (1760 x 560). Left: the solar disk of one test image as an instrument view (dark disk, limb,
graticule of the solar sphere, position-angle ticks) with the outlines of the instances that the final model keeps in
that image, decoded from the run-length masks of ``reference/submission.csv``. A scan beam sweeps the disk in 5.6 s;
each outline is traced when the beam reaches it and then filled; the complete drawing holds until 9.2 s and fades out
by 9.7 s, before the next sweep (loop of 10 s). Static labels give the observation time, the instrument and the
number of kept instances. Right: title, subtitle and four headline numbers taken from
``tools/readme_media/numbers.json`` (the CPU run time is the one stated in the README). With animation switched off the
figure shows every outline, filled.

Choice of the test image: the images of the reference submission are ranked by their number of kept instances (ties
by file stem) and the first one whose instance centroids lie in all four quadrants of the disk and spread over at
least 0.45 disk radii (root-mean-square distance of the centroids from their mean) is drawn. The masks are checked
before use (pixel-disjoint, plausible areas, centroids on the disk). The graticule is tilted by the heliographic
latitude B0 of the disk centre at the observation time read from the file stem, with solar north up.

``docs/media/divider.svg`` (1760 x 24, transparent): a slate hairline with a short orange-to-cyan segment that travels
from left to right every 6 s.

Usage: ``python tools/readme_media/hero.py``; prints the chosen image and the text-width budget of every label.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

import style  # noqa: E402
from style import Label  # noqa: E402
from pqtree.config import DISK_RADIUS, H, MIN_AREA, W  # noqa: E402

# --- inputs -----------------------------------------------------------------------------------------------------------
SUBMISSION = ROOT / "reference" / "submission.csv"
NUMBERS = Path(__file__).resolve().parent / "numbers.json"
CPU_RUNTIME = "1 min"
"""Run time of ``python -m pqtree.pipeline`` on 4 CPU cores, as stated in the README."""
IMAGE_CENTRE = (W / 2, H / 2)
"""Centre of the solar disk in image coordinates (x, y)."""
MIN_SPREAD = 0.45
"""Smallest RMS distance of the instance centroids from their mean (in disk radii) for the drawn image."""
MAX_AREA_SHARE = 0.02
"""Largest plausible instance area as a share of the disk area (a sanity check of the decoded masks)."""
APPROX_EPS = 1.5
"""Tolerance (image pixels) of the polygon simplification of the outlines."""

# --- canvas and grid --------------------------------------------------------------------------------------------------
FIG_W, FIG_H = 1760, 560
MARGIN = 48
DISK_X, DISK_Y, DISK_R = 448.0, 300.0, 186.0
"""Centre and radius of the drawn disk (figure pixels)."""
SCALE = DISK_R / DISK_RADIUS
"""Figure pixels per image pixel."""
CLEARANCE = 14
"""Distance that text keeps from the limb (the halo and the ticks lie inside it)."""
HALO_WIDTH = 26
TEXT_X, TEXT_RIGHT = 688, FIG_W - MARGIN
"""Left and right edge of the text block."""
PANEL_RIGHT = TEXT_X - MARGIN
"""Right edge of the free space of the disk panel."""

# --- type -------------------------------------------------------------------------------------------------------------
ASCENT, DESCENT = 0.75, 0.25
"""Share of the font size above and below the baseline used for text boxes."""
HUD_SIZE = 26
HUD_TOP_Y, HUD_BOTTOM_Y, HUD_STEP = 84, 474, 36
EYEBROW_Y, EYEBROW_SIZE, EYEBROW_SPACING = 80, 26, 4
TITLE_Y, TITLE_SIZE = (158, 234), 70
RULE_Y, RULE_H = 264, 3
SUBTITLE_Y, SUBTITLE_SIZE = (312, 352), 32
TILE_Y, TILE_H, TILE_GAP, TILE_PAD, TILE_RX = 376, 136, 16, 20, 14
N_TILES = 4
TILE_W = (TEXT_RIGHT - TEXT_X - (N_TILES - 1) * TILE_GAP) / N_TILES
VALUE_DY, VALUE_SIZE = 50, 44
LABEL_DY, LABEL_SIZE = (84, 114), 26

# --- motion -----------------------------------------------------------------------------------------------------------
LOOP_S = 10.0
BEAM_END_S = 5.6
"""The beam travels from the left limb (t = 0) until its trail has left the disk at the right."""
BEAM_TRAIL = 90
DRAW_S, FILL_S = 0.8, 0.5
"""Duration of the outline trace and of the fill fade-in of one instance."""
FADE_S = (9.2, 9.7)
"""Start and end of the common fade-out."""
FILL_OPACITY = 0.32
N_STARS, STAR_SEED, STAR_GAP, STAR_TRIES = 12, 7, 40, 5000
GLINT_W, GLINT_S = 60, 8.0
"""Width and period of the light that runs along the accent rule."""
BREATHE_S = 8.0
"""Period of the breathing outline of the stat tiles (neighbouring tiles are shifted by 1 / N_TILES period)."""
DIVIDER_W, DIVIDER_H, SEGMENT_W, DIVIDER_S = 1760, 24, 220, 6.0

# --- graticule --------------------------------------------------------------------------------------------------------
GRATICULE_STEP = 2.0
"""Sampling step (degrees) of the graticule arcs."""
JD_UNIX_EPOCH, JD_J2000 = 2440587.5, 2451545.0
SOLAR_AXIS_INCLINATION = 7.25
"""Inclination (degrees) of the solar equator on the ecliptic."""


# --- image selection --------------------------------------------------------------------------------------------------
def observation_time(stem: str) -> datetime:
    """UTC time encoded in the first 14 characters of a file stem (YYYYMMDDhhmmss)."""
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def disk_centre_latitude(when: datetime) -> float:
    """Heliographic latitude B0 (degrees) of the centre of the solar disk at ``when``.

    Low-precision solar longitude (Meeus, Astronomical Algorithms, 2nd ed., ch. 25) and the B0 formula of ch. 29; far
    more precise than the graticule needs.
    """
    jd = JD_UNIX_EPOCH + when.timestamp() / 86400.0
    t = (jd - JD_J2000) / 36525.0
    mean_longitude = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    anomaly = math.radians(357.52911 + 35999.05029 * t - 0.0001537 * t * t)
    centre = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(anomaly)
              + (0.019993 - 0.000101 * t) * math.sin(2 * anomaly) + 0.000289 * math.sin(3 * anomaly))
    node = 73.6667 + 1.3958333 * (jd - 2396758.0) / 36525.0
    phase = math.radians(mean_longitude + centre - node)
    return math.degrees(math.asin(math.sin(phase) * math.sin(math.radians(SOLAR_AXIS_INCLINATION))))


def decode_masks(rles: list[str]) -> np.ndarray:
    """Decode the run-length masks of one image to an (H, W, n) array and check them.

    The masks are written by :func:`pqtree.selection.submission_rows` from Fortran-ordered (H, W) arrays, so the decoded
    array is indexed [y, x]. Raises ``ValueError`` if the masks overlap, an area is implausible or a centroid lies off
    the disk.
    """
    objs = [{"size": [H, W], "counts": r.encode("ascii")} for r in rles]
    areas = mu.area(objs)
    if mu.area(mu.merge(objs)) != areas.sum():
        raise ValueError("masks overlap")
    if areas.min() < MIN_AREA or areas.max() > MAX_AREA_SHARE * math.pi * DISK_RADIUS ** 2:
        raise ValueError(f"implausible areas {areas.min()} .. {areas.max()}")
    masks = mu.decode(objs)
    radius = np.hypot(*(centroids(masks) - np.array(IMAGE_CENTRE)).T)
    if radius.max() >= DISK_RADIUS:
        raise ValueError("instance centroid off the disk")
    return masks


def centroids(masks: np.ndarray) -> np.ndarray:
    """(n, 2) array of the (x, y) centroids of the masks of an (H, W, n) array."""
    out = []
    for k in range(masks.shape[2]):
        ys, xs = np.nonzero(masks[:, :, k])
        out.append((xs.mean(), ys.mean()))
    return np.array(out)


def centroid_spread(points: np.ndarray) -> tuple[int, float]:
    """Number of disk quadrants occupied by the points and their RMS distance from their mean (in disk radii)."""
    offset = points - np.array(IMAGE_CENTRE)
    quadrants = len(set(zip(offset[:, 0] > 0, offset[:, 1] > 0)))
    rms = np.sqrt(((points - points.mean(0)) ** 2).sum(1).mean()) / DISK_RADIUS
    return quadrants, float(rms)


def pick_image(sub: pd.DataFrame) -> tuple[str, np.ndarray]:
    """The drawn test image (rule of the module docstring) and its decoded masks."""
    stems = sub.filament_id.str.rsplit("_", n=1).str[0]
    counts = stems.value_counts()
    for stem in sorted(counts.index, key=lambda s: (-counts[s], s)):
        masks = decode_masks(sub.segmentation_rle[stems == stem].tolist())
        quadrants, rms = centroid_spread(centroids(masks))
        if quadrants == 4 and rms >= MIN_SPREAD:
            return stem, masks
    raise ValueError("no test image satisfies the spread rule")


# --- geometry ---------------------------------------------------------------------------------------------------------
def to_figure(points: np.ndarray) -> np.ndarray:
    """Image coordinates (x, y) -> figure coordinates of the drawn disk, rounded to 0.1."""
    centre = np.array(IMAGE_CENTRE)
    return np.round((points - centre) * SCALE + np.array([DISK_X, DISK_Y]), 1)


@dataclass(frozen=True)
class Outline:
    """Outer contours of one kept instance in figure coordinates."""

    rings: tuple[np.ndarray, ...]

    @property
    def perimeter(self) -> float:
        """Total length of the closed rings."""
        return float(sum(np.hypot(*(np.roll(r, -1, axis=0) - r).T).sum() for r in self.rings))

    @property
    def dash(self) -> int:
        """Dash length (and hidden offset) of the trace: the perimeter rounded up, plus one."""
        return math.ceil(self.perimeter) + 1

    @property
    def xmin(self) -> float:
        """Leftmost figure x of the instance (the beam reaches it there)."""
        return float(min(r[:, 0].min() for r in self.rings))

    def path(self) -> str:
        """SVG path data of the rings."""
        return " ".join(style.path_from_points(r) for r in self.rings)


def outline(mask: np.ndarray) -> Outline:
    """Outer contours of a mask, simplified by ``APPROX_EPS`` image pixels and mapped to the figure."""
    contours, _ = cv2.findContours(np.ascontiguousarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    rings = []
    for c in contours:
        ring = to_figure(cv2.approxPolyDP(c, APPROX_EPS, True)[:, 0, :].astype(float))
        keep = np.any(ring != np.roll(ring, 1, axis=0), axis=1)
        if keep.sum() >= 3:
            rings.append(ring[keep])
    return Outline(tuple(rings))


def limb_ticks(step: int = 15, major: int = 90) -> tuple[str, str]:
    """Path data of the minor and the major position-angle ticks just outside the limb."""
    minor_d, major_d = [], []
    for angle in range(0, 360, step):
        length = 8 if angle % major == 0 else 4
        ux, uy = math.sin(math.radians(angle)), -math.cos(math.radians(angle))
        inner, outer = DISK_R + 4, DISK_R + 4 + length
        seg = (f"M {DISK_X + inner * ux:.1f} {DISK_Y + inner * uy:.1f} "
               f"L {DISK_X + outer * ux:.1f} {DISK_Y + outer * uy:.1f}")
        (major_d if angle % major == 0 else minor_d).append(seg)
    return " ".join(minor_d), " ".join(major_d)


# --- text -------------------------------------------------------------------------------------------------------------
def coordinate(value: float) -> str:
    """Coordinate attribute of the hero's text: always one decimal, like the other coordinates of this figure."""
    return f"{value:.1f}"


def text_box(label: Label) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the budgeted box of a left-aligned label, slack included."""
    return label.x, label.y - ASCENT * label.size, label.x + label.budget, label.y + DESCENT * label.size


def hud_available(x: float, y: float, size: int) -> float:
    """Width available to a HUD line that starts at ``x``: up to the disk clearance circle or the panel edge."""
    top, bottom = y - ASCENT * size, y + DESCENT * size
    dy = 0.0 if top <= DISK_Y <= bottom else min(abs(top - DISK_Y), abs(bottom - DISK_Y))
    reach = DISK_R + CLEARANCE
    if dy >= reach:
        return PANEL_RIGHT - x
    return DISK_X - math.sqrt(reach ** 2 - dy ** 2) - x


def hud_labels(stem: str, n_kept: int) -> list[Label]:
    """Observation time and number of kept instances (top left), instrument and image size (bottom left)."""
    when = observation_time(stem).strftime("%Y-%m-%d %H:%M UT")
    lines = [(HUD_TOP_Y, when, style.MUTED), (HUD_TOP_Y + HUD_STEP, f"{n_kept} filaments kept", style.GREEN),
             (HUD_BOTTOM_Y, "GONG H-alpha", style.MUTED), (HUD_BOTTOM_Y + HUD_STEP, f"{W} x {H}", style.MUTED)]
    return [Label(text, MARGIN, y, HUD_SIZE, hud_available(MARGIN, y, HUD_SIZE), colour, mono=True)
            for y, text, colour in lines]


def tile_specs(numbers: dict) -> list[tuple[str, tuple[str, str]]]:
    """Value and two label lines of each stat tile."""
    final = numbers["ablation"][-1]
    counts = numbers["counts"]
    return [(f"{final['oof_pq']:.4f}", ("out-of-fold", "PQ")),
            (f"{final['public']:.2f}", ("public", "leaderboard")),
            (f"{counts['test_instances']:,}", ("instances in", f"{counts['test_images']} images")),
            (CPU_RUNTIME, ("CPU", "reproduction"))]


def text_block_labels(numbers: dict) -> list[Label]:
    """Eyebrow, title, subtitle and the text of the stat tiles."""
    width = TEXT_RIGHT - TEXT_X
    labels = [Label("SOLAR FILAMENT SEGMENTATION CHALLENGE 2026", TEXT_X, EYEBROW_Y, EYEBROW_SIZE, width, style.SUN,
                    mono=True, spacing=EYEBROW_SPACING)]
    labels += [Label(t, TEXT_X, y, TITLE_SIZE, width, style.TEXT, bold=True)
               for t, y in zip(("Solar filament", "instance segmentation"), TITLE_Y)]
    labels += [Label(t, TEXT_X, y, SUBTITLE_SIZE, width, style.MUTED)
               for t, y in zip(("Panoptic-quality-optimal selection", "on a component tree"), SUBTITLE_Y)]
    inner = TILE_W - 2 * TILE_PAD
    for i, (value, lines) in enumerate(tile_specs(numbers)):
        x = TEXT_X + i * (TILE_W + TILE_GAP) + TILE_PAD
        labels.append(Label(value, x, TILE_Y + VALUE_DY, VALUE_SIZE, inner, style.TEXT, mono=True))
        labels += [Label(t, x, TILE_Y + dy, LABEL_SIZE, inner, style.MUTED) for t, dy in zip(lines, LABEL_DY)]
    return labels


# --- hero -------------------------------------------------------------------------------------------------------------
def pct(seconds: float, loop: float = LOOP_S) -> str:
    """Keyframe position of a time within the loop."""
    return f"{100 * seconds / loop:.2f}%"


def beam_span() -> tuple[float, float]:
    """Translation of the beam line at the start and at the end of the sweep."""
    return DISK_X - DISK_R - 4, DISK_X + DISK_R + BEAM_TRAIL + 4


def reveal_time(x: float) -> float:
    """Time (s) at which the beam line reaches figure x."""
    x0, x1 = beam_span()
    return (x - x0) / (x1 - x0) * BEAM_END_S


def stars(avoid: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float, float]]:
    """Deterministic faint stars in the disk panel: (x, y, radius, period s, phase s), clear of disk and labels."""
    rng = np.random.default_rng(STAR_SEED)
    out: list[tuple[float, float, float, float, float]] = []
    for _ in range(STAR_TRIES):
        x, y = rng.uniform(MARGIN, PANEL_RIGHT), rng.uniform(MARGIN, FIG_H - MARGIN)
        radius, period, phase = rng.uniform(0.9, 1.7), rng.uniform(6.0, 10.0), rng.uniform(0.0, 10.0)
        if math.hypot(x - DISK_X, y - DISK_Y) < DISK_R + HALO_WIDTH + STAR_GAP / 2:
            continue
        if any(x0 - 12 <= x <= x1 + 12 and y0 - 12 <= y <= y1 + 12 for x0, y0, x1, y1 in avoid):
            continue
        if any(math.hypot(x - s[0], y - s[1]) < STAR_GAP for s in out):
            continue
        out.append((x, y, radius, period, phase))
        if len(out) == N_STARS:
            break
    return out


def hero_defs() -> str:
    """Gradients, clip paths and the fine glow of the hero, added to the shared definitions."""
    limb = DISK_R / (DISK_R + HALO_WIDTH)
    disk = f'cx="{DISK_X}" cy="{DISK_Y}"'
    horizontal = 'x1="0" y1="0" x2="1" y2="0"'
    return "\n".join([
        f'<radialGradient id="disk-fill" gradientUnits="userSpaceOnUse" {disk} r="{DISK_R}">'
        + style.gradient_stops(style.DISK_FILL_STOPS) + "</radialGradient>",
        f'<radialGradient id="disk-halo" gradientUnits="userSpaceOnUse" {disk} r="{DISK_R + HALO_WIDTH}">'
        + style.gradient_stops(style.halo_stops(limb)) + "</radialGradient>",
        f'<linearGradient id="beam-trail" {horizontal}>'
        + style.gradient_stops([(0, style.CYAN, 0), (1, style.CYAN, 0.2)]) + "</linearGradient>",
        f'<linearGradient id="glint-fill" {horizontal}>'
        + style.gradient_stops([(0, style.TEXT, 0), (0.5, style.TEXT, 0.9), (1, style.TEXT, 0)]) + "</linearGradient>",
        f'<clipPath id="disk-clip"><circle {disk} r="{DISK_R}"/></clipPath>',
        f'<clipPath id="rule-clip"><rect x="{TEXT_X}" y="{RULE_Y}" width="{TILE_W:g}" height="{RULE_H}"/></clipPath>',
        '<filter id="glow-fine" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
    ]) + "\n"


def trace_keyframes(i: int, o: Outline) -> str:
    """Keyframes of instance ``i``: hidden until the beam reaches it, outline traced, then filled."""
    t0 = reveal_time(o.xmin)
    hidden = f"stroke-dashoffset: {o.dash}px; fill-opacity: 0;"
    traced = "stroke-dashoffset: 0px; fill-opacity: 0;"
    filled = f"stroke-dashoffset: 0px; fill-opacity: {FILL_OPACITY};"
    return (f"@keyframes f{i} {{ 0%, {pct(t0)} {{ {hidden} }} {pct(t0 + DRAW_S)} {{ {traced} }} "
            f"{pct(t0 + DRAW_S + FILL_S)}, 100% {{ {filled} }} }}")


def hero_css(outlines: list[Outline]) -> str:
    """One style block: sweep, per-instance trace, common fade-out, twinkle, glint, breathing tiles."""
    x0, x1 = beam_span()
    rules = [
        f".beam {{ animation: sweep {LOOP_S:g}s linear infinite; }}",
        f"@keyframes sweep {{ 0% {{ transform: translateX({x0:.1f}px); }} "
        f"{pct(BEAM_END_S)}, 100% {{ transform: translateX({x1:.1f}px); }} }}",
        f".fils {{ animation: cycle {LOOP_S:g}s linear infinite; }}",
        f"@keyframes cycle {{ 0%, {pct(FADE_S[0])} {{ opacity: 1; }} {pct(FADE_S[1])}, 100% {{ opacity: 0; }} }}",
    ]
    for i, o in enumerate(outlines):
        rules += [f".f{i} {{ animation: f{i} {LOOP_S:g}s ease-in-out infinite; }}", trace_keyframes(i, o)]
    rules += [
        ".star { animation-name: twinkle; animation-timing-function: ease-in-out; "
        "animation-iteration-count: infinite; }",
        "@keyframes twinkle { 0%, 100% { opacity: 0.2; } 50% { opacity: 0.85; } }",
        f".glint {{ animation: glint {GLINT_S:g}s ease-in-out infinite; }}",
        f"@keyframes glint {{ 0%, 10% {{ transform: translateX({TEXT_X - GLINT_W}px); }} "
        f"45%, 100% {{ transform: translateX({TEXT_X + TILE_W:g}px); }} }}",
        f".breathe {{ opacity: 0; animation: breathe {BREATHE_S:g}s ease-in-out infinite; }}",
        "@keyframes breathe { 0%, 100% { opacity: 0; } 50% { opacity: 0.55; } }",
        style.REDUCED_MOTION_CSS,
    ]
    return "<style>\n" + "\n".join(rules) + "\n</style>\n"


def disk_layer(b0: float) -> str:
    """Halo, disk, graticule, position-angle ticks and limb."""
    minor, major = limb_ticks()
    graticule = style.graticule_path((DISK_X, DISK_Y), DISK_R, b0, GRATICULE_STEP)
    return (f'<circle cx="{DISK_X}" cy="{DISK_Y}" r="{DISK_R + HALO_WIDTH}" fill="url(#disk-halo)"/>\n'
            f'<circle cx="{DISK_X}" cy="{DISK_Y}" r="{DISK_R}" fill="url(#disk-fill)"/>\n'
            f'<path d="{graticule}" fill="none" stroke="{style.PANEL_EDGE}" stroke-width="1.2"/>\n'
            f'<path d="{minor}" stroke="{style.PANEL_EDGE}" stroke-width="1.5"/>\n'
            f'<path d="{major}" stroke="{style.SLATE}" stroke-width="1.5"/>\n'
            f'<circle cx="{DISK_X}" cy="{DISK_Y}" r="{DISK_R}" fill="none" stroke="{style.SUN}" stroke-width="1.6" '
            f'stroke-opacity="0.9"/>\n')


def beam_layer() -> str:
    """Scan line with its trailing gradient, clipped to the disk; at rest it lies outside the clip."""
    top, height = DISK_Y - DISK_R, 2 * DISK_R
    return (f'<g clip-path="url(#disk-clip)"><g class="beam">'
            f'<rect x="{-BEAM_TRAIL}" y="{top}" width="{BEAM_TRAIL}" height="{height}" fill="url(#beam-trail)"/>'
            f'<line x1="0" y1="{top}" x2="0" y2="{top + height}" stroke="{style.CYAN}" stroke-width="6" '
            f'stroke-opacity="0.15"/>'
            f'<line x1="0" y1="{top}" x2="0" y2="{top + height}" stroke="{style.CYAN}" stroke-width="1.5"/>'
            f'</g></g>\n')


def filament_layer(outlines: list[Outline]) -> str:
    """Outlines of the kept instances (cyan trace, green fill)."""
    paths = "".join(f'<path class="f{i}" d="{o.path()}" stroke-dasharray="{o.dash}"/>\n'
                    for i, o in enumerate(outlines))
    return (f'<g class="fils" fill="{style.GREEN}" fill-opacity="{FILL_OPACITY}" stroke="{style.CYAN}" '
            f'stroke-width="1.3" stroke-linejoin="round" filter="url(#glow-fine)">\n{paths}</g>\n')


def star_layer(points: list[tuple[float, float, float, float, float]]) -> str:
    """Twinkling stars (visible at rest)."""
    return "".join(f'<circle class="star" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{style.TEXT}" opacity="0.5" '
                   f'style="animation-duration:{period:.1f}s;animation-delay:-{phase:.1f}s"/>\n'
                   for x, y, r, period, phase in points)


def text_block_shapes() -> str:
    """Accent rule with its glint and the four tile cards with their breathing outline."""
    out = [f'<rect x="{TEXT_X}" y="{RULE_Y}" width="{TILE_W:g}" height="{RULE_H}" fill="url(#accent)"/>\n',
           f'<g clip-path="url(#rule-clip)"><rect class="glint" x="0" y="{RULE_Y}" width="{GLINT_W}" '
           f'height="{RULE_H}" fill="url(#glint-fill)"/></g>\n']
    for i in range(N_TILES):
        x = TEXT_X + i * (TILE_W + TILE_GAP)
        box = f'x="{x:g}" y="{TILE_Y}" width="{TILE_W:g}" height="{TILE_H}" rx="{TILE_RX}"'
        delay = BREATHE_S * i / N_TILES
        out.append(f'<rect {box} fill="{style.PANEL}" stroke="{style.PANEL_EDGE}" stroke-width="1.5"/>\n')
        out.append(f'<rect class="breathe" {box} fill="none" stroke="url(#accent)" stroke-width="1.5"'
                   + (f' style="animation-delay:-{delay:g}s"' if delay else "") + "/>\n")
    return "".join(out)


def build_hero(stem: str, outlines: list[Outline], numbers: dict) -> tuple[str, style.TextLayer]:
    """The hero SVG and the text layer that holds its labels."""
    b0 = disk_centre_latitude(observation_time(stem))
    hud = hud_labels(stem, len(outlines))
    layer = style.TextLayer(coord=coordinate)
    text = "".join(layer.add(label) for label in hud + text_block_labels(numbers))
    final, counts = numbers["ablation"][-1], numbers["counts"]
    description = (f"Full-disk view of test image {stem} with the outlines of the {len(outlines)} filament "
                   f"instances kept by the model. Out-of-fold PQ {final['oof_pq']:.4f}, public leaderboard "
                   f"{final['public']:.2f}, {counts['test_instances']:,} instances in {counts['test_images']} "
                   f"test images, reproduction on CPU in about {CPU_RUNTIME}.")
    parts = [style.svg_open(FIG_W, FIG_H, "Solar filament instance segmentation", description),
             style.svg_defs(hero_defs()), hero_css(outlines), style.svg_background(FIG_W, FIG_H),
             star_layer(stars([text_box(label) for label in hud])), disk_layer(b0), beam_layer(),
             filament_layer(outlines), text_block_shapes(), text, "</svg>\n"]
    return "".join(parts), layer


# --- divider ----------------------------------------------------------------------------------------------------------
def build_divider() -> str:
    """Transparent hairline with a travelling orange-to-cyan segment; at rest the segment sits in the middle."""
    x_rest = (DIVIDER_W - SEGMENT_W) / 2
    y = DIVIDER_H / 2
    stops = style.gradient_stops([(0, style.SUN, 0), (0.3, style.SUN, 1), (0.85, style.CYAN, 1), (1, style.CYAN, 0)])
    css = (f".run {{ animation: run {DIVIDER_S:g}s cubic-bezier(0.65, 0, 0.35, 1) infinite; }}\n"
           f"@keyframes run {{ 0% {{ transform: translateX({-(x_rest + SEGMENT_W):g}px); }} "
           f"100% {{ transform: translateX({DIVIDER_W - x_rest:g}px); }} }}\n{style.REDUCED_MOTION_CSS}")
    return (style.svg_open(DIVIDER_W, DIVIDER_H, "Section divider",
                           "A thin slate line with a short orange-to-cyan segment travelling from left to right.")
            + f'<defs><linearGradient id="segment" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient></defs>\n'
            + f"<style>\n{css}\n</style>\n"
            + f'<line x1="0" y1="{y:g}" x2="{DIVIDER_W}" y2="{y:g}" stroke="{style.SLATE}" stroke-width="1.5" '
              f'stroke-opacity="0.8"/>\n'
            + f'<rect class="run" x="{x_rest:g}" y="{y - 1.5:g}" width="{SEGMENT_W}" height="3" rx="1.5" '
              f'fill="url(#segment)"/>\n</svg>\n')


# --- main -------------------------------------------------------------------------------------------------------------
def main() -> None:
    """Pick the image, build both figures, print the budget and write the files."""
    numbers = json.loads(NUMBERS.read_text())
    sub = pd.read_csv(SUBMISSION)
    stem, masks = pick_image(sub)
    outlines = sorted((outline(masks[:, :, k]) for k in range(masks.shape[2])), key=lambda o: o.xmin)
    quadrants, rms = centroid_spread(centroids(masks))
    print(f"image {stem}: {len(outlines)} instances, {quadrants} quadrants, spread {rms:.3f} disk radii, "
          f"B0 {disk_centre_latitude(observation_time(stem)):+.2f} deg")
    hero, layer = build_hero(stem, outlines, numbers)
    layer.report()
    for name, text in (("hero.svg", hero), ("divider.svg", build_divider())):
        print(f"wrote docs/media/{name}: {style.write_text(style.MEDIA / name, text)} bytes")


if __name__ == "__main__":
    main()
