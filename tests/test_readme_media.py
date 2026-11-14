"""Checks of the figures of the README: every referenced file exists and is self-contained."""
from __future__ import annotations

import re
import xml.dom.minidom
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "docs" / "media"
REFERENCE = re.compile(r'src="(docs/media/[^"]+)"')


def referenced_files() -> list[Path]:
    """Figure files that README.md shows."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    return sorted({REPO / name for name in REFERENCE.findall(text)})


def test_every_referenced_figure_exists():
    files = referenced_files()
    assert files, "the README shows no figure"
    missing = [str(f.relative_to(REPO)) for f in files if not f.is_file()]
    assert not missing, missing


def test_every_figure_is_shown_and_has_a_script():
    shown = {f.name for f in referenced_files()}
    stored = {f.name for f in MEDIA.iterdir() if f.is_file()}
    assert stored == shown, stored ^ shown
    scripts = "\n".join(p.read_text(encoding="utf-8") for p in (REPO / "tools" / "readme_media").glob("*.py"))
    assert all(name in scripts for name in stored), [name for name in stored if name not in scripts]


def test_svg_figures_are_well_formed_and_self_contained():
    for path in MEDIA.glob("*.svg"):
        text = path.read_text(encoding="utf-8")
        document = xml.dom.minidom.parseString(text)
        assert document.documentElement.tagName == "svg", path.name
        for forbidden in ("<script", "<foreignObject", "<image", "@import", "href="):
            assert forbidden not in text, (path.name, forbidden)
        urls = set(re.findall(r'https?://[^"\s<)]+', text))
        assert urls <= {"http://www.w3.org/2000/svg"}, (path.name, urls)
        assert "prefers-reduced-motion" in text, path.name


def test_gif_figures_loop():
    for path in MEDIA.glob("*.gif"):
        data = path.read_bytes()
        assert data[:6] == b"GIF89a", path.name
        assert b"NETSCAPE2.0" in data, path.name
        assert len(data) < 4_000_000, (path.name, len(data))
