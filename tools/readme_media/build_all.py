"""Draw every figure of the README into ``docs/media`` from the stored artefacts.

usage:
  python tools/readme_media/build_all.py            # all figures (about 8 minutes on 3 CPU cores)
  python tools/readme_media/build_all.py hero       # only the named ones

Each figure has its own script in this folder; this file runs them one after the other in separate processes, so that
a figure can also be rebuilt alone with ``python tools/readme_media/<name>.py``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FIGURES: dict[str, str] = {
    "hero": "hero.svg and divider.svg: predicted filaments of one full-disk test image; the section divider",
    "pipeline_diagram": "pipeline.svg: the six stages",
    "component_tree": "component_tree.gif: one region from the probability map to the kept instances",
    "threshold_sweep": "threshold_sweep.gif: the selection threshold on a full-disk map and on the out-of-fold curve",
    "results_charts": "results.svg: out-of-fold PQ by configuration, PQ = SQ x RQ, quantiles of the matched pairs",
}
"""Script name (without .py) -> what it writes."""


def build(names: list[str]) -> None:
    """Run the scripts of the named figures; stop at the first one that fails."""
    for name in names:
        print(f"== {name}: {FIGURES[name]}", flush=True)
        subprocess.run([sys.executable, str(HERE / f"{name}.py")], check=True)


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    build(names)


if __name__ == "__main__":
    main()
