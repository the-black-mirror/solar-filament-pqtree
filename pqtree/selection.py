"""Stage 6: choose a pixel-disjoint set of candidates per image and write the submission.

Pooled Panoptic Quality is a ratio, PQ = sum(IoU of matches) / (TP + FP/2 + FN/2). Every predicted instance adds 1/2 per
annotator to the denominator whether it matches or not, so (Dinkelbach's argument) the PQ-optimal set keeps a candidate
exactly when its expected matched IoU exceeds ``tau = PQ / 2``, subject to the kept candidates being disjoint. In the
component tree two candidates overlap only if one is an ancestor of the other, so the best disjoint set is an antichain
of the forest and is found exactly by dynamic programming.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pycocotools import mask as mu

from .config import H, W


def select(df: pd.DataFrame, score, tau: float) -> list[tuple[str, int, int]]:
    """Antichain of each image's component forest that maximises the sum of ``score - tau``.

    ``df`` has the columns stem, level, cid, parent (``parent`` is the component id at ``level - 1``). Returns the chosen
    (stem, level, cid) triples. Candidates with ``score <= tau`` are never chosen.
    """
    df = df.assign(g=score - tau)
    chosen = []
    for stem, d in df.groupby("stem", sort=False):
        d = d.sort_values("level", ascending=False)
        best: dict = {}
        pick: dict = {}
        children: dict = {}
        for row in d.itertuples():
            k = (row.level, row.cid)
            cs = children.get(k, [])
            sc = sum(best[c] for c in cs)
            if row.g >= sc and row.g > 0:
                best[k] = row.g
                pick[k] = [k]
            else:
                best[k] = max(sc, 0.0)
                pick[k] = [p for c in cs for p in pick[c]] if sc > 0 else []
            if row.level > 0:
                children.setdefault((row.level - 1, row.parent), []).append(k)
        for row in d[d.level == 0].itertuples():
            for (lv, cid) in pick[(0, row.cid)]:
                chosen.append((stem, lv, cid))
    return chosen


def submission_rows(df: pd.DataFrame, score, chosen: list[tuple[str, int, int]],
                    label_images: Callable[[str], Callable[[int], np.ndarray]]) -> tuple[list[tuple[str, str]], int]:
    """RLE rows (``<stem>_<k>``, counts) for the chosen candidates of every image in ``df``.

    ``label_images(stem)`` returns a function ``level -> label image``. An image without any chosen candidate gets its
    single highest-scoring candidate (the metric needs at least one instance). Returns the rows and the number of such
    fallback images.
    """
    by_stem: dict = {}
    for st, lv, cid in chosen:
        by_stem.setdefault(st, []).append((lv, cid))
    rows, n_fallback = [], 0
    for stem in sorted(df.stem.unique()):
        labs = label_images(stem)
        items = by_stem.get(stem, [])
        if not items:
            n_fallback += 1
            d = df.assign(s_=score)[df.stem == stem].sort_values("s_", ascending=False).iloc[0]
            items = [(int(d.level), int(d.cid))]
        for j, (lv, cid) in enumerate(items, 1):
            m = np.asfortranarray((labs(lv) == cid).astype(np.uint8))
            rows.append((f"{stem}_{j}", mu.encode(m)["counts"].decode("ascii")))
    return rows, n_fallback


def stored_label_images(npz_path: str | os.PathLike) -> Callable[[str], Callable[[int], np.ndarray]]:
    """Label images from an archive written by :class:`LabelWriter` (one array per stem and level)."""
    z = np.load(npz_path)

    def for_stem(stem: str):
        cache: dict = {}

        def labels(level: int) -> np.ndarray:
            if level not in cache:
                cache[level] = z[f"{stem}__{level}"]
            return cache[level]

        return labels

    return for_stem


class LabelWriter:
    """Streams label images into one compressed ``.npz`` archive, one image at a time (they are large).

    Use as a context manager and call :meth:`add` with a stem and its list of label images (one per level); the arrays are
    stored as uint16 under the names ``<stem>__<level>``.
    """

    def __init__(self, path: str | os.PathLike):
        self._zip = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)

    def add(self, stem: str, labels: list[np.ndarray]) -> None:
        """Append the label images of one stem."""
        for i, lab in enumerate(labels):
            buf = io.BytesIO()
            np.save(buf, lab.astype(np.uint16))
            self._zip.writestr(f"{stem}__{i}.npy", buf.getvalue())

    def close(self) -> None:
        """Finish the archive."""
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_submission(rows: list[tuple[str, str]], path: str | os.PathLike) -> pd.DataFrame:
    """Write the submission CSV (creating the folder if needed) and return it as a DataFrame."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(path, index=False)
    return df


def validate_submission(path: str | os.PathLike, n_images: Optional[int] = None) -> str:
    """Check the file format: columns, unique ids, no empty mask, masks of one image pixel-disjoint.

    Raises ``ValueError`` with the first problem found; returns a one-line summary if the file is valid.
    """
    df = pd.read_csv(path)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError(f"unexpected columns {list(df.columns)}")
    if not df.filament_id.is_unique:
        raise ValueError("duplicate ids")
    stems = df.filament_id.str.rsplit("_", n=1).str[0]
    if n_images is not None and stems.nunique() != n_images:
        raise ValueError(f"{stems.nunique()} images, expected {n_images}")
    for s, g in df.groupby(stems):
        rles = [{"size": [H, W], "counts": r.encode("ascii")} for r in g.segmentation_rle]
        areas = mu.area(rles)
        if not (areas > 0).all():
            raise ValueError(f"empty mask in {s}")
        if mu.area(mu.merge(rles)) != areas.sum():   # disjoint iff union area = sum of areas
            raise ValueError(f"overlapping masks in {s}")
    return f"valid: {len(df)} rows, {stems.nunique()} images, ids unique, no empty masks, masks disjoint"
