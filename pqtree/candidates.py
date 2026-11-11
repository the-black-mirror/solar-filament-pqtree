"""Stage 2: candidate masks as a component tree, with shape, intensity and context features.

For every image the dense probability map is thresholded at the eight levels in ``config.LEVELS``. The 8-connected
components of each binary map are the candidates. A component at level ``l`` lies inside exactly one component at level
``l - 1``, so all candidates of an image form a forest (the "component tree"). For training images each candidate also
gets its target: the IoU with the annotators' polygons (see :func:`candidate_targets`).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Mapping, Optional

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .config import H, LEVELS, MIN_AREA, W

STATION_CODE = {"Bh": 0, "Ch": 1, "Lh": 2, "Mh": 3, "Th": 4, "Uh": 5}


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def normalise_image(img: np.ndarray) -> np.ndarray:
    """Standardise the grey values over the solar disk (pixels > 15); everything off the disk becomes 0."""
    x = img.astype(np.float32)
    m = x > 15
    if m.sum() < 1000:
        m = np.ones_like(m)
    x = (x - x[m].mean()) / (x[m].std() + 1e-6)
    x[~m] = 0
    return x


def threshold_map(prob: np.ndarray, t: float) -> np.ndarray:
    """Binary map ``prob >= round(255 t)`` of a uint8 probability map."""
    return (prob >= int(round(t * 255))).astype(np.uint8)


def label_image(prob: np.ndarray, level: int) -> np.ndarray:
    """Connected-component label image (8-connectivity) of ``prob`` at one level of ``LEVELS``."""
    return cv2.connectedComponentsWithStats(threshold_map(prob, LEVELS[level]), connectivity=8)[1]


def load_annotations(train_json: str | os.PathLike) -> tuple[dict, dict]:
    """Read the competition training annotations.

    Returns ``(by_image, ids_of_stem)``: annotation records per annotation-image id, and the annotation-image ids
    (one per annotator who labelled the image) per image stem.
    """
    with open(train_json) as f:
        data = json.load(f)
    by_image: dict = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    ids_of_stem: dict = {}
    for im in data["images"]:
        ids_of_stem.setdefault(im["file_name"].split(".")[0], []).append(im["id"])
    return by_image, ids_of_stem


def annotator_masks(by_image: Mapping, ids: list[str]) -> list[list[tuple]]:
    """Ground-truth masks per annotator as (y0, y1, x0, x1, cropped bool mask, area), cropped to the bounding box."""
    out = []
    for iid in ids:
        ms = []
        for a in by_image.get(iid, []):
            r = mu.frPyObjects(a["segmentation"], H, W)
            r = mu.merge(r) if isinstance(r, list) else r
            m = mu.decode(r).astype(bool)
            ys, xs = np.where(m)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ms.append((y0, y1, x0, x1, m[y0:y1, x0:x1].copy(), int(m.sum())))
        out.append(ms)
    return out


def candidate_targets(lab: np.ndarray, area_with_bg: np.ndarray, gts: list[list[tuple]]):
    """Targets of the candidates of one level, for one image.

    For every annotator the matched IoU of a candidate is its IoU with the best of that annotator's polygons if that IoU
    exceeds 0.5, else 0. Returns ``(u, h, m)`` for components 1..n-1 where ``u`` is the mean matched IoU over annotators,
    ``h`` the share of annotators with a match and ``m`` the mean IoU over the matching annotators (NaN if none).
    """
    n = len(area_with_bg)
    tgt_sum = np.zeros(n)
    hits = np.zeros(n)
    for gl in gts:
        best = np.zeros(n)
        for (y0, y1, x0, x1, m, ga) in gl:
            inter = np.bincount(lab[y0:y1, x0:x1][m], minlength=n).astype(np.float64)
            inter[0] = 0
            iou = inter / np.maximum(ga + area_with_bg - inter, 1)
            iou[0] = 0
            best = np.maximum(best, np.where(iou > 0.5, iou, 0))
        tgt_sum += best
        hits += best > 0
    u = tgt_sum[1:] / max(len(gts), 1)
    h = hits[1:] / max(len(gts), 1)
    m = np.where(hits[1:] > 0, tgt_sum[1:] / np.maximum(hits[1:], 1), np.nan)
    return u, h, m


# ---------------------------------------------------------------------------------------------------------------------
# one image
# ---------------------------------------------------------------------------------------------------------------------
def image_candidates(stem: str, prob: np.ndarray, img: np.ndarray, extra: Mapping[str, np.ndarray],
                     gts: Optional[list[list[tuple]]] = None) -> tuple[list[dict], list[np.ndarray]]:
    """All candidate rows of one image and the label images of the eight levels.

    ``prob`` is the uint8 probability map, ``img`` the grey H-alpha image, ``extra`` maps a prefix (``"unc"``, ``"fam"``)
    to a further uint8 map of which the mean and maximum under each candidate are stored as ``<prefix>_mean/_max``.
    ``gts`` (training images only) enables the targets ``u``, ``h``, ``m`` and ``nann``.
    """
    x = normalise_image(img)
    dark = x - cv2.blur(x, (101, 101))
    rows: list[dict] = []
    lab_levels: list[np.ndarray] = []
    prev_lab = None
    S = prob >= int(round(LEVELS[0] * 255))
    prob_s, dark_s, x_s = prob[S], dark[S], x[S]
    pm_s = prob[S].astype(np.float32)
    ex_s = {k: z[S].astype(np.float32) for k, z in extra.items()}
    pf = prob.astype(np.float32)
    pb10_s = cv2.GaussianBlur(pf, (0, 0), 10)[S]
    pb30_s = cv2.GaussianBlur(pf, (0, 0), 30)[S]
    dk31_s = (x - cv2.blur(x, (31, 31)))[S]
    dk301_s = (x - cv2.blur(x, (301, 301)))[S]
    ys_s, xs_s = np.nonzero(S)
    ys_s = ys_s.astype(np.float64)
    xs_s = xs_s.astype(np.float64)
    nann = len(gts) if gts is not None else 0
    for li, t in enumerate(LEVELS):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(threshold_map(prob, t), connectivity=8)
        lab_levels.append(lab)
        if n <= 1:
            prev_lab = lab
            continue
        idx = np.arange(1, n)
        area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        lab_s = lab[S]
        pmean = ndi.mean(prob_s, lab_s, idx)
        pmax = ndi.maximum(prob_s, lab_s, idx)
        pstd = ndi.standard_deviation(prob_s, lab_s, idx)
        dmean = ndi.mean(dark_s, lab_s, idx)
        dmin = ndi.minimum(dark_s, lab_s, idx)
        xmean = ndi.mean(x_s, lab_s, idx)
        pm0 = ndi.mean(pm_s, lab_s, idx)  # equals pmean; kept as a column of the stored tables
        exs = {}
        for k, v in ex_s.items():
            exs[k + "_mean"] = ndi.mean(v, lab_s, idx)
            exs[k + "_max"] = ndi.maximum(v, lab_s, idx)
        pb10 = ndi.mean(pb10_s, lab_s, idx)
        pb30 = ndi.mean(pb30_s, lab_s, idx)
        dk31 = ndi.mean(dk31_s, lab_s, idx)
        dk301 = ndi.mean(dk301_s, lab_s, idx)
        cc = cent[1:]
        if n > 2:
            dd, _ = cKDTree(cc).query(cc, k=2)
            nn_d = dd[:, 1]
            n150 = np.array([len(v) - 1 for v in cKDTree(cc).query_ball_point(cc, 150)])
            balls = cKDTree(cc).query_ball_point(cc, 200)
            n200 = np.array([len(v) - 1 for v in balls], float)
            arank = np.array([float((area[v] > area[i]).sum()) for i, v in enumerate(balls)])
        else:
            nn_d = np.full(n - 1, 3000.0)
            n150 = np.zeros(n - 1)
            n200 = np.zeros(n - 1)
            arank = np.zeros(n - 1)
        # second moments of the component -> elongation and axis lengths
        cnt = np.bincount(lab_s, minlength=n)[1:].astype(np.float64)
        sx = np.bincount(lab_s, xs_s, n)[1:] / np.maximum(cnt, 1)
        sy = np.bincount(lab_s, ys_s, n)[1:] / np.maximum(cnt, 1)
        vxx = np.bincount(lab_s, xs_s ** 2, n)[1:] / np.maximum(cnt, 1) - sx ** 2
        vyy = np.bincount(lab_s, ys_s ** 2, n)[1:] / np.maximum(cnt, 1) - sy ** 2
        vxy = np.bincount(lab_s, xs_s * ys_s, n)[1:] / np.maximum(cnt, 1) - sx * sy
        tr_ = vxx + vyy
        det = vxx * vyy - vxy ** 2
        disc = np.sqrt(np.maximum(tr_ ** 2 / 4 - det, 0))
        l1 = tr_ / 2 + disc + 0.1
        l2 = np.maximum(tr_ / 2 - disc, 0) + 0.1
        # parent: the label of the previous (lower-threshold) level that contains the component
        par = ndi.maximum(prev_lab[S], lab_s, idx).astype(int) if li > 0 else np.zeros(n - 1, int)
        if gts is not None:
            u, hh, mm = candidate_targets(lab, stats[:, cv2.CC_STAT_AREA], gts)
        else:
            u = np.full(n - 1, np.nan)
            hh = mm = u
        w = stats[1:, cv2.CC_STAT_WIDTH]
        h = stats[1:, cv2.CC_STAT_HEIGHT]
        r = np.hypot(cent[1:, 0] - 1024, cent[1:, 1] - 1024)
        for j in range(n - 1):
            if area[j] < MIN_AREA:
                continue
            rows.append(dict(stem=stem, level=li, t=t, cid=j + 1, parent=int(par[j]), area=area[j],
                             pmean=pmean[j], pmax=pmax[j], pstd=pstd[j], dmean=dmean[j], dmin=dmin[j], xmean=xmean[j],
                             bw=w[j], bh=h[j], fill=area[j] / (w[j] * h[j]), r=r[j], u=u[j], nann=nann,
                             elong=np.sqrt(l1[j] / l2[j]), major=4 * np.sqrt(l1[j]), minor=4 * np.sqrt(l2[j]),
                             h=hh[j], m=mm[j], cx=cent[j + 1, 0], cy=cent[j + 1, 1], n200=n200[j], arank=arank[j],
                             adx=abs(cent[j + 1, 0] - 1023.1) / 899.7,
                             mu=np.sqrt(max(0.0, 1 - ((cent[j + 1, 0] - 1023.1) ** 2 + (cent[j + 1, 1] - 1024.1) ** 2) / 899.7 ** 2)),
                             pb10=pb10[j], pb30=pb30[j], dk31=dk31[j], dk301=dk301[j], nn_d=nn_d[j], n150=float(n150[j]),
                             pm0=pm0[j], **{k: v[j] for k, v in exs.items()}))
        prev_lab = lab
    return rows, lab_levels


def add_tree_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that need the whole tree: parent area, area ratio, number of children, log area, per-image counts, station."""
    pa = df[["stem", "level", "cid", "area"]].rename(columns={"cid": "parent", "area": "parent_area"})
    pa["level"] += 1
    df = df.merge(pa, on=["stem", "level", "parent"], how="left")
    df["area_ratio"] = df.area / df.parent_area
    ch = df[df.level > 0].groupby(["stem", "level", "parent"]).size().rename("n_child").reset_index()
    ch["level"] -= 1
    df = df.merge(ch.rename(columns={"parent": "cid"}), on=["stem", "level", "cid"], how="left")
    df["n_child"] = df.n_child.fillna(0)
    df["log_area"] = np.log1p(df.area)
    cnt = df[(df.t > 0.49) & (df.t < 0.51)].groupby("stem").size()
    df["img_n50"] = df.stem.map(cnt).fillna(0).astype(float)
    df["station"] = df.stem.str[-2:].map(STATION_CODE).fillna(6)
    return df


# ---------------------------------------------------------------------------------------------------------------------
# whole split
# ---------------------------------------------------------------------------------------------------------------------
def build_candidates(prob_maps: Mapping[str, np.ndarray], read_image: Callable[[str], np.ndarray],
                     extra_maps: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
                     annotations: Optional[tuple[dict, dict]] = None, stems: Optional[list[str]] = None,
                     label_sink: Optional[Callable[[str, list[np.ndarray]], None]] = None,
                     log: Optional[Callable[[str], None]] = print) -> tuple[pd.DataFrame, dict]:
    """Candidate table of a split.

    ``prob_maps`` maps a stem to its uint8 probability map (an ``NpzFile`` works), ``read_image(stem)`` returns the
    grey image, ``extra_maps`` is ``{"unc": maps, "fam": maps}``. With ``annotations`` (from :func:`load_annotations`)
    the targets are added and the returned dict holds the number of ground-truth polygons per image (otherwise zeros).
    ``label_sink(stem, label_images)`` is called for every image, e.g. to store the label images of the test split.
    """
    extra_maps = extra_maps or {}
    stems = list(stems) if stems is not None else sorted(prob_maps.files if hasattr(prob_maps, "files") else prob_maps)
    rows: list[dict] = []
    n_gt: dict = {}
    for ii, stem in enumerate(stems):
        gts = None
        if annotations is not None:
            by_image, ids_of_stem = annotations
            gts = annotator_masks(by_image, ids_of_stem[stem])
        img = read_image(stem)
        r, labs = image_candidates(stem, prob_maps[stem], img, {k: z[stem] for k, z in extra_maps.items()}, gts)
        rows.extend(r)
        if label_sink is not None:
            label_sink(stem, labs)
        n_gt[stem] = sum(len(gl) for gl in gts) if gts is not None else 0
        if log and ii % 20 == 0:
            log(f"{ii} {stem} {len(rows)}")
    return add_tree_features(pd.DataFrame(rows)), n_gt
