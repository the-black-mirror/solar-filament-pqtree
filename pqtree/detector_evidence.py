"""Stage 4: instance evidence from the public detector fold models.

Weights: Kaggle dataset ``notoverkil/solar-filament-weights`` (AGPL-3.0). Four model families, five fold models each,
all trained on the competition training images only: YOLO11m-seg at 1536 and at 2048 px, YOLO26m-seg at 1536 px
(Ultralytics) and Mask R-CNN ResNet50-FPN v2 at 1536 px (torchvision).

* :func:`run_detectors` (GPU) writes every detected instance (confidence >= 0.05) as a COCO run-length mask. Training
  images are predicted only by the fold model that did not see them; test images by all five fold models.
* :func:`evidence_features` (CPU) turns the instance tables into per-candidate features.
"""
from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

from .config import FAMILIES, H, LEVELS, W

CONF = 0.05
MAX_DET = 80


def _rle_counts(mask_bool: np.ndarray) -> str:
    """COCO run-length counts of a boolean mask."""
    return mu.encode(np.asfortranarray(mask_bool.astype(np.uint8)))["counts"].decode("ascii")


# ---------------------------------------------------------------------------------------------------------------------
# GPU part
# ---------------------------------------------------------------------------------------------------------------------
def _load_yolo(path: str, scratch: Path):
    """Open a notoverkil YOLO checkpoint.

    The files hold ``{'ema': state_dict, 'config': ..., 'architecture': model yaml as dict}``, which the safe loader of
    Ultralytics refuses. The model is rebuilt from the architecture, the EMA weights are loaded strictly and the module
    is re-saved as a plain Ultralytics checkpoint in ``scratch``. If Ultralytics cannot open that file, the module is run
    through a ``SegmentationPredictor`` instead (same predictions).

    The checkpoints are unpickled with ``weights_only=False``; only use files from the trusted dataset.
    """
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import DEFAULT_CFG_DICT

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and isinstance(ck.get("architecture"), dict):
        arch = ck["architecture"]
        m = SegmentationModel(cfg=arch, ch=3, nc=int(arch.get("nc", 1)), verbose=False)
        m.load_state_dict(ck["ema"], strict=True)
        m.names = {0: "filament"}
        m.args = {**DEFAULT_CFG_DICT, "task": "segment"}
        m = m.float().eval()
        tmp = Path(scratch) / "_clean_yolo.pt"
        torch.save({"model": m, "train_args": {"task": "segment"}}, tmp)
        try:
            return YOLO(str(tmp))
        except Exception:
            from ultralytics.models.yolo.segment import SegmentationPredictor

            class _Predictor:
                """Minimal stand-in with the ``predict`` signature used by :func:`_predict_yolo`."""

                def __init__(self, module):
                    self.module = module
                    self.p = None

                def predict(self, img, imgsz, **kw):
                    if self.p is None:
                        self.p = SegmentationPredictor(overrides=dict(imgsz=imgsz, conf=CONF, iou=0.7, max_det=MAX_DET,
                                                                      retina_masks=True, device=0, half=True, verbose=False,
                                                                      save=False, task="segment"))
                        self.p.setup_model(model=self.module)
                    return self.p(img)

            return _Predictor(m)
    return YOLO(path)


def _predict_yolo(model, img_gray: np.ndarray, imgsz: int) -> list[tuple]:
    """(confidence, area, RLE) of every instance a YOLO model finds in one image."""
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    r = model.predict(rgb, imgsz=imgsz, conf=CONF, iou=0.7, max_det=MAX_DET, retina_masks=True, device=0, half=True, verbose=False)[0]
    out = []
    if r.masks is None:
        return out
    md = r.masks.data
    conf = r.boxes.conf.float().cpu().numpy()
    for i in range(md.shape[0]):
        m = (md[i] > 0.5).cpu().numpy()
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        a = int(m.sum())
        if a < 20:
            continue
        out.append((float(conf[i]), a, _rle_counts(m)))
    return out


def _load_maskrcnn(path: str, device):
    """Open a notoverkil Mask R-CNN checkpoint as a torchvision ``maskrcnn_resnet50_fpn_v2`` (two classes), strict key check."""
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = None
    if isinstance(ck, dict):
        for key in ("ema", "model", "state_dict", "model_state_dict"):
            if key in ck and isinstance(ck[key], dict) and all(hasattr(v, "shape") for v in list(ck[key].values())[:3]):
                sd = ck[key]
                break
        if sd is None and all(hasattr(v, "shape") for v in list(ck.values())[:3]):
            sd = ck
    if sd is None and hasattr(ck, "state_dict"):
        sd = ck.state_dict()
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    mdl = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=1536, max_size=1536,
                                   box_detections_per_img=MAX_DET, box_score_thresh=CONF)
    mdl.roi_heads.box_predictor = FastRCNNPredictor(mdl.roi_heads.box_predictor.cls_score.in_features, 2)
    mdl.roi_heads.mask_predictor = MaskRCNNPredictor(mdl.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    res = mdl.load_state_dict(sd, strict=False)
    assert len(res.missing_keys) == 0, "state dict does not match torchvision maskrcnn_resnet50_fpn_v2"
    return mdl.to(device).eval()


def _predict_maskrcnn(model, img_gray: np.ndarray, device) -> list[tuple]:
    """(confidence, area, RLE) of every instance the Mask R-CNN finds in one image."""
    import torch

    with torch.no_grad():
        x = torch.from_numpy(img_gray).to(device).float().div_(255.0)[None].repeat(3, 1, 1)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = model([x])[0]
    out = []
    sc = o["scores"].float().cpu().numpy()
    ms = o["masks"]
    for i in range(len(sc)):
        m = (ms[i, 0] > 0.5).cpu().numpy()
        a = int(m.sum())
        if a < 20:
            continue
        out.append((float(sc[i]), a, _rle_counts(m)))
    return out


def run_detectors(weights_dir: str | os.PathLike, train_dir: str | os.PathLike, test_dir: str | os.PathLike,
                  out_dir: str | os.PathLike, families=FAMILIES, oof_stems=None, test_stems=None, log=print) -> tuple[Path, Path]:
    """Detect instances with all fold models and write ``detector_instances_oof.csv.gz`` and ``detector_instances_test.csv.gz``.

    ``weights_dir`` is the downloaded dataset folder (``manifest.json``, ``splits/folds.csv``, one folder per model run).
    ``oof_stems`` / ``test_stems`` optionally restrict the images (smoke test; an out-of-fold stem is still predicted
    only by the fold model that did not see it). Needs a CUDA GPU.
    """
    import torch

    device = torch.device("cuda")
    wd, out = Path(weights_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(wd / "manifest.json"))
    folds = pd.read_csv(wd / "splits" / "folds.csv")
    fold_of = dict(zip(folds.image_stem, folds.fold))
    all_test = sorted(p[:-5] for p in os.listdir(test_dir) if p.endswith(".jpeg"))
    train_stems = sorted(fold_of)
    assert len(train_stems) == 707 and not (set(train_stems) & set(all_test))
    test_use = list(test_stems) if test_stems is not None else all_test
    oof_use = set(oof_stems) if oof_stems is not None else set(train_stems)
    runs = {(m["family"], m["fold"]): m["run"] for m in man["models"]}
    rows_oof, rows_test = [], []
    cols = ["family", "fold", "stem", "conf", "area", "rle"]
    t0 = time.time()
    scratch = Path(tempfile.mkdtemp())
    for fam in families:
        imgsz = 2048 if fam.endswith("2048") else 1536
        for k in range(5):
            path = f"{wd}/{runs[(fam, k)]}/last.pt"
            if fam.startswith("maskrcnn"):
                model = _load_maskrcnn(path, device)
                pred = lambda im: _predict_maskrcnn(model, im, device)
            else:
                model = _load_yolo(path, scratch)
                pred = lambda im: _predict_yolo(model, im, imgsz)
            oof = [s for s in train_stems if fold_of[s] == k and s in oof_use]
            te = test_use
            for s in oof:
                for c, a, r in pred(cv2.imread(f"{train_dir}/{s}.jpeg", cv2.IMREAD_GRAYSCALE)):
                    rows_oof.append((fam, k, s, c, a, r))
            for s in te:
                for c, a, r in pred(cv2.imread(f"{test_dir}/{s}.jpeg", cv2.IMREAD_GRAYSCALE)):
                    rows_test.append((fam, k, s, c, a, r))
            log(f"[{time.time() - t0:7.1f}s] {fam} fold {k}: {len(rows_oof)} / {len(rows_test)} instances so far")
            del model
            gc.collect()
            torch.cuda.empty_cache()
    p_oof, p_test = out / "detector_instances_oof.csv.gz", out / "detector_instances_test.csv.gz"
    pd.DataFrame(rows_oof, columns=cols).to_csv(p_oof, index=False)
    pd.DataFrame(rows_test, columns=cols).to_csv(p_test, index=False)
    return p_oof, p_test


# ---------------------------------------------------------------------------------------------------------------------
# CPU part
# ---------------------------------------------------------------------------------------------------------------------
def _decode(rle: str) -> np.ndarray:
    """Boolean mask of a COCO run-length string."""
    return mu.decode({"size": [H, W], "counts": rle.encode("ascii")}).astype(bool)


def evidence_features(instances: pd.DataFrame, cands: pd.DataFrame, prob_maps: Mapping[str, np.ndarray],
                      stems: Optional[list[str]] = None, log=print) -> pd.DataFrame:
    """Per-candidate agreement with the detector instances.

    For every candidate (stem, level, cid) and every detector family ``f`` and fold model: ``f_iou`` is the best IoU with
    any instance, ``f_conf`` the confidence of that instance, ``f_cov`` the share of the candidate covered by instances
    with confidence >= 0.3 and ``f_nov`` the number of instances with IoU > 0.1. The values are averaged over the fold
    models that predicted the image (one for training images, five for test images). ``iou_mean`` and ``conf_mean`` are
    the means over the four families.

    ``cands`` needs the columns stem, level, cid, area; the label images are recomputed from ``prob_maps``.
    """
    inst = instances.copy()
    inst["fold"] = inst.fold.astype(str)
    cand = cands[["stem", "level", "cid", "area"]]
    fams = sorted(inst.family.unique())
    by_stem = {s: g for s, g in inst.groupby("stem")}
    out = []
    t0 = time.time()
    for ii, (stem, cg) in enumerate(cand.groupby("stem")):
        if stems is not None and stem not in stems:
            continue
        prob = prob_maps[stem]
        labs, areas = [], []
        for t in LEVELS:
            n, lab, stats, _ = cv2.connectedComponentsWithStats((prob >= int(round(t * 255))).astype(np.uint8), connectivity=8)
            labs.append(lab)
            areas.append(stats[:, cv2.CC_STAT_AREA].astype(np.float64))
        g = by_stem.get(stem)
        per_fam_fold = {}
        if g is not None:
            for (fam, fold), gg in g.groupby(["family", "fold"]):
                masks = []  # (conf, y0, y1, x0, x1, cropped mask, area)
                cov_union = np.zeros((H, W), bool)
                for c, r in zip(gg.conf, gg.rle):
                    m = _decode(r)
                    ys, xs = np.where(m)
                    if len(ys) == 0:
                        continue
                    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    masks.append((float(c), y0, y1, x0, x1, m[y0:y1, x0:x1].copy(), len(ys)))
                    if c >= 0.3:
                        cov_union[y0:y1, x0:x1] |= m[y0:y1, x0:x1]
                    del m, ys, xs
                res = {}
                for li in range(len(LEVELS)):
                    lab = labs[li]
                    n = len(areas[li])
                    best_iou = np.zeros(n)
                    best_conf = np.zeros(n)
                    nov = np.zeros(n)
                    for c, y0, y1, x0, x1, mc, ma in masks:
                        inter = np.bincount(lab[y0:y1, x0:x1][mc], minlength=n).astype(np.float64)
                        inter[0] = 0
                        iou = inter / np.maximum(areas[li] + ma - inter, 1)
                        upd = iou > best_iou
                        best_conf = np.where(upd, c, best_conf)
                        best_iou = np.maximum(best_iou, iou)
                        nov += iou > 0.1
                    cov = np.bincount(lab[cov_union], minlength=n).astype(np.float64) / np.maximum(areas[li], 1)
                    res[li] = (best_iou, best_conf, cov, nov)
                per_fam_fold[(fam, fold)] = res
        for _, r in cg.iterrows():
            row = {"stem": stem, "level": int(r.level), "cid": int(r.cid)}
            for fam in fams:
                vals = [v[int(r.level)] for (f_, k_), v in per_fam_fold.items() if f_ == fam]
                if vals:
                    iou = np.mean([v[0][int(r.cid)] for v in vals])
                    conf = np.mean([v[1][int(r.cid)] for v in vals])
                    cov = np.mean([v[2][int(r.cid)] for v in vals])
                    nov = np.mean([v[3][int(r.cid)] for v in vals])
                else:
                    iou = conf = cov = nov = 0.0
                short = fam.replace("-", "_")
                row[f"{short}_iou"] = iou
                row[f"{short}_conf"] = conf
                row[f"{short}_cov"] = cov
                row[f"{short}_nov"] = nov
            row["iou_mean"] = np.mean([row[f"{f.replace('-', '_')}_iou"] for f in fams])
            row["conf_mean"] = np.mean([row[f"{f.replace('-', '_')}_conf"] for f in fams])
            out.append(row)
        if ii % 50 == 0:
            log(f"[{time.time() - t0:6.0f}s] {ii} {stem} {len(out)}")
    return pd.DataFrame(out)
