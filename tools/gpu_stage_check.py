"""Check the packaged GPU stages against the stored results on six images (run it in a Kaggle GPU session).

The dense maps (stage 1), the detector instances (stage 4a) and the CNN scorer (stage 3) were produced on Kaggle T4
GPUs with the scripts that this package reorganises. GPU inference is not bit-reproducible, so the stored results are
part of the repository. This script runs the packaged code on 3 test images and 3 training images (one per fold where
possible) and compares with the stored results.

Tolerances (chosen for fp16 inference on different hardware, which can only move a few boundary pixels or scores):
  dense maps   PASS if, for every map type, mean abs difference <= 0.5 and max abs difference <= 24 (uint8 units, 1/255)
  detectors    PASS per family if the instance count is within 10 % of the stored count, >= 97 % of the stored instances with
               confidence >= 0.2 have a new instance (same family, fold and image) with IoU >= 0.9, and the confidence
               difference over these matched pairs is <= 0.05 (maximum). Unmatched stored instances and matched pairs with a larger
               confidence difference are listed in the json.
  CNN scorer   PASS if mean abs difference <= 0.01 and max abs difference <= 0.05 against the stored scores

usage (defaults are the Kaggle input paths):
  python tools/gpu_stage_check.py [--repo .] [--competition DIR] [--unet-weights DIR] [--detector-weights DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def locate(default: str, marker: str) -> str:
    """The default folder if it exists, else the first folder under /kaggle/input that contains ``marker``."""
    if os.path.isdir(default):
        return default
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    return os.path.dirname(hits[0]) if hits else default


def versions() -> dict:
    """Versions of the packages that matter for the GPU stages."""
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "timm", "segmentation_models_pytorch", "ultralytics", "numpy", "cv2", "lightgbm", "pandas"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # missing package
            out[name] = f"not importable ({type(e).__name__})"
    try:
        import torch
        out["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        out["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    return out


def choose_images(artifacts, fold_of, cand_stems, test_stems):
    """3 test images (spread over the sorted list) and 3 training images with candidates, from different folds if possible."""
    test = [test_stems[i] for i in (0, len(test_stems) // 3, 2 * len(test_stems) // 3)]
    train, used = [], set()
    for s in sorted(cand_stems):
        f = fold_of[s]
        if f not in used:
            train.append(s)
            used.add(f)
        if len(train) == 3:
            break
    return train, test


def compare_maps(new_path, old_path, stems):
    """max / mean absolute difference (uint8 units) of the maps of ``stems``."""
    a, b = np.load(new_path), np.load(old_path)
    mx, tot, n = 0, 0.0, 0
    per = {}
    for s in stems:
        d = np.abs(a[s].astype(np.int32) - b[s].astype(np.int32))
        per[s] = (int(d.max()), float(d.mean()))
        mx = max(mx, int(d.max()))
        tot += float(d.sum())
        n += d.size
    return dict(max_abs=mx, mean_abs=tot / n, per_image=per)


def check_dense(args, repo, art, train, test, work):
    from pqtree import dense_maps
    t = time.time()
    paths = dense_maps.compute_dense_maps(args.unet_weights, args.train_images, args.test_images, work / "maps",
                                          train_subset=train, test_subset=test, log=lambda m: None)
    res = {}
    for kind in ("probs", "unc", "fam"):
        res[f"oof_{kind}"] = compare_maps(paths[f"oof_{kind}_unet"], art["oof_%s_unet.npz" % kind], train)
        res[f"test_{kind}"] = compare_maps(paths[f"test_{kind}_unet"], art["test_%s_unet.npz" % kind], test)
    ok = all(v["mean_abs"] <= 0.5 and v["max_abs"] <= 24 for v in res.values())
    return ok, dict(seconds=round(time.time() - t, 1), tolerance="mean<=0.5, max<=24 (uint8 units)",
                    **{k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in res.items()}), paths


def _masks(df):
    from pycocotools import mask as mu
    return [mu.decode({"size": [2048, 2048], "counts": r.encode("ascii")}).astype(bool) for r in df.rle]


def check_detectors(args, art, train, test, work):
    import pandas as pd
    from pqtree import detector_evidence
    t = time.time()
    p_oof, p_test = detector_evidence.run_detectors(args.detector_weights, args.train_images, args.test_images, work / "detections",
                                                    oof_stems=train, test_stems=test, log=lambda m: None)
    report, ok = {}, True
    for split, new_p, old_p, stems in (("oof", p_oof, art["detector_instances_oof.csv.gz"], train),
                                       ("test", p_test, art["detector_instances_test.csv.gz"], test)):
        new = pd.read_csv(new_p)
        old = pd.read_csv(old_p)
        old = old[old.stem.isin(stems)]
        for fam in sorted(old.family.unique()):
            n_old, n_new = int((old.family == fam).sum()), int((new.family == fam).sum())
            hit = tot = 0
            dconf, unmatched, big_conf_diff = [], [], []
            for (fold, stem), go in old[(old.family == fam) & (old.conf >= 0.2)].groupby(["fold", "stem"]):
                gn = new[(new.family == fam) & (new.fold == fold) & (new.stem == stem)]
                mo, mn = _masks(go), _masks(gn)
                for i, m in enumerate(mo):
                    tot += 1
                    best, bj = 0.0, -1
                    for j, q in enumerate(mn):
                        iou = float((m & q).sum()) / max(float((m | q).sum()), 1.0)
                        if iou > best:
                            best, bj = iou, j
                    c_old = float(go.conf.iloc[i])
                    if best >= 0.9:
                        hit += 1
                        d = abs(c_old - float(gn.conf.iloc[bj]))
                        dconf.append(d)
                        if d > 0.05:
                            big_conf_diff.append(dict(image=stem, fold=int(fold), stored_conf=round(c_old, 3),
                                                      new_conf=round(float(gn.conf.iloc[bj]), 3), iou=round(best, 3)))
                    else:
                        unmatched.append(dict(image=stem, fold=int(fold), stored_conf=round(c_old, 3), best_iou=round(best, 3)))
            share = hit / tot if tot else 1.0
            good = abs(n_new - n_old) <= 0.1 * max(n_old, 1) and share >= 0.97 and (max(dconf) <= 0.05 if dconf else True)
            ok &= bool(good)
            report[f"{split}/{fam}"] = dict(
                stored_instances=n_old, new_instances=n_new, stored_conf_ge_0_2=tot,
                share_stored_conf_ge_0_2_matched_iou_ge_0_9=round(share, 3),
                matched_pair_max_conf_diff=round(max(dconf), 4) if dconf else None,
                matched_pair_mean_conf_diff=round(float(np.mean(dconf)), 4) if dconf else None,
                unmatched_stored_instances=unmatched, matched_pairs_with_conf_diff_gt_0_05=big_conf_diff, ok=bool(good))
    return ok, dict(seconds=round(time.time() - t, 1),
                    rule="per family: new count within 10 % of stored; >= 97 % of stored instances with confidence >= 0.2 have a new "
                         "instance (same family, fold, image) with IoU >= 0.9; confidence difference over these matched pairs <= 0.05 (maximum)",
                    **report)


def check_cnn(args, repo, art, train, test, maps):
    import cv2
    import pandas as pd
    from pqtree import cnn_scorer
    from pqtree.artifacts import load_table
    t = time.time()
    weights = [art[f"cnn_scorer_f{k}.pt"] for k in range(5)]
    cands_oof = load_table(art["candidates_oof.tbl.npz"])
    cands_test = load_table(art["candidates_test.tbl.npz"])
    files = np.array(sorted(cands_oof.stem.unique()))
    rng = np.random.RandomState(7)
    grp = {f: i % 5 for i, f in enumerate(rng.permutation(files))}   # same fold rule as the training of the scorer
    read = lambda folder: (lambda s: cv2.imread(f"{folder}/{s}.jpeg", cv2.IMREAD_GRAYSCALE))
    out = {}
    ok = True
    for split, stems, cands, folder, probs, score_file in (
            ("oof", train, cands_oof, args.train_images, maps["oof"], "cnn_scores_oof.csv.gz"),
            ("test", test, cands_test, args.test_images, maps["test"], "cnn_scores_test.csv.gz")):
        sub = cands[cands.stem.isin(stems)].reset_index(drop=True)
        crops, sc = cnn_scorer.extract_crops(sub, probs, read(folder))
        new = np.zeros(len(sub), np.float32)
        if split == "test":
            new = cnn_scorer.predict_with_stored_weights(weights, crops, sc)
        else:
            for k in range(5):
                m = sub.stem.map(grp).values == k
                if m.any():
                    new[m] = cnn_scorer.predict_with_stored_weights([weights[k]], crops[m], sc[m])
        old = sub[["stem", "level", "cid"]].merge(pd.read_csv(art[score_file]), on=["stem", "level", "cid"], how="left").cnn_u.values
        d = np.abs(new - old)
        good = float(d.mean()) <= 0.01 and float(d.max()) <= 0.05
        ok &= bool(good)
        out[split] = dict(candidates=int(len(sub)), max_abs=round(float(d.max()), 5), mean_abs=round(float(d.mean()), 5), ok=bool(good))
    return ok, dict(seconds=round(time.time() - t, 1), tolerance="mean<=0.01, max<=0.05", **out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--competition", default="/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--unet-weights", default="/kaggle/input/datasets/dariushafshar/solar-filament-unet-weights")
    ap.add_argument("--detector-weights", default="/kaggle/input/datasets/notoverkil/solar-filament-weights")
    ap.add_argument("--out", default="/kaggle/working")
    args = ap.parse_args()
    args.unet_weights = locate(args.unet_weights, "solar_unet_receipt.json")
    args.detector_weights = locate(args.detector_weights, "manifest.json")
    args.train_images = str(Path(args.competition) / "train" / "train_images")
    args.test_images = str(Path(args.competition) / "test" / "test_images")
    if not os.path.isdir(args.test_images):
        hits = glob.glob("/kaggle/input/**/test_images", recursive=True)
        if hits:
            args.test_images = hits[0]
            args.train_images = glob.glob("/kaggle/input/**/train_images", recursive=True)[0]
    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    import pandas as pd
    from pqtree.artifacts import fetch_artifact, load_manifest
    art_dir = repo / "artifacts"
    art = {n: fetch_artifact(n, art_dir) for n in load_manifest(art_dir)["files"] if os.path.exists(art_dir / n)}
    work = Path(args.out) / "gpu_stage_check_work"
    work.mkdir(parents=True, exist_ok=True)
    report = {"versions": versions()}
    print(json.dumps(report["versions"], indent=1))

    folds = pd.read_csv(Path(args.unet_weights) / "folds.csv")
    fold_of = dict(zip(folds.stem, folds.fold))
    test_stems = sorted(p[:-5] for p in os.listdir(args.test_images) if p.endswith(".jpeg"))
    from pqtree.artifacts import load_table
    cand_stems = set(load_table(art["candidates_oof.tbl.npz"]).stem)
    train, test = choose_images(art, fold_of, cand_stems, test_stems)
    report["images"] = dict(train=train, train_folds=[int(fold_of[s]) for s in train], test=test)
    print("images:", report["images"])

    stored_maps = {"oof": np.load(art["oof_probs_unet.npz"]), "test": np.load(art["test_probs_unet.npz"])}
    verdict = {}
    for name, fn in (("dense_maps", lambda: check_dense(args, repo, art, train, test, work)),
                     ("detectors", lambda: check_detectors(args, art, train, test, work)),
                     ("cnn_scorer", lambda: check_cnn(args, repo, art, train, test, stored_maps))):
        try:
            res = fn()
            ok, info = res[0], res[1]
        except Exception:
            ok, info = False, {"error": traceback.format_exc()}
        report[name] = info
        verdict[name] = ok
        print(f"{name}: {'PASS' if ok else 'FAIL'}", flush=True)
    report["verdict"] = {k: ("PASS" if v else "FAIL") for k, v in verdict.items()}
    out_json = Path(args.out) / "gpu_stage_check.json"
    out_json.write_text(json.dumps(report, indent=1, default=str))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
