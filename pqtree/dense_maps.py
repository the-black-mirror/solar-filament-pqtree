"""Stage 1 (GPU): dense probability maps from the public 10-model U-Net ensemble.

The weights come from the Kaggle dataset ``dariushafshar/solar-filament-unet-weights`` (CC BY-NC-SA 4.0). Five folds of
two architectures (ResNet34 and EfficientNetV2-S encoders, ``segmentation_models_pytorch.Unet``) were trained on the
competition training images only; ``folds.csv`` lists the fold in which each training image was held out.

* Training images: out-of-fold map = mean of the two models (one per architecture) of the fold that never saw the image.
* Test images: mean of all ten models.
* Each model sees the full 2048 x 2048 image, standardised over the solar disk, in eight dihedral views (4 rotations,
  with and without a horizontal flip); the view predictions are mapped back and averaged.
* Two further maps are stored per image: the standard deviation over the eight views (averaged over the models; "view
  uncertainty") and the absolute difference between the two architectures ("family gap").

All maps are stored as uint8 (probability x 255).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .artifacts import sha256_file
from .config import DISK_RADIUS, H, N_TRAIN_IMAGES, UNET_DATASET, W

# (number of 90-degree rotations, horizontal flip applied first) for the eight dihedral views, in the order used
# when the ensemble was trained and published.
D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]


def _disk_mask() -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    return ((yy - (H - 1) / 2.0) ** 2 + (xx - (W - 1) / 2.0) ** 2) <= DISK_RADIUS ** 2


def predict_image(model, img_u8: np.ndarray, device, disk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probability map and view-uncertainty map of one model for one image (both uint8, H x W)."""
    import torch

    def tf(z, k, flip):
        if flip:
            z = torch.flip(z, dims=[3])
        return torch.rot90(z, k, dims=[2, 3]) if k else z

    def itf(p, k, flip):
        if k:
            p = torch.rot90(p, -k, dims=[2, 3])
        return torch.flip(p, dims=[3]) if flip else p

    v = img_u8[disk].astype(np.float32) / 255.0
    mean, std = v.mean(), max(float(v.std()), 1e-3)
    with torch.no_grad():
        x = torch.from_numpy(img_u8).to(device).float().div_(255.0)
        z = ((x - float(mean)) / float(std))[None, None]
        acc = torch.zeros((H, W), dtype=torch.float32, device=device)
        acc2 = torch.zeros_like(acc)
        for j in range(0, 8, 4):
            chunk = D4[j:j + 4]
            inp = torch.cat([tf(z, k, fl) for k, fl in chunk]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inp)
            prob = torch.sigmoid(out.float())
            for q, (k, fl) in enumerate(chunk):
                p = itf(prob[q:q + 1], k, fl)[0, 0]
                acc += p
                acc2 += p * p
        acc /= 8
        sd = (acc2 / 8 - acc * acc).clamp_(min=0).sqrt_()
        pm = (acc * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        su = (sd * 4 * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
    return pm, su


def _round8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def compute_dense_maps(weights_dir: str | os.PathLike, train_dir: str | os.PathLike, test_dir: str | os.PathLike,
                       out_dir: str | os.PathLike, train_subset=None, test_subset=None, log=print) -> dict[str, Path]:
    """Run the ensemble and write ``{oof,test}_{probs,unc,fam}_unet.npz`` into ``out_dir``.

    ``weights_dir`` is the downloaded dataset folder (checkpoints, ``folds.csv``, ``solar_unet_receipt.json``).
    ``train_subset`` / ``test_subset`` optionally restrict the images (smoke test).
    Needs a CUDA GPU, torch and segmentation_models_pytorch 0.5.0. Models are spread over all visible GPUs.
    """
    import segmentation_models_pytorch as smp
    import torch

    wd, out = Path(weights_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    disk = _disk_mask()
    if not (wd / "solar_unet_receipt.json").exists():
        raise FileNotFoundError(f"{wd / 'solar_unet_receipt.json'} not found: the U-Net weights are not part of this repository; "
                                f"download the public Kaggle dataset {UNET_DATASET} and set PQTREE_UNET_WEIGHTS to its folder")
    receipt = json.load(open(wd / "solar_unet_receipt.json"))
    models = receipt["ensemble"]["models"]
    folds = pd.read_csv(wd / "folds.csv")
    fold_of = dict(zip(folds.stem, folds.fold))
    test_stems = sorted(p[:-5] for p in os.listdir(test_dir) if p.endswith(".jpeg"))
    train_stems = sorted(fold_of)
    if len(train_stems) != N_TRAIN_IMAGES or set(train_stems) & set(test_stems):
        raise ValueError(f"expected {N_TRAIN_IMAGES} training images in folds.csv, none of them among the test images; "
                         f"got {len(train_stems)} with {len(set(train_stems) & set(test_stems))} in the test folder")
    if train_subset is not None:
        train_stems = [s for s in train_stems if s in set(train_subset)]
    if test_subset is not None:
        test_stems = [s for s in test_stems if s in set(test_subset)]
    t0 = time.time()

    oof_fam = {"resnet34": {}, "efficientnetv2_s": {}}   # stem -> uint8 map from the one model of the family that held the stem out
    oof_sd: dict = {}                                    # stem -> uint16 sum of the (halved) uncertainty maps over the two models
    te_fam = {"resnet34": {}, "efficientnetv2_s": {}}    # stem -> uint16 sum over the 5 folds
    te_sd: dict = {}
    lock = threading.Lock()
    todo = list(models)
    errors: list = []

    def run_models(device):
        torch.backends.cudnn.benchmark = True
        while True:
            with lock:
                if not todo or errors:
                    return
                m = todo.pop(0)
            try:
                if sha256_file(wd / m["file"]) != m["sha256"]:
                    raise RuntimeError(f"checksum mismatch for {m['file']}")
                model = smp.Unet(encoder_name=m["encoder"], encoder_weights=None, in_channels=1, classes=1)
                model.load_state_dict(torch.load(wd / m["file"], map_location="cpu", weights_only=True))
                model = model.to(device).to(memory_format=torch.channels_last).eval()
                fam, k = m["family"], m["fold"]
                for s in [s for s in train_stems if fold_of[s] == k]:
                    pm, su = predict_image(model, cv2.imread(f"{train_dir}/{s}.jpeg", cv2.IMREAD_GRAYSCALE), device, disk)
                    with lock:
                        oof_fam[fam][s] = pm
                        oof_sd[s] = (oof_sd[s] + (su >> 1)) if s in oof_sd else (su >> 1)
                for s in test_stems:
                    pm, su = predict_image(model, cv2.imread(f"{test_dir}/{s}.jpeg", cv2.IMREAD_GRAYSCALE), device, disk)
                    with lock:
                        te_fam[fam][s] = te_fam[fam].get(s, 0) + pm.astype(np.uint16)
                        te_sd[s] = te_sd.get(s, 0) + su.astype(np.uint16)
                del model
                torch.cuda.empty_cache()
                log(f"[{time.time() - t0:7.1f}s] {m['file']} on {device} done")
            except Exception as e:  # report after all threads stopped
                with lock:
                    errors.append(f"{m['file']}: {e!r}")

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if not devices:
        raise RuntimeError("compute_dense_maps needs a CUDA GPU")
    threads = [threading.Thread(target=run_models, args=(d,)) for d in devices]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise RuntimeError("; ".join(errors))

    oof_p, oof_u, oof_g = {}, {}, {}
    for s in train_stems:
        a = oof_fam["resnet34"][s].astype(np.float32)
        b = oof_fam["efficientnetv2_s"][s].astype(np.float32)
        oof_p[s] = _round8((a + b) / 2)
        oof_u[s] = oof_sd.pop(s)
        oof_g[s] = _round8(np.abs(a - b))
    te_p, te_u, te_g = {}, {}, {}
    for s in test_stems:
        a = te_fam["resnet34"].pop(s) / 5.0
        b = te_fam["efficientnetv2_s"].pop(s) / 5.0
        te_p[s] = _round8((a + b) / 2)
        te_u[s] = _round8(te_sd.pop(s) / 10.0)
        te_g[s] = _round8(np.abs(a - b))
    paths = {}
    for name, d in [("oof_probs_unet", oof_p), ("oof_unc_unet", oof_u), ("oof_fam_unet", oof_g),
                    ("test_probs_unet", te_p), ("test_unc_unet", te_u), ("test_fam_unet", te_g)]:
        paths[name] = out / f"{name}.npz"
        np.savez_compressed(paths[name], **d)
    log(f"[{time.time() - t0:7.1f}s] done")
    return paths
