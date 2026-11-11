"""Stage 3 (GPU for training): CNN candidate scorer.

A ResNet18 (timm, 4 input channels) looks at a square crop around a candidate and predicts the expected matched IoU
``u`` of the candidate. The four channels are the standardised image, the probability map, the candidate mask and the
mask ``prob >= 0.2``; three scalars (log crop side, tree level threshold, log area) are concatenated to the pooled
features. Training is 5-fold cross-fitted by image on the candidates of the training images; the test prediction is the
mean of the five fold models. Only the training annotations enter through ``u``.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np
import pandas as pd

from .config import CNN_CONFIG, LEVELS


def normalise_to_u8(img: np.ndarray) -> np.ndarray:
    """Disk-standardised image mapped to uint8 (mean 128, 40 grey levels per standard deviation)."""
    x = img.astype(np.float32)
    m = x > 15
    if m.sum() < 1000:
        m = np.ones_like(m)
    x = (x - x[m].mean()) / (x[m].std() + 1e-6)
    x[~m] = 0
    return np.clip(x * 40 + 128, 0, 255).astype(np.uint8)


def extract_crops(cands: pd.DataFrame, prob_maps: Mapping[str, np.ndarray], read_image: Callable[[str], np.ndarray],
                  size: int = CNN_CONFIG["size"]) -> tuple[np.ndarray, np.ndarray]:
    """uint8 crops ``[N, 4, size, size]`` and scalars ``[N, 3]`` aligned with the rows of ``cands``.

    ``cands`` needs the columns stem, level, cid, area. The crop side is 1.6 x the larger bounding-box side (at least
    64 px); the window is zero-padded outside the image and resized with area interpolation.
    """
    S = size
    df = cands.reset_index(drop=True)
    crops = np.zeros((len(df), 4, S, S), np.uint8)
    sc = np.zeros((len(df), 3), np.float32)
    for stem, idx in df.groupby("stem").groups.items():
        prob = prob_maps[stem]
        img = normalise_to_u8(read_image(stem))
        low = (prob >= 51).astype(np.uint8) * 255
        labs, stats = {}, {}
        sub = df.loc[idx]
        for lv in sorted(sub.level.unique()):
            n, lab, st, _ = cv2.connectedComponentsWithStats((prob >= int(round(LEVELS[lv] * 255))).astype(np.uint8), connectivity=8)
            labs[lv], stats[lv] = lab, st
        for i, row in zip(idx, sub.itertuples()):
            st = stats[row.level][row.cid]
            x0, y0, w, h = st[0], st[1], st[2], st[3]
            side = int(max(64, 1.6 * max(w, h)))
            cx, cy = x0 + w / 2, y0 + h / 2
            a0, b0 = int(cx - side / 2), int(cy - side / 2)

            def win(arr):
                out = np.zeros((side, side), np.uint8)
                sx0, sy0 = max(a0, 0), max(b0, 0)
                sx1, sy1 = min(a0 + side, 2048), min(b0 + side, 2048)
                if sx1 > sx0 and sy1 > sy0:
                    out[sy0 - b0:sy1 - b0, sx0 - a0:sx1 - a0] = arr[sy0:sy1, sx0:sx1]
                return cv2.resize(out, (S, S), interpolation=cv2.INTER_AREA)

            cm = labs[row.level][max(b0, 0):min(b0 + side, 2048), max(a0, 0):min(a0 + side, 2048)] == row.cid
            full = np.zeros((side, side), np.uint8)
            sx0, sy0 = max(a0, 0), max(b0, 0)
            full[sy0 - b0:sy0 - b0 + cm.shape[0], sx0 - a0:sx0 - a0 + cm.shape[1]] = cm * 255
            crops[i] = np.stack([win(img), win(prob), cv2.resize(full, (S, S), interpolation=cv2.INTER_AREA), win(low)])
            sc[i] = [math.log(side) / 8, LEVELS[row.level], math.log1p(row.area) / 10]
    return crops, sc


def make_model(pretrained: bool = False):
    """The scorer network. ``pretrained=True`` starts from the ImageNet weights of the timm ResNet18 (needs internet)."""
    import timm
    import torch
    import torch.nn as nn

    class ScorerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = timm.create_model(CNN_CONFIG["backbone"], pretrained=pretrained, in_chans=4, num_classes=0)
            self.h = nn.Sequential(nn.Linear(self.b.num_features + 3, 256), nn.ReLU(), nn.Linear(256, 1))

        def forward(self, x, s):
            return self.h(torch.cat([self.b(x), s], 1))[:, 0]

    return ScorerNet()


def predict_scores(model, crops: np.ndarray, scalars: np.ndarray, device, tta: bool = True) -> np.ndarray:
    """Sigmoid output of the scorer, averaged over four test-time views (identity, two flips, one rotation)."""
    import torch

    model.eval()
    out = []
    use_amp = getattr(device, "type", str(device)) == "cuda"
    with torch.no_grad():
        for i in range(0, len(crops), 512):
            x = torch.from_numpy(crops[i:i + 512]).to(device).float() / 255
            s = torch.from_numpy(scalars[i:i + 512]).to(device)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp):
                p = torch.sigmoid(model(x, s).float())
                if tta:
                    p = p + torch.sigmoid(model(x.flip(3), s).float()) + torch.sigmoid(model(x.flip(2), s).float()) \
                        + torch.sigmoid(model(torch.rot90(x, 1, (2, 3)), s).float())
                    p = p / 4
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def train_cross_fit(train_cands: pd.DataFrame, train_crops: np.ndarray, train_scalars: np.ndarray,
                    test_crops: np.ndarray, test_scalars: np.ndarray, weights_dir: str | Path | None = None,
                    log=print) -> tuple[np.ndarray, np.ndarray]:
    """5-fold cross-fit by image. Returns (out-of-fold predictions for the training rows, mean test predictions).

    The five fold checkpoints are written to ``weights_dir`` as ``cnn_scorer_f{k}.pt`` if given. Needs a CUDA GPU.
    """
    import torch
    import torch.nn.functional as F

    cfg = CNN_CONFIG
    dev = torch.device("cuda")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tr = train_cands.reset_index(drop=True)
    files = np.array(sorted(tr.stem.unique()))
    rng = np.random.RandomState(cfg["fold_seed"])
    grp = {f: i % cfg["folds"] for i, f in enumerate(rng.permutation(files))}
    g = tr.stem.map(grp).values
    y = tr.u.values.astype(np.float32)
    oof = np.zeros(len(tr), np.float32)
    test_pred = np.zeros(len(test_crops), np.float32)

    def aug(x):
        k = random.randrange(4)
        x = torch.rot90(x, k, (2, 3))
        if random.random() < 0.5:
            x = x.flip(3)
        return x

    bs = cfg["batch_size"]
    for f in range(cfg["folds"]):
        tri = np.flatnonzero(g != f)
        vai = np.flatnonzero(g == f)
        model = make_model(pretrained=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        steps = cfg["epochs"] * (len(tri) // bs)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        for ep in range(cfg["epochs"]):
            perm = np.random.permutation(tri)
            tot = 0.0
            for i in range(0, len(perm) - bs + 1, bs):
                b = np.sort(perm[i:i + bs])
                x = aug(torch.from_numpy(train_crops[b]).to(dev).float() / 255)
                s = torch.from_numpy(train_scalars[b]).to(dev)
                t = torch.from_numpy(y[b]).to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    logit = model(x, s)
                loss = F.binary_cross_entropy_with_logits(logit.float(), t)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sch.step()
                tot += loss.item()
            model.train()
            log(f"fold {f} epoch {ep} loss {tot / max(1, len(perm) // bs):.4f}")
        oof[vai] = predict_scores(model, train_crops[vai], train_scalars[vai], dev)
        log(f"fold {f} correlation {np.corrcoef(oof[vai], y[vai])[0, 1]:.4f}")
        test_pred += predict_scores(model, test_crops, test_scalars, dev) / cfg["folds"]
        if weights_dir is not None:
            Path(weights_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(weights_dir) / f"cnn_scorer_f{f}.pt")
    return oof, test_pred


def predict_with_stored_weights(weight_files: list[str | Path], crops: np.ndarray, scalars: np.ndarray, device=None) -> np.ndarray:
    """Mean prediction of the five stored fold models for new crops (the test-set prediction of the pipeline)."""
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = np.zeros(len(crops), np.float32)
    for wf in weight_files:
        model = make_model(pretrained=False)
        model.load_state_dict(torch.load(wf, map_location="cpu", weights_only=True))
        total += predict_scores(model.to(device), crops, scalars, device) / len(weight_files)
    return total
