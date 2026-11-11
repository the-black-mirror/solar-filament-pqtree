"""Stage 5: hurdle LightGBM model for the expected matched IoU of a candidate.

The target of a candidate is ``u``, its IoU with an annotator's polygon if that IoU exceeds 0.5, else 0, averaged over
the annotators of the image. It is modelled as ``u = h x m``:

* ``h``: share of annotators for whom the candidate is a match (cross-entropy objective, trained on all candidates),
* ``m``: mean matched IoU of the annotators that matched (regression, trained on candidates with ``h > 0``).

Both models weight a candidate by the number of annotators of its image. The prediction is
``h_hat * clip(m_hat, 0, 1)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import KEY, LGB_PARAMS, LGB_PARAMS_HIT, LGB_ROUNDS, N_FOLDS, MODEL_FEATURES

HIT_FILE = "lgbm_hit.txt"
IOU_FILE = "lgbm_iou.txt"


@dataclass
class HurdleModel:
    """The pair of boosters and the feature order they were trained with."""
    hit: lgb.Booster
    iou: lgb.Booster
    features: list[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Expected matched IoU of every row of ``df``."""
        x = df[self.features]
        return self.hit.predict(x) * np.clip(self.iou.predict(x), 0, 1)

    def save(self, folder: str | Path) -> None:
        """Write the two boosters as text files into ``folder``."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.hit.save_model(str(folder / HIT_FILE))
        self.iou.save_model(str(folder / IOU_FILE))

    @classmethod
    def load(cls, hit_path: str | Path, iou_path: str | Path, features: Sequence[str] = MODEL_FEATURES) -> "HurdleModel":
        """Read the two boosters from text files."""
        return cls(lgb.Booster(model_file=str(hit_path)), lgb.Booster(model_file=str(iou_path)), list(features))


def fit_hurdle(train: pd.DataFrame, features: Sequence[str] = MODEL_FEATURES, rounds: int = LGB_ROUNDS) -> HurdleModel:
    """Fit both boosters on ``train`` (needs the columns ``h``, ``m``, ``nann`` and the features)."""
    features = list(features)
    w = train["nann"]
    hit = lgb.train(LGB_PARAMS_HIT, lgb.Dataset(train[features], train["h"], weight=w), num_boost_round=rounds)
    pos = train[train.h > 0]
    iou = lgb.train(LGB_PARAMS, lgb.Dataset(pos[features], pos["m"], weight=pos["nann"]), num_boost_round=rounds)
    return HurdleModel(hit, iou, features)


def cross_fit(train: pd.DataFrame, features: Sequence[str] = MODEL_FEATURES, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions for every training row (5 folds, split by image; ``seed`` shuffles the images)."""
    files = np.array(sorted(train.stem.unique()))
    rng = np.random.RandomState(seed)
    grp = train.stem.map({f: i % N_FOLDS for i, f in enumerate(rng.permutation(files))}).values
    oof = np.zeros(len(train))
    for g in range(N_FOLDS):
        m = grp != g
        oof[~m] = fit_hurdle(train[m], features).predict(train[~m])
    return oof


def attach_features(cands: pd.DataFrame, cnn_scores: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Join the candidate table with the CNN scores and the detector evidence (missing evidence becomes 0)."""
    df = cands.merge(cnn_scores, on=KEY, how="left").merge(evidence, on=KEY, how="left")
    ev = [c for c in evidence.columns if c not in KEY]
    df[ev] = df[ev].fillna(0)
    return df
