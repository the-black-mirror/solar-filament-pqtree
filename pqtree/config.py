"""Constants shared by all stages of the pipeline."""
from __future__ import annotations

H = W = 2048
"""Side length of every image and mask (pixels)."""

LEVELS: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Probability thresholds that define the component tree (level index 0 is the lowest threshold)."""

MIN_AREA = 30
"""Components smaller than this many pixels are not candidates."""

TAU = 0.34
"""Selection threshold of the final model: a candidate is kept only if its predicted matched IoU exceeds it."""

DISK_RADIUS = 950
"""Radius (pixels) of the solar disk used for intensity normalisation in the dense-map stage."""

# --- feature lists (order matters: it fixes LightGBM's feature sub-sampling) -------------------------------------
SHAPE_FEATURES = ["t", "log_area", "pmean", "pmax", "pstd", "dmean", "dmin", "xmean", "bw", "bh", "fill", "r",
                  "area_ratio", "n_child"]
SHAPE_FEATURES2 = SHAPE_FEATURES + ["elong", "major", "minor", "img_n50"]
SHAPE_FEATURES3 = SHAPE_FEATURES2 + ["pb10", "pb30", "dk31", "dk301", "nn_d", "n150"]
CANDIDATE_FEATURES = SHAPE_FEATURES3 + ["n200", "arank", "adx", "mu", "station"]
"""Per-candidate descriptors computed from the map and the image (stage 2)."""

UNCERTAINTY_FEATURES = ["unc_mean", "unc_max", "fam_mean", "fam_max"]
"""Mean / max under the candidate of the view-uncertainty map and of the family-gap map (stage 1 outputs)."""

CNN_FEATURE = "cnn_u"
"""Out-of-fold prediction of the CNN scorer (stage 3)."""

FAMILIES = ("maskrcnn-1536", "yolo11m-1536", "yolo11m-2048", "yolo26m-1536")
"""Detector families (model type and input size) whose fold models supply the evidence."""
EVIDENCE_FEATURES = [f"{f.replace('-', '_')}_{s}" for f in FAMILIES for s in ("iou", "conf", "cov", "nov")] + ["iou_mean", "conf_mean"]
"""Instance-evidence features from the detector fold models (stage 4)."""

MODEL_FEATURES = CANDIDATE_FEATURES + UNCERTAINTY_FEATURES + [CNN_FEATURE] + EVIDENCE_FEATURES
"""The 52 inputs of the hurdle LightGBM model."""

KEY = ["stem", "level", "cid"]
"""Columns that identify a candidate: image, tree level, component id at that level."""

# --- LightGBM ---------------------------------------------------------------------------------------------------
LGB_ROUNDS = 600
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2)
"""Parameters of the IoU regressor. The hit model uses the same parameters with objective='cross_entropy'."""
LGB_PARAMS_HIT = dict(LGB_PARAMS, objective="cross_entropy")

N_FOLDS = 5
"""Number of cross-fitting folds (grouped by image)."""

# --- CNN scorer -------------------------------------------------------------------------------------------------
CNN_CONFIG = dict(backbone="resnet18", size=128, epochs=8, batch_size=256, lr=1e-3, folds=5, seed=0, fold_seed=7)

# --- third-party model weights on Kaggle (read-only inputs of the recompute path) ----------------------------------
UNET_DATASET = "dariushafshar/solar-filament-unet-weights"
DETECTOR_DATASET = "notoverkil/solar-filament-weights"
