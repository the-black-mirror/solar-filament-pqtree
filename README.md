# Solar filament instance segmentation: selection on a component tree

Code of a solution to the *Solar Filament Segmentation Challenge 2026* (MAGFiLO, GONG H-alpha, 2048 x 2048 images,
metric: pooled multi-annotator Panoptic Quality). Author: Yuvraj Sharma.

**Method in short.** Pre-trained dense networks give a probability map per image. Thresholding the map at eight levels
and taking connected components gives candidate masks that form a tree. A LightGBM model predicts for every candidate its
expected matched IoU with the annotators, using shape and intensity descriptors, the output of a small CNN that looks at
the candidate, and the agreement of the candidate with instances found by detector networks. A dynamic programme then
selects the set of non-overlapping candidates that maximises the sum of (predicted IoU - tau) (tau = 0.30). The result is
one run-length-encoded mask per kept candidate. The component tree, the hurdle estimate of the matched IoU, the exact
dynamic programme and the use of a single global threshold follow the public notebook of tushhaaarrrr (see
`THIRD_PARTY.md`); the candidate descriptors, the CNN scorer, the detector-agreement features and this code were written for this project.

The reference submission is `reference/submission.csv` (1093 instances in 180 test images). One command or one notebook
reproduces it exactly, without GPU and without retraining.

## Quick start (CPU only, fresh clone)

The repository contains everything the notebook loads, including the model weights and the stored stage outputs
(`artifacts/`, about 360 MB, plain git files, no download step). The clone is large because the five CNN scorer weight
files (227 MB) and the stored maps and component images (96 MB) are part of it: the host requires weights to be stored in the repository.

```bash
pip install -r requirements.txt                  # exact versions, Python 3.11
python -m pqtree.artifacts                        # optional: verify every artefact against artifacts/MANIFEST.json
python -m pqtree.pipeline outputs/submission.csv  # writes the CSV (about 1 minute) and prints its sha256
python -m pytest tests                            # unit tests and the reproduction test
jupyter nbconvert --to notebook --execute notebooks/pipeline.ipynb --output-dir outputs   # the full walk-through
```

The sha256 of the expected CSV is `0d277cfceb562591a8562a163bdb026007da084de9d5b77fea3023353adaef29`.

`pqtree.artifacts.fetch_artifact(name)` returns the path of a file in `artifacts/` (or in `$PQTREE_ARTIFACTS`) after
checking its sha256 against the manifest. If a file is missing it raises an error that asks for the complete repository.

Optional inputs of the notebook (environment variables): the competition folder (`PQTREE_COMPETITION_DIR`, with
`train/train_images` and `test/test_images`) and the training annotation file (`PQTREE_TRAIN_JSON`). Without them the
pictures and the exact out-of-fold evaluation of stage 7 are skipped; everything else runs. The repository contains no
competition image files and no annotation file (see "Data, labels and licences").

## Pipeline

| stage | module | what it does | input | output (artefact) | switch in the notebook |
|---|---|---|---|---|---|
| 1 | `pqtree/dense_maps.py` | 10 U-Nets x 8 dihedral views at 2048 px; out-of-fold maps for training images, 10-model mean for test images; view-uncertainty and family-gap maps | images, U-Net weights | `{oof,test}_{probs,unc,fam}_unet.npz` | `RECOMPUTE_MAPS` |
| 2 | `pqtree/candidates.py` | components at 8 thresholds, component tree, 33 descriptors, targets from the training annotations | maps, images, annotations | `candidates_{oof,test}.tbl.npz`, `test_components.npz`, `ngt_train.json` | `RECOMPUTE_CANDIDATES` |
| 3 | `pqtree/cnn_scorer.py` | ResNet18 on 4-channel crops, 5-fold cross-fitted, predicts the matched IoU | candidates, maps, images | `cnn_scores_{oof,test}.csv.gz`, `cnn_scorer_f{0-4}.pt` | `RETRAIN_CNN_SCORER`, `APPLY_STORED_CNN_WEIGHTS` |
| 4a | `pqtree/detector_evidence.py` | instances of 4 detector families x 5 fold models | images, detector weights | `detector_instances_{oof,test}.csv.gz` | `RECOMPUTE_DETECTIONS` |
| 4b | `pqtree/detector_evidence.py` | per-candidate IoU / confidence / coverage / count w.r.t. the instances | instances, candidates | `evidence_{oof,test}.csv.gz` | `RECOMPUTE_EVIDENCE` |
| 5 | `pqtree/lgbm_scorer.py` | hurdle LightGBM (hit probability x matched IoU), 52 inputs; cross-fitted scores for evaluation | all of the above | `lgbm_hit.txt`, `lgbm_iou.txt`, `oof_scores.csv.gz` | `REFIT_LGBM` |
| 6 | `pqtree/selection.py` | antichain dynamic programme, run-length encoding, validation | scores, component images | `submission.csv` | - |
| 7 | `pqtree/evaluation.py`, `pqtree/visualize.py` | PQ, IoU and Dice distributions, one-to-many / many-to-one relations on the out-of-fold predictions; pictures | selection, annotations | figures, tables | - |

In file names, `oof` means out-of-fold (training images scored by models that did not see them), `unet` the dense maps of
the public U-Net ensemble of stage 1. Setting a switch to `True` runs the stage code and uses its output for the
following stages instead of the stored artefact; the notebook refuses inconsistent combinations. Stage outputs are written
to `outputs/`, never over the artefacts. `APPLY_STORED_CNN_WEIGHTS` needs torch and timm (`requirements-gpu.txt`) and the
test images.

### GPU stages

Stages 1, 3 (training) and 4a were produced on Kaggle T4 GPUs with the scripts that this package reorganises. Their
results are stored in `artifacts/` because GPU inference is not bit-reproducible. `tools/gpu_stage_check.py` runs the
packaged stages on six images in a Kaggle GPU session and compares them with the stored results (maximum and mean
differences of the maps, instance-level agreement of the detections, difference of the CNN scores; PASS / FAIL per stage
with the tolerances stated in the script). The CPU stages were checked by exact reproduction of the submission; the
result of the GPU check is in the next section.
`run_detectors` loads third-party checkpoints with `torch.load(weights_only=False)`: use only the files of the public dataset.

### GPU stage check

`tools/gpu_stage_check.py` was run twice in a Kaggle GPU session (Tesla T4; Python 3.13.16, torch 2.11.0+cu130, torchvision
0.26.0+cu130, timm 1.0.30, segmentation_models_pytorch 0.5.0, ultralytics 8.4.121, numpy 2.1.3, OpenCV 4.14.0, pandas 2.3.3,
LightGBM 4.6.0), with the same outcome both times. The second run used the current version of the script and is stored in
full as `tools/gpu_stage_check_result.json`. It re-ran the packaged stages on 3 test images and 3 training images (folds
2, 3, 4) and compared them with the stored results.

| stage | measured (second run) | verdict |
|---|---|---|
| dense maps (6 map types, uint8 units) | max abs difference 1 to 10 grey levels, mean 4e-5 to 2.5e-4 | PASS |
| CNN scorer (stored weights, 164 training and 205 test candidates) | max abs difference 0.00035 (training images) and 0.00007 (test images); mean 8e-5 and 2e-5 | PASS |
| detectors, 7 of 8 family/split rows (training images: Mask R-CNN, YOLO11m 1536, YOLO11m 2048, YOLO26m; test images: YOLO11m 1536, YOLO11m 2048, YOLO26m) | 100 % of the stored instances with confidence >= 0.2 have a new instance at IoU >= 0.9; confidence differences of these pairs <= 0.0054; instance counts equal, except 37 stored against 38 new (YOLO11m 1536, training images) | PASS |
| detectors, test images, Mask R-CNN 1536 | 162 stored and 162 new instances; 85 of 86 stored instances with confidence >= 0.2 matched (98.8 %); largest confidence difference of a matched pair 0.4339, mean 0.0053 | **FAIL (outside the rule)** |

The detector rule: per family and split, the new instance count is within 10 % of the stored count, at least 97 % of the
stored instances with confidence >= 0.2 have a new instance at IoU >= 0.9 (same family, fold and image), and the maximum
confidence difference over these matched pairs is <= 0.05. The test Mask R-CNN row is outside the rule because of two
instances in image 20130609133454Bh: a matched pair (fold 1, IoU 0.911) whose confidence is 0.201 in the stored run and 0.635
in the new run, and a stored instance (fold 2, confidence 0.831) whose best new instance has IoU 0.826. The rule was not
changed to accept the row. The stored detections are the ones used by the pipeline; this stage only feeds the evidence
features of stage 4b.

### Third-party models used by the recompute path

Stages 1 and 4a read public Kaggle datasets (not copied into this repository, see `THIRD_PARTY.md`):

* `dariushafshar/solar-filament-unet-weights` (CC BY-NC-SA 4.0): ten U-Net fold models; set `PQTREE_UNET_WEIGHTS` to the
  folder containing the checkpoints, `folds.csv` and `solar_unet_receipt.json`.
* `notoverkil/solar-filament-weights` (AGPL-3.0): detector fold models; set `PQTREE_DETECTOR_WEIGHTS` to the dataset folder
  (`manifest.json`, `splits/folds.csv`, one folder per model run).

Download both datasets into `third_party_weights/solar-filament-unet-weights` and `third_party_weights/solar-filament-weights`
(the defaults of the notebook) or point the variables at other folders. Both were trained on the competition training images
only, with published fold assignments. Training images are processed only by the models that did not see them, so the
training-set features used by the LightGBM model are out-of-fold.

### Choice of the threshold

With pooled PQ = S / (TP + FP/2 + FN/2), every kept instance adds exactly 1/2 per annotator to the denominator, matched or
not. By Dinkelbach's argument for ratio objectives, with exact expected IoUs the optimal threshold equals half the
attainable PQ (about 0.23 here), and the best disjoint set is found exactly by dynamic programming over the tree
(`pqtree.selection.select`): two candidates overlap only if one is an ancestor of the other, so the disjoint sets are the
antichains of the tree. Our predicted IoUs are optimistic on the kept set and the out-of-fold curve is flat, so we use
tau = 0.30, chosen from the range that scored best on the public leaderboard.

## Results

Out-of-fold results on the 707 training images (every image is scored by models that never saw it; 1154 image-annotator
pairs; pooled PQ). Public leaderboard values are those displayed by the competition site.

| configuration | out-of-fold PQ | public score |
|---|---|---|
| dense maps + plain keep rule | 0.4523 | 0.39 |
| + component tree + hurdle LightGBM + CNN scorer | 0.4570 | 0.39 |
| **+ detector evidence (this submission, tau 0.30)** | **0.4652** (3 cross-fitting shuffles: 0.4646, 0.4650, 0.4659) | **0.41** |
| agreement between two annotators (reference) | 0.339 | - |

The exact evaluation of `notebooks/pipeline.ipynb` (stage 7) gives for shuffle 0: PQ 0.4647, SQ 0.6905, RQ 0.6730 (TP 5097,
FP 1852, FN 3102); the value computed from the stored candidate targets differs by 6e-5. The out-of-fold PQ as a function of
tau (shuffle 0): 0.4645 at 0.20, 0.4658 at 0.24, 0.4668 at 0.26, 0.4646 at 0.30, 0.4608 at 0.34, 0.4569 at 0.38.

Distribution of the matched pairs (IoU > 0.5), n = 5097:

| | mean | 10% | 25% | 50% | 75% | 90% |
|---|---|---|---|---|---|---|
| IoU | 0.690 | 0.568 | 0.629 | 0.698 | 0.757 | 0.798 |
| Dice | 0.814 | 0.724 | 0.772 | 0.822 | 0.862 | 0.888 |

62.2 % of the annotated masks have a prediction with IoU > 0.5 (best Dice > 0.5: 70.2 %).

One-to-many and many-to-one relations (8199 annotated masks, 6948 predictions; two definitions of a partner, see the
notebook):

| partner rule | one-to-many (annotated masks with 2 or more partners) | many-to-one (predictions with 2 or more partners) | annotated masks without partner | predictions without partner |
|---|---|---|---|---|
| overlap (IoU > 0) | 178 (2.2 %) | 144 (2.1 %) | 2025 | 730 |
| half area | 165 (2.0 %) | 128 (1.8 %) | 2404 | 1420 |

## Hardware, run time, reproducibility

| path | hardware | time |
|---|---|---|
| `python -m pqtree.pipeline` (stored artefacts) | CPU, 4 cores, 0.3 GB peak RAM | about 1 min |
| `notebooks/pipeline.ipynb`, all switches `False` | CPU, 3-4 cores, 0.8 GB peak RAM | about 6 min measured with 3 cores (13 min on a shared machine); the exact evaluation of stage 7 takes most of it |
| `REFIT_LGBM = True` | CPU | about 15 min (final fit 1 min; three cross-fitting shuffles 4.5 min each) |
| `RECOMPUTE_EVIDENCE = True` | CPU | about 55 min |
| `RECOMPUTE_MAPS = True` | 2 x GPU (Tesla T4 class) | about 40 min |
| `RETRAIN_CNN_SCORER = True` | 1 GPU (T4) | about 25 min |
| `RECOMPUTE_DETECTIONS = True` | 1 GPU (T4) | about 1 h |

Run times of the GPU stages are those of the Kaggle runs that produced the stored results. Package versions: see
`requirements.txt` (default path, notebook and tests; versions of the container in which they were run) and
`requirements-gpu.txt` (GPU stages). The LightGBM fit is deterministic here: two fits on identical inputs gave
bit-identical predictions, and the stored models give the reference CSV exactly. Recomputing the GPU stages gives slightly
different results and therefore possibly a slightly different CSV.

## Data, labels and licences

Competition data: Ahmadzadeh et al., *A dataset of manually annotated filaments from H-alpha observations*, Scientific Data
11, 1031 (2024), doi:10.1038/s41597-024-03876-y; competition: *Solar Filament Segmentation Challenge 2026*,
https://kaggle.com/competitions/filament-segmentation-2026. This work uses GONG data obtained by the NSO Integrated
Synoptic Program, managed by the National Solar Observatory (AURA, NSF), with contributions from NOAA.

The repository contains no competition image files, no annotation file, and no polygons or masks of the annotators. The executed notebook shows a few small illustrative crops (H-alpha images from the GONG network, NSO; MAGFiLO annotation outlines; CC BY-NC 4.0, non-commercial use, credit as in the data reference above). Label-derived
content is limited to: the columns `u`, `h`, `m` and `nann` of the training candidate table
(`artifacts/candidates_oof.tbl.npz`: mean matched IoU, share of annotators with a match, mean IoU of the matching
annotators and number of annotators, all aggregated over the annotators of an image) and the number of annotated masks per
training image (`artifacts/ngt_train.json`). There are no annotator identifiers. Image identifiers are the competition file
names.

Code: MIT licence (`LICENSE`), copyright 2026 Yuvraj Sharma. Stored artefacts (`artifacts/`): CC BY-NC-SA 4.0, the most
restrictive licence among their inputs. Pre-trained models and software are listed with their licences in `THIRD_PARTY.md`.

## Repository layout

```
pqtree/                   package, one module per stage (see the table above)
notebooks/pipeline.ipynb  the walk-through with all switches (stored with its outputs)
tests/                    unit tests (selection, PQ, relations) and the reproduction test
tools/gpu_stage_check.py  comparison of the packaged GPU stages with the stored results
tools/gpu_stage_check_result.json  output of that check on a Kaggle T4 (second run)
reference/submission.csv  the reference submission (sha256 above)
artifacts/                weights and stored stage outputs, MANIFEST.json lists name, size, sha256 and stage
requirements.txt, requirements-gpu.txt, THIRD_PARTY.md, LICENSE, CITATION.cff
```
