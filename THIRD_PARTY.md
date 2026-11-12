# Third-party components and licences

The code in this repository is released under the MIT licence (`LICENSE`). The stored artefacts in `artifacts/` are
released under CC BY-NC-SA 4.0, the most restrictive licence among their inputs (see below). The repository uses the
following third-party material. No third-party model weights are copied into the repository; the stages that need them read
them from their public Kaggle datasets.

## Method source

| source | author | use here |
|---|---|---|
| Kaggle notebook "Solar Filament Segmentation 2026 (0.40)", https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40 | tushhaaarrrr | Source of the method description: the component tree of thresholded maps, the hurdle estimate of the expected matched IoU (hit probability x matched IoU), the exact antichain dynamic programme, and the single global threshold derived with Dinkelbach's argument. These were re-implemented here; the descriptors, the CNN scorer, the detector-agreement features and the code were written for this project. The idea of ensemble-disagreement descriptors also appears in that notebook; the implementation here (view-spread and encoder-gap maps) is ours. No code of that notebook is included: `pqtree/lgbm_scorer.py`, `pqtree/selection.py` and `pqtree/candidates.py` were written independently from its description (the notebook shows the dynamic programme only as a short illustration on a one-dimensional profile and does not contain the code of its scorer). |

## Model weights and data

| component | author | licence | use here |
|---|---|---|---|
| U-Net ensemble `dariushafshar/solar-filament-unet-weights`: `unet_resnet34_fold{0-4}.pt`, `unet_tu-tf_efficientnetv2_s_fold{0-4}.pt`, `folds.csv`, `solar_unet_receipt.json`. https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights | dariushafshar | **CC BY-NC-SA 4.0** | Stage 1 (`pqtree/dense_maps.py`), inference only, read from the Kaggle dataset. The stored probability maps (`*_unet.npz`) are derived from these weights and carry the same licence (attribution, non-commercial, share-alike). The public notebook by the same author, "Solar Filament U-Net: OOF 0.45 vs Public 0.39", describes the out-of-fold protocol. |
| Detector fold models `notoverkil/solar-filament-weights`: YOLO11m-seg (1536 and 2048 px), YOLO26m-seg (1536 px), Mask R-CNN ResNet50-FPN v2 (1536 px), 5 folds each. https://www.kaggle.com/datasets/notoverkil/solar-filament-weights | notoverkil | **AGPL-3.0** (release level; Ultralytics base) | Stage 4 (`pqtree/detector_evidence.py`), inference only, read from the Kaggle dataset. The AGPL applies to the model weights; the stored instance tables (`detector_instances_*.csv.gz`: run-length masks and confidences) are outputs of these models. |
| MAGFiLO H-alpha images and annotations (Solar Filament Segmentation Challenge 2026, https://kaggle.com/competitions/filament-segmentation-2026) | organisers of the challenge (Ahmadzadeh, Kempton, Li, Pevtsov) | competition data terms: non-commercial use and academic research | Training labels and inference images. No image file or annotation file is redistributed; the executed notebook `notebooks/pipeline.ipynb` shows crops and one complete image of H-alpha test images with our predicted outlines (one crop in three figures, a gallery of eight crops, one complete image) and one training crop with annotation outlines (GONG network, NSO), for non-commercial illustration (CC BY-NC 4.0), credited as in the data reference (Ahmadzadeh et al., Scientific Data 11, 1031, 2024). The artefacts hold only label-derived aggregates: the columns `u`, `h`, `m`, `nann` of `candidates_oof.tbl.npz` and the per-image mask counts `ngt_train.json` (aggregated over the annotators of an image; no polygon, mask or annotator identifier). The CNN scorer weights and the LightGBM models were trained on these targets. |
| ImageNet pre-trained ResNet18 (timm `resnet18`) | Ross Wightman and the timm contributors | Apache-2.0 (timm code); weights as published with timm | Initialisation of the CNN scorer when it is retrained (`RETRAIN_CNN_SCORER`). Downloaded by timm at that point. |

## Software

| package | licence | use |
|---|---|---|
| segmentation-models-pytorch 0.5.0 (Pavel Iakubovskii) | MIT | U-Net definition for the dense-map stage |
| timm 1.0.30 (Ross Wightman) | Apache-2.0 | ResNet18 backbone of the CNN scorer; encoder of the EfficientNetV2 U-Nets |
| PyTorch 2.11.0, torchvision 0.26.0 | BSD-3-Clause | GPU stages; Mask R-CNN definition |
| Ultralytics 8.4.121 | AGPL-3.0 | loads the YOLO fold models in `run_detectors` (not needed for the stored-artefact path) |
| LightGBM 4.6.0 (Microsoft) | MIT | hurdle scorer |
| pycocotools 2.0.11 | BSD-2-Clause | run-length encoding of masks |
| OpenCV (opencv-python-headless) | Apache-2.0 | connected components, filtering |
| NumPy, SciPy, pandas | BSD-3-Clause | numerics, tables |
| Matplotlib | Matplotlib licence (PSF-based) | figures in the notebook |
| Jupyter (nbconvert, ipykernel, ...), pytest | BSD-3-Clause, MIT | running the notebook and the tests |

## Notes on licences

* The competition rules (section 5a) exempt pre-trained models with incompatible licences from the open-source licence
  of the winning submission. The dense-map weights are non-commercial and the detector weights are AGPL-3.0, so the
  complete pipeline inherits those restrictions; the code in this repository does not.
* The artefacts are intermediate results of the pipeline and the weights of our own CNN scorer and LightGBM models. They are
  released together under CC BY-NC-SA 4.0 because they derive from non-commercial, share-alike inputs.
