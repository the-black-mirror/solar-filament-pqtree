"""Evaluation on the out-of-fold predictions for the training images.

Panoptic Quality as defined by the competition: a predicted and a ground-truth mask are a true positive if IoU > 0.5;
unmatched predictions are false positives and unmatched ground-truth masks false negatives;
PQ = sum(IoU of true positives) / (TP + FP/2 + FN/2). An image has several annotators; every annotator's masks are scored
against the prediction separately and the counts are pooled over all (image, annotator) pairs.

Besides PQ this module reports the distribution of IoU and Dice scores and the one-to-many / many-to-one relations
between ground-truth and predicted masks.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .candidates import annotator_masks, label_image


def crop_mask(m: np.ndarray) -> Optional[tuple]:
    """(y0, y1, x0, x1, cropped mask, area) of a boolean mask, or None if it is empty."""
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return y0, y1, x0, x1, m[y0:y1, x0:x1].copy(), int(m.sum())


def compare_image(pred_label: np.ndarray, n_pred: int, gts: Sequence[tuple]) -> dict:
    """Compare one annotator's masks with the predicted label image.

    ``pred_label`` holds 0 for background and 1..n_pred for the (disjoint) predictions, ``gts`` are cropped masks as
    returned by :func:`crop_mask`. Returns the per-ground-truth and per-prediction records and the PQ counts.
    """
    parea = np.bincount(pred_label.ravel(), minlength=n_pred + 1).astype(np.float64)
    inter = np.zeros((len(gts), n_pred + 1))
    for g, (y0, y1, x0, x1, m, _) in enumerate(gts):
        inter[g] = np.bincount(pred_label[y0:y1, x0:x1][m], minlength=n_pred + 1)
    ga = np.array([a for (*_, a) in gts], float)
    union = ga[:, None] + parea[None, :] - inter
    iou = inter / np.maximum(union, 1)
    iou[:, 0] = 0
    dice = 2 * inter / np.maximum(ga[:, None] + parea[None, :], 1)
    dice[:, 0] = 0
    n_pred_overlap = (inter[:, 1:] > 0).sum(1)
    n_gt_overlap = (inter[:, 1:] > 0).sum(0)
    match = iou > 0.5
    tp = int(match.sum())
    gt_rec, pr_rec = [], []
    for g in range(len(gts)):
        j = int(np.argmax(iou[g])) if n_pred else 0
        frag = int(sum(1 for k in range(1, n_pred + 1) if inter[g, k] > 0.5 * parea[k]))
        gt_rec.append(dict(gt=g, area=ga[g], best_iou=iou[g, j] if n_pred else 0.0, best_dice=dice[g, j] if n_pred else 0.0,
                           matched=bool(match[g].any()), n_pred_inside=frag,
                           n_pred_overlap=int(n_pred_overlap[g])))
    for k in range(1, n_pred + 1):
        inside = int(sum(1 for g in range(len(gts)) if inter[g, k] > 0.5 * ga[g]))
        pr_rec.append(dict(pred=k, area=parea[k], best_iou=iou[:, k].max() if len(gts) else 0.0, matched=bool(match[:, k].any()),
                           n_gt_inside=inside, n_gt_overlap=int(n_gt_overlap[k - 1])))
    return dict(sum_iou=float(iou[match].sum()), tp=tp, fp=int(n_pred - match[:, 1:].any(0).sum()),
                fn=int(len(gts) - match.any(1).sum()), matched_iou=iou[match], matched_dice=dice[match], gt=gt_rec, pred=pr_rec)


def panoptic_quality(sum_iou: float, tp: int, fp: int, fn: int) -> dict:
    """PQ and its factors: segmentation quality SQ = sum_iou / TP and recognition quality RQ."""
    pq = sum_iou / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    sq = sum_iou / tp if tp else 0.0
    return dict(PQ=pq, SQ=sq, RQ=pq / sq if sq else 0.0, TP=tp, FP=fp, FN=fn)


def evaluate_selection(selected: pd.DataFrame, prob_maps: Mapping[str, np.ndarray], annotations: tuple[dict, dict],
                       stems: Sequence[str], log=print) -> dict:
    """Exact evaluation of a selection of candidates against the annotators.

    ``selected`` has the columns stem, level, cid. The label images are recomputed from ``prob_maps`` (the same
    thresholding that built the candidates). Returns the pooled PQ counts, the arrays of matched IoU and Dice values and
    the per-ground-truth / per-prediction tables.
    """
    by_image, ids_of_stem = annotations
    by_stem = {s: g for s, g in selected.groupby("stem")}
    tot = dict(sum_iou=0.0, tp=0, fp=0, fn=0)
    iou_m, dice_m, gt_rows, pr_rows = [], [], [], []
    for ii, stem in enumerate(stems):
        sel = by_stem.get(stem)
        pred = np.zeros(prob_maps[stem].shape, np.int32)
        n_pred = 0
        if sel is not None:
            labs = {int(lv): label_image(prob_maps[stem], int(lv)) for lv in sel.level.unique()}
            for k, row in enumerate(sel.itertuples(), 1):
                pred[labs[int(row.level)] == row.cid] = k
            n_pred = len(sel)
            del labs
        for ai, gts in enumerate(annotator_masks(by_image, ids_of_stem[stem])):
            cmp = compare_image(pred, n_pred, gts)
            for k in tot:
                tot[k] += cmp[k]
            iou_m.append(cmp["matched_iou"])
            dice_m.append(cmp["matched_dice"])
            gt_rows += [dict(stem=stem, annotator=ai, **r) for r in cmp["gt"]]
            pr_rows += [dict(stem=stem, annotator=ai, **r) for r in cmp["pred"]]
        if log and ii % 100 == 0:
            log(f"evaluated {ii} / {len(stems)} images")
    return dict(counts=tot, pq=panoptic_quality(tot["sum_iou"], tot["tp"], tot["fp"], tot["fn"]),
                matched_iou=np.concatenate(iou_m), matched_dice=np.concatenate(dice_m),
                gt=pd.DataFrame(gt_rows), pred=pd.DataFrame(pr_rows))


def relation_counts(result: Mapping, rule: str = "half_area") -> dict:
    """One-to-one, one-to-many and many-to-one relations between ground-truth and predicted masks.

    Two definitions of a *partner* are available:

    * ``rule="overlap"``: any pixel overlap (IoU > 0). One-to-many: annotated masks overlapped by two or more
      predictions (the object is split); many-to-one: predictions overlapping two or more annotated masks (objects are
      merged); ``gt_without_partner`` / ``pred_without_partner``: masks without any overlap.
    * ``rule="half_area"``: a prediction is a fragment of an annotated mask if at least half of its area lies inside it, and an
      annotated mask is covered by a prediction if at least half of its area lies inside that prediction; the same
      counts are made with these partners. This is stricter and ignores thin slivers.

    ``one_to_one`` is the number of annotated masks matched at IoU > 0.5 under both rules.
    """
    gt, pr = result["gt"], result["pred"]
    if rule == "overlap":
        g, p = gt.n_pred_overlap, pr.n_gt_overlap
    elif rule == "half_area":
        g, p = gt.n_pred_inside, pr.n_gt_inside
    else:
        raise ValueError(rule)
    return dict(gt_masks=len(gt), predictions=len(pr), one_to_one=int(gt.matched.sum()),
                one_to_many=int((g >= 2).sum()), many_to_one=int((p >= 2).sum()),
                gt_without_partner=int((g == 0).sum()), pred_without_partner=int((p == 0).sum()))


def quantile_table(result: Mapping) -> pd.DataFrame:
    """Mean and 10/25/50/75/90 % quantiles of the IoU and Dice of the matched pairs."""
    rows = {}
    for name, v in (("IoU", result["matched_iou"]), ("Dice", result["matched_dice"])):
        rows[name] = [v.mean()] + list(np.quantile(v, [.1, .25, .5, .75, .9]))
    return pd.DataFrame(rows, index=["mean", "10%", "25%", "50%", "75%", "90%"]).T.round(3)


def target_based_pq(cands: pd.DataFrame, chosen: Sequence[tuple], n_gt_total: int) -> float:
    """Pooled PQ from the stored candidate targets: sum(u x annotators) / (N/2 + K/2), N ground-truth masks, K kept x annotators."""
    sel = cands.set_index(["stem", "level", "cid"]).loc[list(chosen)]
    return float((sel.u * sel.nann).sum() / (0.5 * n_gt_total + 0.5 * sel.nann.sum()))
