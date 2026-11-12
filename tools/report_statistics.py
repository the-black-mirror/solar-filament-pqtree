"""Evaluation statistics of the technical report, recomputed from the repository.

For one cross-fitting shuffle (default 0) and the final threshold ``config.TAU`` this script selects instances on the
out-of-fold scores of the 707 training images, scores them against every annotator and prints

* pooled PQ, SQ, RQ with bootstrap 95 % percentile intervals over images, and TP, FP, FN;
* PQ for five groups of annotation batches (with intervals and image counts);
* PQ per annotator-image (quantiles, inter-quartile range, share of exact zeros);
* the matched pairs (IoU > 0.5): IoU and Dice mean and quantiles; best-IoU / best-Dice statistics per annotated mask;
* relation degrees (partners with IoU > 0) and the counts of the half-area rule;
* morphology of kept instances and annotated masks, and the share of kept instances per tree threshold;
* the three-shuffle mean PQ at tau 0.26, 0.30 and 0.34 from the stored candidate targets.

It needs the artefacts of the repository and the training annotation file of the competition (no image files):

    python tools/report_statistics.py [--train-json PATH] [--seed 0] [--json outputs/report_statistics.json]

The path of the annotation file defaults to ``$PQTREE_TRAIN_JSON``. The annotation batches are the first four characters
of the annotation-image identifiers (``<batch>-<...>`` in the annotation file). They are distributed over five groups by
a greedy rule: batches in decreasing order of their number of images (images that have candidates), each into the
group that currently holds the fewest images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pqtree import candidates, evaluation, selection  # noqa: E402
from pqtree.artifacts import fetch_artifact, load_table  # noqa: E402
from pqtree.config import H, KEY, LEVELS, TAU, W  # noqa: E402

N_GROUPS = 5
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0
QUANTILES = (10, 25, 50, 75, 90)


# ---------------------------------------------------------------------------------------------------------------------
# pure functions (tested on synthetic inputs in tests/test_report_statistics.py)
# ---------------------------------------------------------------------------------------------------------------------
def pooled_pq(rows: np.ndarray) -> tuple[float, float, float]:
    """PQ, SQ, RQ of pooled counts; ``rows`` has one line (sum of IoU of matches, TP, FP, FN) per image."""
    denominator = rows[:, 1].sum() + 0.5 * rows[:, 2].sum() + 0.5 * rows[:, 3].sum()
    sum_iou, tp = rows[:, 0].sum(), rows[:, 1].sum()
    return float(sum_iou / denominator), float(sum_iou / tp), float(tp / denominator)


def bootstrap_resamples(rows: np.ndarray, rng: np.random.Generator, reps: int, factors: bool = True) -> np.ndarray:
    """Pooled PQ (and SQ, RQ if ``factors``) of ``reps`` resamples of the lines of ``rows`` (images drawn with replacement)."""
    n = len(rows)
    out = [pooled_pq(rows[rng.integers(0, n, n)]) for _ in range(reps)]
    out = np.array(out)
    return out if factors else out[:, 0]


def percentile_interval(values: np.ndarray) -> list[float]:
    """Bootstrap 95 % percentile interval."""
    return np.percentile(values, [2.5, 97.5], axis=0).tolist()


def elongation(xs: np.ndarray, ys: np.ndarray) -> float:
    """Square root of the ratio of the eigenvalues of the pixel covariance (each increased by 0.1, as in the candidate descriptors)."""
    if len(xs) < 3:
        return 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(np.float64)))
    return float(np.sqrt((ev[1] + 0.1) / (max(ev[0], 0) + 0.1)))


def batch_groups(ids_of_stem: Mapping[str, Sequence[str]], stems: Sequence[str], n_groups: int = N_GROUPS) -> dict[str, int]:
    """Group index of every image: annotation batches (first four characters of the smallest annotation-image id) balanced greedily.

    Batches are taken in decreasing order of their number of images among ``stems`` (ties in the order of
    ``pandas.Series.value_counts``) and each goes to the group that holds the fewest images so far.
    """
    batch = {s: sorted(i.split("-")[0] for i in ids_of_stem[s])[0][:4] for s in ids_of_stem}
    count = pd.Series({s: batch[s] for s in stems}).value_counts()
    load = np.zeros(n_groups)
    group_of_batch = {}
    for b in count.index:
        k = int(np.argmin(load))
        group_of_batch[b] = k
        load[k] += count[b]
    return {s: group_of_batch.get(batch[s], -1) for s in ids_of_stem}


def summarise_quantiles(values: np.ndarray, qs: Sequence[int] = QUANTILES) -> list[float]:
    """Percentiles ``qs`` of ``values`` as a list."""
    return np.percentile(values, qs).tolist()


# ---------------------------------------------------------------------------------------------------------------------
# evaluation loop
# ---------------------------------------------------------------------------------------------------------------------
def score_selection(sel: pd.DataFrame, prob_maps: Mapping[str, np.ndarray], annotations: tuple[dict, dict], stems: Sequence[str]) -> dict:
    """Score the selected instances against every annotator and collect everything the statistics need."""
    by_image, ids_of_stem = annotations
    by_stem = {s: g for s, g in sel.groupby("stem")}
    entries = []                       # one line per annotator-image: stem, n_gt, n_pred, sum of IoU, TP, FP, FN
    matched_iou, matched_dice, gt_rows, pr_rows = [], [], [], []
    gt_area, gt_elong, pr_area, pr_elong = [], [], [], []
    for stem in stems:
        chosen = by_stem.get(stem)
        pred = np.zeros((H, W), np.int32)
        n_pred = 0
        if chosen is not None:
            labs = {}
            for k, row in enumerate(chosen.itertuples(), 1):
                lv = int(row.level)
                if lv not in labs:
                    labs[lv] = candidates.label_image(prob_maps[stem], lv)
                pred[labs[lv] == row.cid] = k
            n_pred = len(chosen)
            ys, xs = np.nonzero(pred)
            lab = pred[ys, xs]
            for k in range(1, n_pred + 1):
                inside = lab == k
                pr_area.append(float(inside.sum()))
                pr_elong.append(elongation(xs[inside], ys[inside]))
        for gts in candidates.annotator_masks(by_image, ids_of_stem[stem]):
            for (y0, y1, x0, x1, m, a) in gts:
                gy, gx = np.nonzero(m)
                gt_area.append(float(a))
                gt_elong.append(elongation(gx, gy))
            cmp = evaluation.compare_image(pred, n_pred, gts)
            entries.append((stem, len(gts), n_pred, cmp["sum_iou"], cmp["tp"], cmp["fp"], cmp["fn"]))
            matched_iou.append(cmp["matched_iou"])
            matched_dice.append(cmp["matched_dice"])
            gt_rows += cmp["gt"]
            pr_rows += cmp["pred"]
    entries = pd.DataFrame(entries, columns=["stem", "ngt", "npred", "siou", "tp", "fp", "fn"])
    return dict(entries=entries, matched_iou=np.concatenate(matched_iou), matched_dice=np.concatenate(matched_dice),
                gt=pd.DataFrame(gt_rows), pred=pd.DataFrame(pr_rows), gt_area=np.array(gt_area), gt_elong=np.array(gt_elong),
                pr_area=np.array(pr_area), pr_elong=np.array(pr_elong))


def statistics(res: dict, sel: pd.DataFrame, group_of: Mapping[str, int], n_images: int, reps: int = BOOTSTRAP_REPS) -> dict:
    """All statistics of the report from the scored selection."""
    E, gt, pr = res["entries"], res["gt"], res["pred"]
    S = E.groupby("stem").agg(siou=("siou", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum")).reset_index()
    S["group"] = S.stem.map(group_of)
    A = S[["siou", "tp", "fp", "fn"]].to_numpy(float)
    pq, sq, rq = pooled_pq(A)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ci = percentile_interval(bootstrap_resamples(A, rng, reps))
    out = dict(n_images=int(len(S)), n_annotator_images=int(len(E)), bootstrap_reps=reps,
               pq=pq, sq=sq, rq=rq, pq_ci=[ci[0][0], ci[1][0]], sq_ci=[ci[0][1], ci[1][1]], rq_ci=[ci[0][2], ci[1][2]],
               TP=int(A[:, 1].sum()), FP=int(A[:, 2].sum()), FN=int(A[:, 3].sum()),
               n_pred=int(E.npred.sum()), n_gt=int(E.ngt.sum()))
    out["groups"] = {}
    for k in range(N_GROUPS):
        a = S[S.group == k][["siou", "tp", "fp", "fn"]].to_numpy(float)
        b = bootstrap_resamples(a, rng, reps, factors=False)
        out["groups"][str(k)] = dict(n_images=int(len(a)), pq=pooled_pq(a)[0], ci=percentile_interval(b))
    out["unassigned_images"] = int((S.group < 0).sum())
    den = S.tp + 0.5 * S.fp + 0.5 * S.fn
    out["per_image_pq_quantiles"] = summarise_quantiles((S.siou / den).to_numpy())
    # PQ per annotator-image
    pq_ai = (E.siou / (E.tp + 0.5 * E.fp + 0.5 * E.fn)).to_numpy()
    out["pq_per_annotator_image"] = dict(quantiles=summarise_quantiles(pq_ai), iqr=float(np.percentile(pq_ai, 75) - np.percentile(pq_ai, 25)),
                                         share_zero=float((pq_ai == 0).mean()))
    # matched pairs and best scores per annotated mask
    mi, md = res["matched_iou"], res["matched_dice"]
    out["matched_pairs"] = dict(n=int(len(mi)), iou_mean=float(mi.mean()), iou_quantiles=summarise_quantiles(mi),
                                dice_mean=float(md.mean()), dice_quantiles=summarise_quantiles(md))
    out["best_per_annotated_mask"] = dict(share_iou_gt_05=float((gt.best_iou > 0.5).mean()), share_dice_gt_05=float((gt.best_dice > 0.5).mean()),
                                          median_iou_with_misses=float(gt.best_iou.median()), median_dice_with_misses=float(gt.best_dice.median()))
    # relations
    g, p = gt.n_pred_overlap.to_numpy(), pr.n_gt_overlap.to_numpy()
    n_pairs = int(g.sum())
    out["overlapping_pairs"] = dict(n=n_pairs, share_iou_gt_05=float(len(mi) / n_pairs))
    degree = lambda v: {"0": int((v == 0).sum()), "1": int((v == 1).sum()), "2": int((v == 2).sum()), "3+": int((v >= 3).sum()), "total": int(len(v))}
    out["degree_annotated"], out["degree_predicted"] = degree(g), degree(p)
    gf, pi = gt.n_pred_inside.to_numpy(), pr.n_gt_inside.to_numpy()
    out["half_area_rule"] = dict(one_to_many=int((gf >= 2).sum()), one_to_many_pct=float((gf >= 2).mean() * 100),
                                 many_to_one=int((pi >= 2).sum()), many_to_one_pct=float((pi >= 2).mean() * 100),
                                 gt_without_fragment=int((gf == 0).sum()), gt_without_fragment_pct=float((gf == 0).mean() * 100),
                                 pred_without_gt=int((pi == 0).sum()), pred_without_gt_pct=float((pi == 0).mean() * 100), n_pred_rows=int(len(pi)))
    # morphology
    out["morphology"] = dict(
        predicted_per_image=float(len(sel) / n_images), predicted_per_annotator_image=float(E.npred.sum() / len(E)),
        annotated_per_annotator_image=float(E.ngt.sum() / len(E)), annotated_per_annotator_image_median=float(E.ngt.median()),
        kept_area_median=float(np.median(res["pr_area"])), kept_area_iqr=summarise_quantiles(res["pr_area"], (25, 75)),
        annotated_area_median=float(np.median(res["gt_area"])), annotated_area_iqr=summarise_quantiles(res["gt_area"], (25, 75)),
        kept_elongation_median=float(np.median(res["pr_elong"])), kept_elongation_iqr=summarise_quantiles(res["pr_elong"], (25, 75)),
        annotated_elongation_median=float(np.median(res["gt_elong"])), annotated_elongation_iqr=summarise_quantiles(res["gt_elong"], (25, 75)),
        level_share={f"{LEVELS[l]:.1f}": float((sel.level == l).mean()) for l in range(len(LEVELS))},
        level_share_0203=float((sel.level <= 1).mean()), n_selected=int(len(sel)))
    return out


def three_shuffle_means(cands: pd.DataFrame, scores: pd.DataFrame, n_gt: int, taus: Sequence[float] = (0.26, 0.30, 0.34)) -> dict:
    """Mean over the cross-fitting shuffles 0, 1, 2 of the pooled PQ from the stored candidate targets, for each tau."""
    table = cands.merge(scores, on=KEY, how="left")
    result = {}
    for tau in taus:
        per_seed = []
        for seed in (0, 1, 2):
            chosen = selection.select(table, table[f"score_seed{seed}"].values, tau)
            per_seed.append(evaluation.target_based_pq(table, chosen, n_gt))
        result[f"{tau:.2f}"] = dict(per_shuffle=per_seed, mean=float(np.mean(per_seed)))
    return result


# ---------------------------------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------------------------------
def format_report(r: dict) -> str:
    """Readable text table of the statistics."""
    f = lambda v, d=4: f"{v:.{d}f}"
    iv = lambda ci, d=4: f"[{f(ci[0], d)}, {f(ci[1], d)}]"
    q = lambda v, d=3: "  ".join(f"{x:.{d}f}" for x in v)
    L = [f"Evaluation statistics (shuffle {r['shuffle']}, tau {r['tau']:.2f}, {r['n_images']} images, {r['n_annotator_images']} annotator-images)", ""]
    L += [f"Pooled metric (bootstrap 95 % intervals over images, {r['bootstrap_reps']} resamples)",
          f"  PQ {f(r['pq'])} {iv(r['pq_ci'])}   SQ {f(r['sq'])} {iv(r['sq_ci'])}   RQ {f(r['rq'])} {iv(r['rq_ci'])}",
          f"  TP {r['TP']}  FP {r['FP']}  FN {r['FN']}   ({r['n_pred']} predictions, {r['n_gt']} annotated masks)", ""]
    L += ["PQ per group of annotation batches"]
    for k, v in r["groups"].items():
        L.append(f"  group {k}: {v['n_images']:3d} images  PQ {f(v['pq'])} {iv(v['ci'])}")
    L += [f"  images without group: {r['unassigned_images']}", ""]
    a = r["pq_per_annotator_image"]
    L += ["PQ per annotator-image", f"  quantiles 10/25/50/75/90 %: {q(a['quantiles'])}   IQR {f(a['iqr'])}   share exactly zero {f(a['share_zero'])}",
          f"  per image (annotators pooled), quantiles 10/25/50/75/90 %: {q(r['per_image_pq_quantiles'])}", ""]
    m, b = r["matched_pairs"], r["best_per_annotated_mask"]
    L += [f"Matched pairs (IoU > 0.5), n = {m['n']}", f"  IoU  mean {f(m['iou_mean'], 3)}  quantiles 10/25/50/75/90 %: {q(m['iou_quantiles'])}",
          f"  Dice mean {f(m['dice_mean'], 3)}  quantiles 10/25/50/75/90 %: {q(m['dice_quantiles'])}",
          f"  annotated masks with best IoU > 0.5: {f(b['share_iou_gt_05'], 3)}, best Dice > 0.5: {f(b['share_dice_gt_05'], 3)}",
          f"  median best IoU / Dice with misses counted as zero: {f(b['median_iou_with_misses'], 3)} / {f(b['median_dice_with_misses'], 3)}",
          f"  overlapping pairs (IoU > 0): {r['overlapping_pairs']['n']}, of which IoU > 0.5: {f(r['overlapping_pairs']['share_iou_gt_05'], 3)}", ""]
    L += ["Relation degrees (partners with IoU > 0)", f"  annotated masks: {r['degree_annotated']}", f"  predictions:     {r['degree_predicted']}"]
    h = r["half_area_rule"]
    L += [f"  half-area rule: one-to-many {h['one_to_many']} ({f(h['one_to_many_pct'], 2)} %), many-to-one {h['many_to_one']} ({f(h['many_to_one_pct'], 2)} %), "
          f"annotated without fragment {h['gt_without_fragment']} ({f(h['gt_without_fragment_pct'], 2)} %), predictions without annotated mask {h['pred_without_gt']} ({f(h['pred_without_gt_pct'], 2)} %)", ""]
    o = r["morphology"]
    L += ["Morphology", f"  predictions per image {f(o['predicted_per_image'], 3)}, per annotator-image {f(o['predicted_per_annotator_image'], 3)}; "
          f"annotated masks per annotator-image {f(o['annotated_per_annotator_image'], 3)} (median {o['annotated_per_annotator_image_median']:.0f})",
          f"  area (px): kept median {o['kept_area_median']:.0f} IQR {q(o['kept_area_iqr'], 0)}; annotated median {o['annotated_area_median']:.0f} IQR {q(o['annotated_area_iqr'], 0)}",
          f"  elongation: kept median {f(o['kept_elongation_median'], 3)} IQR {q(o['kept_elongation_iqr'])}; annotated median {f(o['annotated_elongation_median'], 3)} IQR {q(o['annotated_elongation_iqr'])}",
          "  share of kept instances per tree threshold: " + ", ".join(f"{t} {f(v, 3)}" for t, v in o["level_share"].items()),
          f"  share at thresholds 0.2 and 0.3: {f(o['level_share_0203'], 3)}   ({o['n_selected']} kept instances)", ""]
    L += ["Mean PQ over the three cross-fitting shuffles (stored targets)"]
    for tau, v in r["three_shuffle_pq"].items():
        L.append(f"  tau {tau}: mean {f(v['mean'], 5)}  (shuffles {', '.join(f(x, 5) for x in v['per_shuffle'])})")
    return "\n".join(L)


def main() -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-json", default=os.environ.get("PQTREE_TRAIN_JSON"), help="competition training annotation file (default: $PQTREE_TRAIN_JSON)")
    ap.add_argument("--seed", type=int, default=0, choices=(0, 1, 2), help="cross-fitting shuffle (default 0)")
    ap.add_argument("--json", default="outputs/report_statistics.json", help="output file of the statistics")
    ap.add_argument("--reps", type=int, default=BOOTSTRAP_REPS, help="bootstrap resamples")
    args = ap.parse_args()
    if not args.train_json or not Path(args.train_json).exists():
        raise SystemExit("the training annotation file is needed: pass --train-json or set PQTREE_TRAIN_JSON")

    cands = load_table(fetch_artifact("candidates_oof.tbl.npz"))
    scores = pd.read_csv(fetch_artifact("oof_scores.csv.gz"), float_precision="round_trip")
    n_gt = sum(json.load(open(fetch_artifact("ngt_train.json"))).values())
    table = cands.merge(scores, on=KEY, how="left")
    sel = pd.DataFrame(selection.select(table, table[f"score_seed{args.seed}"].values, TAU), columns=KEY)
    prob_maps = np.load(fetch_artifact("oof_probs_unet.npz"))
    annotations = candidates.load_annotations(args.train_json)
    stems = sorted(annotations[1])
    res = score_selection(sel, prob_maps, annotations, stems)
    groups = batch_groups(annotations[1], sorted(cands.stem.unique()))
    out = statistics(res, sel, groups, len(stems), args.reps)
    out.update(shuffle=args.seed, tau=TAU, three_shuffle_pq=three_shuffle_means(cands, scores, n_gt))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(format_report(out))
    print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
