"""Unit tests on tiny synthetic inputs: the selection dynamic programme and the Panoptic Quality computation."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pqtree.evaluation import compare_image, crop_mask, panoptic_quality, relation_counts  # noqa: E402
from pqtree.selection import select  # noqa: E402


def _forest(rows):
    """rows: (level, cid, parent, score) of one image -> DataFrame and score vector."""
    df = pd.DataFrame(rows, columns=["level", "cid", "parent", "score"])
    df.insert(0, "stem", "img")
    return df, df.score.values


def _brute_force(df, score, tau):
    """Best antichain by exhaustive search (candidates conflict if one is an ancestor of the other)."""
    parent = {(r.level, r.cid): (r.level - 1, r.parent) for r in df.itertuples() if r.level > 0}

    def ancestors(k):
        while k in parent:
            k = parent[k]
            yield k

    keys = [(r.level, r.cid) for r in df.itertuples()]
    val = {k: s - tau for k, s in zip(keys, score)}
    best = 0.0
    for r in range(len(keys) + 1):
        for subset in itertools.combinations(keys, r):
            s = set(subset)
            if any(a in s for k in subset for a in ancestors(k)):
                continue
            best = max(best, sum(val[k] for k in subset))
    return best


def test_selection_prefers_children_when_their_sum_wins():
    # root (score .5) with two children (.45 each); tau .3 -> children give .30 versus parent .20
    df, sc = _forest([(0, 1, 0, 0.5), (1, 1, 1, 0.45), (1, 2, 1, 0.45)])
    assert sorted(select(df, sc, 0.3)) == [("img", 1, 1), ("img", 1, 2)]
    # with tau .4 the children give .10 versus the parent's .10 -> the parent is kept on ties
    assert select(df, sc, 0.4) == [("img", 0, 1)]
    # with tau .5 nothing has positive gain
    assert select(df, sc, 0.5) == []


def test_selection_is_optimal_and_disjoint():
    rng = np.random.RandomState(0)
    for _ in range(20):
        rows, cid = [], {0: 0}
        for lv in range(3):                       # random forest with up to 3 levels
            n = rng.randint(1, 4)
            for c in range(1, n + 1):
                parent = 0 if lv == 0 else rng.randint(1, cid[lv - 1] + 1)
                rows.append((lv, c, parent, rng.rand()))
            cid[lv] = n
        df, sc = _forest(rows)
        tau = 0.3
        chosen = select(df, sc, tau)
        got = sum(sc[i] - tau for i, r in enumerate(df.itertuples()) if ("img", r.level, r.cid) in set(chosen))
        assert abs(got - _brute_force(df, sc, tau)) < 1e-9
        # no chosen candidate is the ancestor of another chosen candidate
        parent = {(r.level, r.cid): (r.level - 1, r.parent) for r in df.itertuples() if r.level > 0}
        keys = {(lv, cid) for _, lv, cid in chosen}
        for k in keys:
            a = k
            while a in parent:
                a = parent[a]
                assert a not in keys


def test_panoptic_quality_on_a_small_example():
    pred = np.zeros((10, 10), np.int32)
    pred[0:4, 0:4] = 1          # matches GT 0 exactly (IoU 1)
    pred[6:9, 6:9] = 2          # false positive
    gt0 = np.zeros((10, 10), bool); gt0[0:4, 0:4] = True
    gt1 = np.zeros((10, 10), bool); gt1[0:2, 7:10] = True      # missed
    gt2 = np.zeros((10, 10), bool); gt2[5:9, 6:9] = True       # IoU with pred 2 = 9 / 12 = 0.75
    gts = [crop_mask(m) for m in (gt0, gt1, gt2)]
    r = compare_image(pred, 2, gts)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 1)
    assert abs(r["sum_iou"] - 1.75) < 1e-12
    q = panoptic_quality(r["sum_iou"], r["tp"], r["fp"], r["fn"])
    assert abs(q["PQ"] - 1.75 / (2 + 0.5)) < 1e-12 and abs(q["SQ"] - 0.875) < 1e-12 and abs(q["RQ"] - 0.8) < 1e-12
    # Dice of the second match: 2 * 9 / (12 + 9)
    assert abs(sorted(r["matched_dice"])[0] - 18 / 21) < 1e-12


def test_split_and_merge_relations():
    pred = np.zeros((20, 20), np.int32)
    pred[0:5, 0:10] = 1          # two fragments of one ground-truth mask (each half inside it)
    pred[5:10, 0:10] = 2
    pred[12:14, 0:20] = 3        # one prediction that contains two small ground-truth masks
    gt_split = np.zeros((20, 20), bool); gt_split[0:10, 0:10] = True
    g_a = np.zeros((20, 20), bool); g_a[12:14, 0:4] = True
    g_b = np.zeros((20, 20), bool); g_b[12:14, 10:14] = True
    r = compare_image(pred, 3, [crop_mask(m) for m in (gt_split, g_a, g_b)])
    res = {"gt": pd.DataFrame(r["gt"]), "pred": pd.DataFrame(r["pred"])}
    rel = relation_counts(res)
    assert rel["one_to_many"] == 1 and rel["many_to_one"] == 1
    assert r["tp"] == 0 and r["fp"] == 3 and r["fn"] == 3


def test_unmatched_mask_has_zero_best_dice_and_overlap_relations():
    pred = np.zeros((10, 10), np.int32)
    pred[0:2, 0:2] = 1                                   # far from the ground-truth mask
    gt = np.zeros((10, 10), bool); gt[6:9, 6:9] = True
    r = compare_image(pred, 1, [crop_mask(gt)])
    assert r["gt"][0]["best_iou"] == 0 and r["gt"][0]["best_dice"] == 0
    # a sliver overlap counts as a partner under the overlap rule only
    pred2 = np.zeros((10, 10), np.int32)
    pred2[6:9, 0:7] = 1                                  # one column overlaps (inter 3), most of the prediction lies outside
    pred2[6:9, 8:10] = 2
    r2 = compare_image(pred2, 2, [crop_mask(gt)])
    res = {"gt": pd.DataFrame(r2["gt"]), "pred": pd.DataFrame(r2["pred"])}
    assert relation_counts(res, "overlap")["one_to_many"] == 1
    assert relation_counts(res, "half_area")["one_to_many"] == 0
