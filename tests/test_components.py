"""Small, fast tests of the building blocks on synthetic data (no artefacts and no competition files needed)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pqtree import artifacts, candidates, lgbm_scorer, selection  # noqa: E402
from pqtree.config import H, W  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# candidates: component tree and targets
# ---------------------------------------------------------------------------------------------------------------------
def test_component_tree_of_two_nested_blobs():
    prob = np.zeros((64, 64), np.uint8)
    prob[5:35, 5:35] = 150                     # outer blob: present at thresholds 0.2 ... 0.5 (levels 0-3)
    prob[10:16, 10:16] = 230                   # two inner blobs (36 px each): present at all levels
    prob[25:31, 25:31] = 230
    img = np.full((64, 64), 100, np.uint8)
    rows, labels = candidates.image_candidates("20200101000000Th", prob, img, {})
    df = candidates.add_tree_features(pd.DataFrame(rows))
    per_level = df.groupby("level").size().to_dict()
    assert per_level == {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}
    assert len(labels) == 8
    # the outer blob is one chain of single children; its level-3 node has the two inner blobs as children
    assert df[df.level == 0].parent.tolist() == [0]
    n_child = df.set_index(["level", "cid"]).n_child
    assert [n_child[(lv, 1)] for lv in range(3)] == [1, 1, 1]
    assert n_child[(3, 1)] == 2
    assert (df[(df.level >= 4) & (df.level < 7)].n_child == 1).all() and (df[df.level == 7].n_child == 0).all()
    # both inner blobs at level 4 have the level-3 component (id 1) as parent and 36 / 900 of its area
    inner = df[df.level == 4]
    assert inner.parent.tolist() == [1, 1]
    assert np.allclose(inner.area_ratio, 36 / 900)


def test_candidate_targets_match_hand_computed_iou():
    lab = np.zeros((20, 20), np.int32)
    lab[0:10, 0:10] = 1                        # candidate 1: 100 px
    lab[12:20, 12:20] = 2                      # candidate 2: 64 px, overlaps no annotation
    area = np.bincount(lab.ravel())
    m1 = np.zeros((20, 20), bool)
    m1[0:10, 0:8] = True                       # annotator 1: 80 px inside candidate 1 -> IoU 80 / 100
    gt1 = [(0, 10, 0, 8, m1[0:10, 0:8].copy(), 80)]
    gt2 = []                                   # annotator 2 marked nothing
    u, h, m = candidates.candidate_targets(lab, area, [gt1, gt2])
    assert np.allclose(u, [0.4, 0.0])          # mean matched IoU over the two annotators
    assert np.allclose(h, [0.5, 0.0])          # share of annotators with a match
    assert np.isclose(m[0], 0.8) and np.isnan(m[1])


def test_candidate_below_half_iou_is_not_a_match():
    lab = np.zeros((20, 20), np.int32)
    lab[0:10, 0:10] = 1
    ann = np.zeros((20, 20), bool)
    ann[0:10, 0:5] = True                      # IoU exactly 0.5 does not count (the rule is IoU > 0.5)
    gt = [(0, 10, 0, 5, ann[0:10, 0:5].copy(), 50)]
    u, h, _ = candidates.candidate_targets(lab, np.bincount(lab.ravel()), [gt])
    assert u[0] == 0 and h[0] == 0


# ---------------------------------------------------------------------------------------------------------------------
# selection: validation and fallback rows
# ---------------------------------------------------------------------------------------------------------------------
def _rle(y0, y1, x0, x1):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return mu.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _write(path, rows):
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(path, index=False)
    return path


def test_validate_submission_accepts_a_valid_file(tmp_path):
    p = _write(tmp_path / "ok.csv", [("a_1", _rle(0, 10, 0, 10)), ("a_2", _rle(20, 30, 20, 30)), ("b_1", _rle(0, 10, 0, 10))])
    assert selection.validate_submission(p, n_images=2).startswith("valid: 3 rows, 2 images")


def test_validate_submission_rejects_overlap_duplicates_empty_and_wrong_image_count(tmp_path):
    overlap = _write(tmp_path / "overlap.csv", [("a_1", _rle(0, 10, 0, 10)), ("a_2", _rle(5, 15, 5, 15))])
    with pytest.raises(ValueError, match="overlapping masks in a"):
        selection.validate_submission(overlap)
    dup = _write(tmp_path / "dup.csv", [("a_1", _rle(0, 10, 0, 10)), ("a_1", _rle(20, 30, 20, 30))])
    with pytest.raises(ValueError, match="duplicate ids"):
        selection.validate_submission(dup)
    empty = _write(tmp_path / "empty.csv", [("a_1", _rle(0, 0, 0, 0))])
    with pytest.raises(ValueError, match="empty mask in a"):
        selection.validate_submission(empty)
    ok = _write(tmp_path / "ok.csv", [("a_1", _rle(0, 10, 0, 10))])
    with pytest.raises(ValueError, match="1 images, expected 2"):
        selection.validate_submission(ok, n_images=2)
    columns = tmp_path / "columns.csv"
    pd.DataFrame({"id": ["a_1"], "rle": ["x"]}).to_csv(columns, index=False)
    with pytest.raises(ValueError, match="unexpected columns"):
        selection.validate_submission(columns)


def test_submission_rows_gives_an_image_without_choice_its_best_candidate():
    df = pd.DataFrame({"stem": ["a", "b", "b", "b"], "level": [0, 0, 1, 1], "cid": [1, 1, 1, 2]})
    score = np.array([0.9, 0.2, 0.7, 0.5])
    chosen = [("a", 0, 1)]                      # image b has no chosen candidate
    labels = np.zeros((8, 8), np.int32)
    labels[1:3, 1:3] = 1
    labels[5:8, 5:8] = 2
    rows, n_fallback = selection.submission_rows(df, score, chosen, lambda stem: (lambda level: labels))
    assert n_fallback == 1
    assert [r[0] for r in rows] == ["a_1", "b_1"]                 # exactly one row for b
    best = mu.decode({"size": [8, 8], "counts": rows[1][1].encode("ascii")})
    assert (best == (labels == 1)).all()                          # candidate (level 1, cid 1) has the highest score of b


# ---------------------------------------------------------------------------------------------------------------------
# artefact access
# ---------------------------------------------------------------------------------------------------------------------
def test_fetch_artifact_errors_and_success(tmp_path):
    good, bad = b"stored content", b"other content"
    (tmp_path / "good.bin").write_bytes(good)
    (tmp_path / "bad.bin").write_bytes(bad)
    manifest = {"files": {"good.bin": {"sha256": hashlib.sha256(good).hexdigest()},
                          "bad.bin": {"sha256": hashlib.sha256(b"not the content").hexdigest()},
                          "gone.bin": {"sha256": hashlib.sha256(b"x").hexdigest()}}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert artifacts.fetch_artifact("good.bin", tmp_path) == tmp_path / "good.bin"
    with pytest.raises(KeyError):
        artifacts.fetch_artifact("unknown.bin", tmp_path)
    with pytest.raises(FileNotFoundError, match="complete"):
        artifacts.fetch_artifact("gone.bin", tmp_path)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        artifacts.fetch_artifact("bad.bin", tmp_path)


def test_table_round_trip(tmp_path):
    df = pd.DataFrame({"stem": ["a", "b"], "level": [0, 3], "score": [0.25, 0.5]})
    artifacts.save_table(df, tmp_path / "t.npz")
    back = artifacts.load_table(tmp_path / "t.npz")
    assert list(back.columns) == ["stem", "level", "score"]
    assert back.stem.tolist() == ["a", "b"] and back.level.tolist() == [0, 3] and back.score.tolist() == [0.25, 0.5]


# ---------------------------------------------------------------------------------------------------------------------
# hurdle model
# ---------------------------------------------------------------------------------------------------------------------
def test_hurdle_model_fits_and_predicts_in_unit_interval(tmp_path):
    rng = np.random.RandomState(0)
    n = 300
    x = rng.rand(n, 3)
    h = np.clip(x[:, 0] + 0.1 * rng.randn(n), 0, 1)
    train = pd.DataFrame(x, columns=["f0", "f1", "f2"])
    train["h"] = np.where(h > 0.3, h, 0.0)
    train["m"] = np.where(train.h > 0, 0.5 + 0.4 * x[:, 1], np.nan)
    train["nann"] = rng.randint(1, 4, n)
    model = lgbm_scorer.fit_hurdle(train, features=["f0", "f1", "f2"], rounds=5)
    pred = model.predict(train)
    assert pred.shape == (n,) and (pred >= 0).all() and (pred <= 1).all()
    model.save(tmp_path)
    again = lgbm_scorer.HurdleModel.load(tmp_path / lgbm_scorer.HIT_FILE, tmp_path / lgbm_scorer.IOU_FILE, ["f0", "f1", "f2"])
    assert np.allclose(again.predict(train), pred)
