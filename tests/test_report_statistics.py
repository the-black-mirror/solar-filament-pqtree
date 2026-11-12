"""Pure functions of tools/report_statistics.py on tiny synthetic inputs (no artefacts and no competition data needed)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import report_statistics as rs  # noqa: E402


def test_pooled_pq_sq_rq():
    rows = np.array([[1.0, 1, 0, 0], [0.75, 1, 1, 1]])      # sum of IoU, TP, FP, FN per image
    pq, sq, rq = rs.pooled_pq(rows)
    assert np.isclose(pq, 1.75 / 3) and np.isclose(sq, 0.875) and np.isclose(rq, 2 / 3)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng_rows = np.random.RandomState(1)
    rows = np.column_stack([rng_rows.uniform(0.5, 1, 40), np.ones(40), rng_rows.randint(0, 3, 40), rng_rows.randint(0, 3, 40)])
    a = rs.bootstrap_resamples(rows, np.random.default_rng(0), 200)
    b = rs.bootstrap_resamples(rows, np.random.default_rng(0), 200)
    assert a.shape == (200, 3) and np.array_equal(a, b)
    lo, hi = rs.percentile_interval(a)
    assert lo[0] < rs.pooled_pq(rows)[0] < hi[0]
    assert rs.bootstrap_resamples(rows, np.random.default_rng(0), 5, factors=False).shape == (5,)


def test_elongation_of_a_line_and_a_square():
    ys, xs = np.mgrid[0:10, 0:10]
    assert np.isclose(rs.elongation(xs.ravel(), ys.ravel()), 1.0)                      # isotropic
    x = np.arange(100.0)
    expected = np.sqrt((np.var(x, ddof=1) + 0.1) / 0.1)                                # one eigenvalue is 0 (+ 0.1)
    assert np.isclose(rs.elongation(x, np.zeros(100)), expected)
    assert rs.elongation(np.array([1, 2]), np.array([1, 2])) == 1.0                    # fewer than three pixels


def test_batch_groups_keep_a_batch_together_and_balance_the_load():
    batches = {"A": 3, "B": 2, "C": 2, "D": 1, "E": 1, "F": 1}
    ids_of_stem, k = {}, 0
    for b, n in batches.items():
        for _ in range(n):
            ids_of_stem[f"img{k}"] = [f"{b * 4}01-img{k}"]       # annotation-image ids look like '<batch code>-<image>'
            k += 1
    stems = sorted(ids_of_stem)
    group = rs.batch_groups(ids_of_stem, stems, n_groups=3)
    assert set(group) == set(stems) and set(group.values()) == {0, 1, 2}
    for b in batches:
        assert len({group[s] for s in stems if ids_of_stem[s][0].startswith(b * 4)}) == 1
    assert sorted(np.bincount(list(group.values()))) == [3, 3, 4]


def test_summarise_quantiles():
    assert rs.summarise_quantiles(np.arange(101.0)) == [10.0, 25.0, 50.0, 75.0, 90.0]
    assert rs.summarise_quantiles(np.arange(101.0), (25, 75)) == [25.0, 75.0]
