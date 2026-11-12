"""The stored-artefact path of the pipeline in one function (used by the reproduction test and the command line).

``notebooks/pipeline.ipynb`` runs the same steps one by one and can recompute every stage; this function runs only the
CPU steps that follow the stored stage outputs: join the features, predict with the stored LightGBM models, select,
write the submission.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from . import lgbm_scorer, selection
from .artifacts import fetch_artifact, load_table
from .config import TAU


def build_test_table(artifact_dir=None) -> pd.DataFrame:
    """Candidate table of the test images with all model inputs attached (stages 2-4 from the artefacts)."""
    cands = load_table(fetch_artifact("candidates_test.tbl.npz", artifact_dir))
    cnn = pd.read_csv(fetch_artifact("cnn_scores_test.csv.gz", artifact_dir))
    ev = pd.read_csv(fetch_artifact("evidence_test.csv.gz", artifact_dir))
    return lgbm_scorer.attach_features(cands, cnn, ev)


def default_submission(out_path: str | Path, artifact_dir=None, tau: float = TAU) -> pd.DataFrame:
    """Write the submission CSV from the stored artefacts and return it."""
    te = build_test_table(artifact_dir)
    model = lgbm_scorer.HurdleModel.load(fetch_artifact(lgbm_scorer.HIT_FILE, artifact_dir),
                                         fetch_artifact(lgbm_scorer.IOU_FILE, artifact_dir))
    score = model.predict(te)
    chosen = selection.select(te, score, tau)
    labels = selection.stored_label_images(fetch_artifact("test_components.npz", artifact_dir))
    rows, _ = selection.submission_rows(te, score, chosen, labels)
    return selection.write_submission(rows, out_path)


def sha256_of(path: str | Path) -> str:
    """sha256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description="Write the submission CSV from the stored artefacts.")
    ap.add_argument("out", help="output CSV path")
    ap.add_argument("--artifacts", default=None, help="artefact folder (default: ./artifacts)")
    args = ap.parse_args()
    df = default_submission(args.out, args.artifacts)
    print(f"wrote {args.out}: {len(df)} rows, sha256 {sha256_of(args.out)}")


if __name__ == "__main__":
    main()
