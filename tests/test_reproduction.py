"""Reproduction test: the stored-artefact path must write exactly the reference submission.

Needs the artefacts (see README); set PQTREE_ARTIFACTS if they are not in ./artifacts. Skipped if artifacts/MANIFEST.json
is missing (incomplete clone).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pqtree import pipeline  # noqa: E402
from pqtree.artifacts import ARTIFACT_DIR  # noqa: E402

REFERENCE = ROOT / "reference" / "submission.csv"
REFERENCE_SHA256 = "eab570adfc6309f73db548636041f1dbb8ebb867652dcb90f8fd43ed74b4953a"


@pytest.mark.skipif(not (ARTIFACT_DIR / "MANIFEST.json").exists(), reason="artefacts not available")
def test_default_path_reproduces_reference(tmp_path):
    assert pipeline.sha256_of(REFERENCE) == REFERENCE_SHA256, "reference file was modified"
    out = tmp_path / "submission.csv"
    new = pipeline.default_submission(out)
    ref = pd.read_csv(REFERENCE)
    assert list(new.filament_id) == list(ref.filament_id)
    assert (new.segmentation_rle.values == ref.segmentation_rle.values).all()
    assert pipeline.sha256_of(out) == REFERENCE_SHA256
    assert new.filament_id.str.rsplit("_", n=1).str[0].nunique() == 180
