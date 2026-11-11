"""Access to the stored intermediate results (artefacts) that let the notebook run without a GPU.

The artefacts are part of the repository (folder ``artifacts/``, plain git files). :func:`fetch_artifact` returns the path
of a file after checking it against the sha256 sum in ``artifacts/MANIFEST.json``. There is no download step: if a file
is missing, the clone is incomplete.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("PQTREE_ARTIFACTS", REPO_ROOT / "artifacts"))
"""Folder with the artefacts. Override with the environment variable ``PQTREE_ARTIFACTS``."""

_VERIFIED: set = set()


def sha256_file(path: str | Path) -> str:
    """sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(artifact_dir: str | Path | None = None) -> dict:
    """Read ``MANIFEST.json`` ({"files": {name: {"bytes", "sha256", "stage", "description"}}})."""
    path = Path(artifact_dir or ARTIFACT_DIR) / "MANIFEST.json"
    with open(path) as f:
        return json.load(f)


def fetch_artifact(name: str, artifact_dir: str | Path | None = None) -> Path:
    """Return the path of artefact ``name`` in the artefact folder, verified against the manifest.

    Raises ``KeyError`` for a name that is not in the manifest, ``FileNotFoundError`` if the file is missing (clone the
    complete repository) and ``RuntimeError`` if its sha256 sum does not match.
    """
    folder = Path(artifact_dir or ARTIFACT_DIR)
    manifest = load_manifest(folder)["files"]
    if name not in manifest:
        raise KeyError(f"{name!r} is not listed in MANIFEST.json")
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. The artefacts are part of the repository: clone the complete "
                                f"repository (about 360 MB) or point PQTREE_ARTIFACTS to a complete copy of artifacts/.")
    if (path, manifest[name]["sha256"]) not in _VERIFIED:
        got = sha256_file(path)
        if got != manifest[name]["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {name}: {got} != {manifest[name]['sha256']}")
        _VERIFIED.add((path, manifest[name]["sha256"]))
    return path


def verify_all(artifact_dir: str | Path | None = None) -> None:
    """Check every artefact of the manifest (existence and sha256) and print one line per file."""
    for name in load_manifest(artifact_dir)["files"]:
        fetch_artifact(name, artifact_dir)
        print("ok", name)


# --- lossless table storage ----------------------------------------------------------------------------------------
def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """Store a DataFrame column by column in a compressed ``.npz`` (exact dtypes, no pickle)."""
    cols = {}
    for i, c in enumerate(df.columns):
        a = df[c].to_numpy()
        is_text = a.dtype == object or df[c].dtype.kind in "OUT" or str(df[c].dtype) in ("str", "string")
        cols[f"c{i}"] = a.astype(str) if is_text else a
    np.savez_compressed(path, names=np.array(list(df.columns)), **cols)


def load_table(path: str | Path) -> pd.DataFrame:
    """Inverse of :func:`save_table`."""
    z = np.load(path, allow_pickle=False)
    out = {}
    for i, c in enumerate(z["names"]):
        a = z[f"c{i}"]
        out[str(c)] = pd.Series(a.astype(object), dtype="str") if a.dtype.kind == "U" else a
    return pd.DataFrame(out)


if __name__ == "__main__":
    verify_all()
