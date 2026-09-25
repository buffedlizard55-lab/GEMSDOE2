"""Audits the selected Recall-Union v1 artifact against its declared rule."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/evidence/runs/recall-union-v1/recall_union_report.json"
PUBLIC = ROOT / "docs/gemsdoe2_recall_union_submission.tif"
EVIDENCE = ROOT / "data/evidence/runs/recall-union-v1/submission.tif"
DEEP = ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
CLASSICAL = ROOT / "data/evidence/baseline/submission.tif"
SAMPLE = ROOT / "data/sample_submission.tif"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recall_union_public_and_evidence_bytes_match_report():
    report = json.loads(REPORT.read_text())
    expected = report["artifact"]["sha256"]
    assert _sha(PUBLIC) == expected
    assert _sha(EVIDENCE) == expected
    assert PUBLIC.read_bytes() == EVIDENCE.read_bytes()
    assert report["provenance"]["known_labels_used_to_generate_field"] is False
    assert report["provenance"]["component_filter"].startswith("none")


def test_recall_union_pixels_are_exact_source_union_inside_template():
    with rasterio.open(SAMPLE) as src:
        template = src.read(1)
    with rasterio.open(DEEP) as src:
        deep = src.read(1)
    with rasterio.open(CLASSICAL) as src:
        classical = src.read(1)
    with rasterio.open(PUBLIC) as src:
        output = src.read(1)

    valid = np.isfinite(template)
    expected = valid & (
        (np.isfinite(deep) & (deep > 0.0))
        | (np.isfinite(classical) & (classical > 0.0))
    )
    assert np.array_equal(np.isfinite(output), valid)
    assert np.array_equal(output[valid] > 0.0, expected[valid])
    assert np.all(np.isfinite(output[valid]))
    assert set(np.unique(output[valid]).tolist()) <= {0.0, 1.0}
