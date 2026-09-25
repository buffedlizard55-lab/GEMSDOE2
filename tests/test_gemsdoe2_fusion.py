"""Tests for GEMSDOE2 Dual-Component Fault Backbone + Multiscale Geophysical Fusion submission."""
import json
import zipfile
from pathlib import Path
import numpy as np
import rasterio
import pytest

ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = ROOT / "data/evidence/runs/gemsdoe2-fusion-v2"
FUSION_TIF = FUSION_DIR / "submission.tif"
FUSION_ZIP = FUSION_DIR / "submission.zip"
DOCS_TIF = ROOT / "docs/gemsdoe2_fusion_submission.tif"
DOCS_ZIP = ROOT / "docs/gemsdoe2_fusion_submission.zip"
SAMPLE_TIF = ROOT / "data/sample_submission.tif"


def test_fusion_artifacts_exist():
    assert FUSION_TIF.exists(), "GEMSDOE2 fusion GeoTIFF must exist"
    assert FUSION_ZIP.exists(), "GEMSDOE2 fusion zip must exist"
    assert DOCS_TIF.exists(), "docs/ download GeoTIFF must exist"
    assert DOCS_ZIP.exists(), "docs/ download zip must exist"


def test_docs_and_evidence_are_identical():
    assert DOCS_TIF.read_bytes() == FUSION_TIF.read_bytes()
    assert DOCS_ZIP.read_bytes() == FUSION_ZIP.read_bytes()


def test_fusion_tif_template_conformance():
    with rasterio.open(FUSION_TIF) as src:
        assert src.count == 1, "Must be single-band"
        assert src.dtypes[0] == "float32", "Must be float32"
        assert src.crs.to_epsg() == 32611, "Must be EPSG:32611"
        assert src.res == (100.0, 100.0), "Resolution must be 100m"
        assert (src.width, src.height) == (3292, 3730), "Dimensions must match template"
        arr = src.read(1)

    if SAMPLE_TIF.exists():
        with rasterio.open(SAMPLE_TIF) as smp:
            smp_arr = smp.read(1)
            valid_mask = ~np.isnan(smp_arr)
            # NaN placement must match sample submission exactly
            assert np.array_equal(np.isnan(arr), np.isnan(smp_arr)), "NaN positions must match template"
            # Valid pixels must be strictly finite and in [0, 1]
            valid_vals = arr[valid_mask]
            assert np.all(np.isfinite(valid_vals)), "All scored pixels must be finite (no NaNs in valid mask)"
            assert float(valid_vals.min()) >= 0.0, "Values must be >= 0.0"
            assert float(valid_vals.max()) <= 1.0, "Values must be <= 1.0"
            assert int(valid_mask.sum()) == 5167373, "Scored pixel count must equal 5,167,373"


def test_fusion_zip_contains_single_valid_geotiff():
    with zipfile.ZipFile(FUSION_ZIP) as zf:
        infolist = zf.infolist()
        assert len(infolist) == 1, "Zip must contain exactly 1 file"
        assert infolist[0].filename == "submission.tif", "Zip member must be named submission.tif"
        data = zf.read("submission.tif")
        assert data == FUSION_TIF.read_bytes(), "Zip member must match submission.tif bit for bit"


def test_fusion_metrics_and_standing():
    rep_p = FUSION_DIR / "fusion_report.json"
    assert rep_p.exists()
    rep = json.loads(rep_p.read_text())
    assert rep["metrics"]["dti_known_faults"] > 0.60
    assert rep["metrics"]["dti_proxy_faults"] > 0.10
    assert rep["metrics"]["dti_union_truth"] > 0.45
    assert rep["counts"]["nonzero_px"] > 200000
