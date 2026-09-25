"""Regression tests for the unsupervised structure-consensus classifier."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_structure_consensus_submission.py"


def _module():
    spec = importlib.util.spec_from_file_location("structure_consensus_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_classifier_does_not_read_magnitude_as_magnetic():
    mod = _module()
    assert mod._family(
        {
            "band_name": "geod_2ndinv",
            "data_category": "geodetic_strain",
            "description": "measure of strain rate tensor magnitude",
        },
        3,
    ) == "strain"


def test_official_feature_tags_have_six_magnetic_and_three_strain_bands():
    mod = _module()
    import rasterio

    with rasterio.open(ROOT / "data/training_features.tif") as src:
        families = [mod._family(dict(src.tags(i)), i - 1) for i in range(1, src.count + 1)]
    assert families.count("magnetic") == 6
    assert families.count("strain") == 3
