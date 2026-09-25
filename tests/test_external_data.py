"""Tests for the optional, explicitly-provenanced 3DEP DEM feature path."""
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.external_data import augment_with_dem_features, augment_from_dem_path  # noqa: E402


def test_dem_derivatives_append_five_bands_and_preserve_nodata():
    y, x = np.mgrid[:32, :32]
    dem = (1000.0 + 2.0 * x + 0.5 * y).astype(np.float32)
    dem[:2, :3] = np.nan
    features = np.zeros((32, 32, 2), np.float32)
    out = augment_with_dem_features(features, dem=dem, resolution=100)
    assert out.shape == (32, 32, 7)
    assert np.isfinite(out[10:, :, 2:]).all()
    assert np.isnan(out[:2, :3, 2:]).all()
    # The original feature stack is not modified and is copied into the prefix.
    assert np.array_equal(out[..., :2], features)


def test_dem_path_is_reprojected_to_the_feature_grid(tmp_path):
    dem_path = tmp_path / "dem.tif"
    transform = from_origin(500000, 4100000, 100, 100)
    dem = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=16, width=16, count=1,
        dtype="float32", crs="EPSG:32611", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(dem, 1)
    X = np.zeros((16, 16, 3), np.float32)
    out = augment_from_dem_path(
        X, dem_path, {"crs": "EPSG:32611", "transform": transform}, resolution=100,
    )
    assert out.shape == (16, 16, 8)
    assert np.isfinite(out[..., 3:]).all()


def test_optional_dem_requires_a_real_local_path():
    from src.dataset import _maybe_add_external_dem
    try:
        _maybe_add_external_dem(
            np.zeros((4, 4, 1), np.float32),
            {"crs": "EPSG:32611", "transform": from_origin(0, 0, 100, 100)},
            use_external_dem=True,
            external_dem_path=None,
        )
    except FileNotFoundError as exc:
        assert "external_dem_path" in str(exc)
    else:
        raise AssertionError("enabled external DEM path must not silently fall back")
