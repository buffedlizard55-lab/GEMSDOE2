"""Submission identity: what makes two submissions "the same", and what must not drift.

Two measured facts from 2026-09-16 motivate this file.

1. Two runs of the identical six fold artifacts produced submissions with bit-identical pixels but
   different file hashes.  Cause: the selected floor is written into the GeoTIFF metadata at full
   float64 precision, and `np.geomspace` rounded one ULP differently under a different numpy build
   (shaping_t0 0.4696741044002384 vs ...2383).  So: the search grid is quantised, and both the file
   hash and a pixel-only hash are recorded - a file hash alone cannot tell "the model changed" from
   "the wheel changed".

2. A container hash is what a reviewer can re-check against a submitted file, so it stays the
   primary identity; the pixel hash is the thing to compare between runs.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_io import sha256_file, sha256_pixels                # noqa: E402
from src.submission_optim import shaping_thresholds                     # noqa: E402


def _write(path, arr, tag_value):
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype="float32", tiled=True, blockxsize=16, blockysize=16)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(shaping_t0=tag_value)            # metadata differs between the two files


def test_pixel_hash_ignores_metadata_but_the_file_hash_does_not(tmp_path):
    arr = np.zeros((32, 32), np.float32)
    arr[5:9, 5:20] = 1.0
    arr[0, 0] = np.nan
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write(a, arr, "0.4696741044002384")
    _write(b, arr, "0.4696741044002383")
    assert sha256_pixels(a) == sha256_pixels(b), "pixels identical -> identity must match"
    assert sha256_file(a) != sha256_file(b), "the container hash is expected to differ here"
    # and a real pixel change must be caught by both
    arr2 = arr.copy()
    arr2[10, 10] = 1.0
    c = tmp_path / "c.tif"
    _write(c, arr2, "0.4696741044002384")
    assert sha256_pixels(c) != sha256_pixels(a)
    assert sha256_file(c) != sha256_file(a)


def test_nan_footprint_is_part_of_the_pixel_identity(tmp_path):
    arr = np.ones((16, 16), np.float32)
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write(a, arr, "x")
    arr2 = arr.copy()
    arr2[0, 0] = np.nan
    _write(b, arr2, "x")
    assert sha256_pixels(a) != sha256_pixels(b), "a footprint change must not hash the same"


def test_shaping_grid_is_quantised_and_stable():
    g = shaping_thresholds(15)
    assert g[0] == 0.0 and len(g) > 5
    for v in g:
        assert float(f"{float(v):.6g}") == float(v), f"{v!r} is not 6-significant-digit stable"
    assert np.array_equal(g, shaping_thresholds(15)), "the grid must be reproducible"
    # the values that are searched are the values that get reported/stored
    assert all(float(f"{float(v):.6g}") == float(v) for v in shaping_thresholds(45))
