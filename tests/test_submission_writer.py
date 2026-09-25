"""Regression tests for the submission writer (defect of Actions run 35042805806).

The failed run's blend job copied `sample_submission.tif`'s rasterio profile and forced
``TILED="YES"``.  The competition template is a STRIPED GeoTIFF, so its profile carries
``blockxsize == width == 3292``; GDAL refuses to interpret that as a TIFF tile width:

    RasterBlockError: The height and width of TIFF dataset blocks must be multiples of 16

Nothing was written, the job still went green (the crash was piped through `tee` without
`pipefail`) and a 110-byte stub was committed as `submission.tif`.

These tests pin the properties that make that failure impossible to repeat silently.
Run: python tests/test_submission_writer.py   (or python -m pytest tests -q)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_io import clean_profile, write_submission  # noqa: E402

WIDTH = 3292          # == competition grid width, and NOT a multiple of 16
HEIGHT = 8


def _striped_template(path: Path) -> dict:
    """The competition template's storage layout: striped, blockxsize == width."""
    with rasterio.open(path, "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=1,
                       dtype="float32", crs="EPSG:32611", nodata=None, compress="lzw",
                       transform=rasterio.transform.from_origin(300000, 4500000, 100, 100)) as dst:
        dst.write(np.zeros((HEIGHT, WIDTH), np.float32), 1)
    with rasterio.open(path) as src:
        return src.profile.copy()


def test_old_approach_reproduced_the_crash():
    """The defect itself: stale blockxsize + TILED=YES cannot be written."""
    with tempfile.TemporaryDirectory() as td:
        prof = _striped_template(Path(td) / "template.tif")
        assert prof["blockxsize"] == WIDTH and not prof["tiled"], "template must be striped"
        stale = dict(prof)
        stale.update(driver="GTiff", count=1, dtype="float32", TILED="YES")
        try:
            with rasterio.open(Path(td) / "out.tif", "w", **stale) as dst:
                dst.write(np.ones((HEIGHT, WIDTH), np.float32), 1)
        except rasterio.errors.RasterBlockError as e:
            assert "multiples of 16" in str(e)
            return
        raise AssertionError("expected RasterBlockError - if GDAL relaxed this, update the "
                             "comment in src/submission_io.py rather than deleting the test")


def test_clean_profile_drops_stale_block_geometry_and_pins_legal_tiles():
    with tempfile.TemporaryDirectory() as td:
        src_prof = _striped_template(Path(td) / "template.tif")
        prof = clean_profile(src_prof, dtype="float32")
        assert prof["blockxsize"] != src_prof["blockxsize"], "stale strip width leaked in"
        assert prof["tiled"] is True
        assert prof["blockxsize"] % 16 == 0 and prof["blockysize"] % 16 == 0
        # georeferencing intent is preserved
        assert prof["crs"] == src_prof["crs"] and prof["transform"] == src_prof["transform"]


def test_writer_succeeds_on_the_actual_competition_width():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src_prof = _striped_template(td / "template.tif")
        prof = clean_profile(src_prof, dtype="float32")
        arr = np.zeros((HEIGHT, WIDTH), np.float32)
        arr[2:4, 100:140] = 0.75
        info = write_submission(td / "submission.tif", arr, prof)
        assert info["bytes"] > 1000
        assert info["width"] == WIDTH and info["height"] == HEIGHT
        assert info["dtype"] == "float32" and info["count"] == 1
        assert info["nonzero_px"] == 2 * 40
        assert all(b % 16 == 0 for blk in info["block_shapes"] for b in blk), info["block_shapes"]
        with rasterio.open(td / "submission.tif") as src:
            back = src.read(1)
        assert np.allclose(np.nan_to_num(back), arr), "round-trip must be lossless for float32"


def test_writer_refuses_degenerate_or_illegal_output():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        prof = clean_profile(None, height=32, width=32, crs="EPSG:32611",
                             transform=rasterio.transform.from_origin(0, 0, 100, 100))
        # all-zero raster == the "total fault absence" template, never a prediction
        try:
            write_submission(td / "zero.tif", np.zeros((32, 32), np.float32), prof)
        except RuntimeError as e:
            assert "all zeros" in str(e)
        else:
            raise AssertionError("all-zero submission must raise")
        # values outside [0,1] violate the stated submission format
        bad = np.zeros((32, 32), np.float32)
        bad[0, 0] = 1.5
        try:
            write_submission(td / "bad.tif", bad, prof)
        except ValueError as e:
            assert "[0,1]" in str(e)
        else:
            raise AssertionError("out-of-range submission must raise")
        # shape/profile disagreement is a programming error, not a silent resize
        try:
            write_submission(td / "shape.tif", np.zeros((16, 16), np.float32), prof)
        except ValueError:
            pass
        else:
            raise AssertionError("profile/array mismatch must raise")


def test_blend_script_uses_the_log_spaced_shaping_grid():
    """PR #9 fixed the floor grid in src/train.py but not in scripts/blend_submission.py,
    even though the 6-fold workflow's BINDING calibration happens in the blend script."""
    text = (ROOT / "scripts" / "blend_submission.py").read_text()
    assert "shaping_thresholds(n_grid)" in text
    assert "np.linspace(0.02, 0.9" not in text, "the old un-reachable grid came back"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} tests passed")
    sys.exit(1 if fails else 0)
