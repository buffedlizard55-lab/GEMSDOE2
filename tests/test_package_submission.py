"""scripts/package_submission.py — the .zip the submit dialog also accepts, and what it must refuse.

The platform's dialog reads "a single-band GeoTIFF (.tif) file, or a .zip file containing a single
GeoTIFF" (transcribed by the team; the account-gated page is not fetchable from the sandbox, and the
rules PDF §3.2 states the GeoTIFF form only — flagged as irregularity 7 on docs/submission.html).
So the archive has exactly one job: hold that one file, unchanged. These tests pin that, plus the
three ways it could silently fail: an extra member, a recompressed member nobody compared, and a
packager that is not deterministic between runs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.io import MemoryFile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/package_submission.py"
ARTIFACT = ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"


def _py():
    return os.environ.get("GEMS_PYTHON", sys.executable)


def _small_tif(path: Path) -> Path:
    """A real GeoTIFF on the competition's CRS/resolution, small enough to test fast."""
    a = np.zeros((40, 50), dtype="float32")
    a[10:14, 5:40] = 1.0
    a[0:3, :] = np.nan
    prof = dict(driver="GTiff", height=40, width=50, count=1, dtype="float32",
                crs="EPSG:32611", transform=rasterio.transform.from_origin(243350.0, 4508550.0, 100.0, 100.0),
                nodata="nan", compress="deflate")
    with rasterio.open(path, "w", **prof) as src:
        src.write(a, 1)
    return path


def test_packaging_a_tif_yields_a_single_member_archive_that_reads_back(tmp_path):
    tif = _small_tif(tmp_path / "submission.tif")
    out = tmp_path / "submission.zip"
    p = subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tif), "--out", str(out), "--json"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    rep = json.loads(p.stdout)
    assert rep["ok"] is True, rep
    assert rep["names"] == ["submission.tif"]
    assert rep["member_bytes"] == tif.stat().st_size
    assert rep["method_stored"] is True, "stored by default: the raster is already compressed"
    assert rep["deterministic"] is True, "re-packaging must give identical bytes, or 'reproducible' is a claim"
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert z.read("submission.tif") == tif.read_bytes()


def test_check_mode_catches_a_tampered_member(tmp_path):
    tif = _small_tif(tmp_path / "submission.tif")
    out = tmp_path / "submission.zip"
    subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tif), "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    # rewrite the archive with a DIFFERENT raster under the same member name. (An earlier draft of
    # this test wrote a second copy of the same array, which is byte-identical, so it proved nothing.)
    other = tmp_path / "other.tif"
    prof = dict(driver="GTiff", height=40, width=50, count=1, dtype="float32",
                crs="EPSG:32611",
                transform=rasterio.transform.from_origin(243350.0, 4508550.0, 100.0, 100.0),
                nodata="nan", compress="deflate")
    with rasterio.open(other, "w", **prof) as src:
        src.write(np.ones((40, 50), dtype="float32"), 1)
    with zipfile.ZipFile(tmp_path / "evil.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(other, arcname="submission.tif")
    p = subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tif),
                        "--out", str(tmp_path / "evil.zip"), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 1, p.stdout
    assert "byte-equal" in p.stdout


def test_check_mode_refuses_a_second_member(tmp_path):
    tif = _small_tif(tmp_path / "submission.tif")
    extra = tmp_path / "notes.txt"
    extra.write_text("a second member is not allowed by the dialog's wording\n")
    out = tmp_path / "two.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.write(tif, arcname="submission.tif")
        z.write(extra, arcname="README.txt")
    p = subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tif), "--out", str(out), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 1, p.stdout
    assert "exactly one member" in p.stdout


def test_deflate_method_is_recorded_and_check_enforces_stored(tmp_path):
    """--method deflate is allowed (smaller archive), but --check on a stored expectation must say
    when the archive recompressed, so the two routes cannot drift silently."""
    tif = _small_tif(tmp_path / "submission.tif")
    out = tmp_path / "d.zip"
    p = subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tif), "--out", str(out),
                        "--method", "deflate", "--json"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-600:]
    rep = json.loads(p.stdout)
    assert rep["method"] == "deflate"
    assert rep["member_bytes"] == tif.stat().st_size          # uncompressed size recorded correctly
    with zipfile.ZipFile(out) as z:
        assert z.getinfo("submission.tif").compress_type == zipfile.ZIP_DEFLATED
        assert z.read("submission.tif") == tif.read_bytes()
    bad = subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tif), "--out", str(out),
                          "--check", "--method", "stored"], cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode == 1 and "store" in bad.stdout


def test_missing_input_is_reported_not_fabricated(tmp_path):
    p = subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tmp_path / "nope.tif"),
                        "--out", str(tmp_path / "x.zip")], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 2
    assert "MISSING" in p.stderr
    assert not (tmp_path / "x.zip").exists()


@pytest.mark.skipif(not ARTIFACT.exists(), reason="the adopted artifact is not in this checkout")
def test_the_committed_artifact_packages_and_verifies(tmp_path):
    out = tmp_path / "submission.zip"
    p = subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(ARTIFACT), "--out", str(out),
                        "--json"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-400:]
    rep = json.loads(p.stdout)
    seen = rep["gdal_sees_after_unzip"]
    assert isinstance(seen, dict), seen
    assert seen["epsg"] == 32611 and seen["count"] == 1 and seen["dtype"] == "float32"
    assert seen["width"] == 3292 and seen["height"] == 3730
    # the invariant: the zip's member IS the artifact's bytes - re-hashed here, never typed
    # (the artifact was conformed to the template on 2026-09-25; a typed pin would rot)
    import hashlib
    want = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert rep["member_sha256"] == want
    sidecar = ARTIFACT.with_name("submission.sha256")
    if sidecar.exists():
        assert sidecar.read_text().split()[0] == want, "sidecar must match the artifact"


@pytest.mark.skipif(not (ROOT / "docs" / "geotiff_writer.js").exists(), reason="writer not built")
def test_the_browser_zip_and_the_cli_zip_agree_on_container_semantics(tmp_path):
    """Two routes write the archive (docs/geotiff_writer.js buildZip, this script). They must agree
    on the three things that matter to the platform: one member, named submission.tif, stored."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on PATH: the browser zip route cannot be run here")
    meta = ROOT / "docs" / "submission_meta.json"
    blob = ROOT / "docs" / "submission_field.bin"
    tif, zipped = tmp_path / "b.tif", tmp_path / "b.zip"
    subprocess.run([node, str(ROOT / "docs" / "geotiff_writer.js"), str(meta), str(blob),
                    str(tif), str(zipped)], cwd=ROOT, check=True, capture_output=True)
    cli_zip = tmp_path / "cli.zip"
    subprocess.run([_py(), str(ROOT / SCRIPT), "--tif", str(tif), "--out", str(cli_zip)],
                   cwd=ROOT, check=True, capture_output=True)
    with zipfile.ZipFile(zipped) as a, zipfile.ZipFile(cli_zip) as b:
        ia, ib = a.getinfo("submission.tif"), b.getinfo("submission.tif")
        assert a.namelist() == b.namelist() == ["submission.tif"]
        assert ia.compress_type == ib.compress_type == zipfile.ZIP_STORED
        assert ia.date_time == ib.date_time, "both stamp the fixed 2020-01-01 instant"
        assert a.read("submission.tif") == b.read("submission.tif") == tif.read_bytes()
