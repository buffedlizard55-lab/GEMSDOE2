"""Template conformance: the check whose absence let a platform rejection through.

THE DEFECT THIS FILE PINS (2026-09-24)
--------------------------------------
A file generated from this project's own site was rejected by the DrivenData submit
dialog with the platform's exact words:

    Predicted values must be in range [0, 1]

Every *finite* value of that file was in [0, 1].  The violation was the placement of
NaN: 3,061 px inside the official sample submission's valid region (which is exactly
the labels raster's valid mask - the region the platform scores) were NaN, and NaN is
not in [0, 1].  The file also carried 1,540 finite px outside that region where the
template is NaN, and no GDAL_NODATA tag while the template declares "nan".

The fixes, each with a test below:
  * src.submission_io.conform_to_template    - the pure mask alignment
  * scripts/sanitize_submission.py           - the CLI that applies it, with evidence
  * scripts/validate_submission.py           - the gate that now fails such a file
  * the shipped artifact                     - conformed; re-checked whenever data is placed
  * docs/submission_meta.json                - strict JSON (a bare NaN token broke JSON.parse)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_io import cli_validate, conformance_findings, conform_to_template  # noqa: E402

SAMPLE = ROOT / "data" / "sample_submission.tif"
SHIPPED = ROOT / "data" / "evidence" / "runs" / "ens12-adopted-floor0.1-w0" / "submission.tif"
SANITIZE = ROOT / "scripts" / "sanitize_submission.py"
VALIDATE = ROOT / "scripts" / "validate_submission.py"


def _template(h=8, w=8, hole=True):
    """A tiny sample-submission stand-in: valid everywhere, NaN at [0:2, 0:2]."""
    t = np.ones((h, w), np.float32)
    if hole:
        t[:2, :2] = np.nan
    return t


# --------------------------------------------------------------------------- pure function
def test_conform_fills_nan_inside_the_valid_region():
    pred = np.full((8, 8), 0.25, np.float32)
    pred[3, 3] = np.nan                       # the platform-rejecting kind
    out, stats = conform_to_template(pred, _template())
    assert np.isfinite(out[3, 3]) and out[3, 3] == 0.0
    # the template's own NaN corner [0:2,0:2] gets masked on the same pass
    assert stats["filled_inside"] == 1 and stats["masked_outside"] == 4
    assert stats["unchanged"] == 64 - 1 - 4


def test_conform_masks_finite_outside_to_nan():
    pred = np.full((8, 8), 0.5, np.float32)
    out, stats = conform_to_template(pred, _template())
    assert np.isnan(out[:2, :2]).all()
    assert np.isfinite(out[2:, 2:]).all()
    assert stats["masked_outside"] == 4


def test_conform_clips_out_of_range_and_counts_it():
    pred = np.zeros((8, 8), np.float32)
    pred[4, 4] = 1.5
    pred[5, 5] = -0.2
    out, stats = conform_to_template(pred, _template())
    assert out[4, 4] == 1.0 and out[5, 5] == 0.0
    assert stats["clipped"] == 2


def test_conform_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match template"):
        conform_to_template(np.zeros((4, 4), np.float32), _template())


def test_findings_reports_without_mutating():
    pred = np.zeros((8, 8), np.float32)
    pred[3, 3] = np.nan
    pred[0, 0] = 0.5                          # finite where the template is NaN
    f = conformance_findings(pred, _template())
    assert f["nan_inside_px"] == 1
    # the template's NaN corner is finite everywhere in pred (zeros + the 0.5 probe)
    assert f["finite_outside_px"] == 4
    assert f["conformant"] is False
    # pure: the input is untouched
    assert np.isnan(pred[3, 3])


# --------------------------------------------------------------------------- the gate
def _write(path, arr, nodata=None, template_nodata=None):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", crs="EPSG:32611",
                transform=rasterio.transform.from_origin(0, 100 * arr.shape[0], 100, 100))
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
    return path


def test_validate_rejects_nan_inside_the_valid_region(tmp_path):
    """THE regression: the validator must fail the exact file shape the platform rejected."""
    sample = _write(tmp_path / "sample.tif", _template(h=16, w=16), nodata=np.nan)
    bad = np.ones((16, 16), np.float32) * 0.3
    bad[7, 7] = np.nan                        # NaN inside the scored region
    pred = _write(tmp_path / "bad.tif", bad, nodata=np.nan)
    p = subprocess.run([sys.executable, str(VALIDATE), "--pred", str(pred),
                        "--sample", str(sample), "--train", str(tmp_path / "no-train.tif")],
                       capture_output=True, text=True)
    assert p.returncode == 1, "a NaN inside the template's valid region must FAIL"
    assert "Predicted values must be in range [0, 1]" in p.stdout


def test_validate_rejects_finite_outside_and_nodata_drift(tmp_path):
    sample = _write(tmp_path / "sample.tif", _template(h=16, w=16), nodata=np.nan)
    bad = np.ones((16, 16), np.float32) * 0.3
    bad[0, 0] = 0.9                           # template NaN here -> finite outside = violation
    pred = _write(tmp_path / "bad.tif", bad, nodata=None)   # nodata drift as well
    p = subprocess.run([sys.executable, str(VALIDATE), "--pred", str(pred),
                        "--sample", str(sample), "--train", str(tmp_path / "no-train.tif")],
                       capture_output=True, text=True)
    assert p.returncode == 1
    assert "outside the template's valid region are finite" in p.stdout
    assert "Nodata tag" in p.stdout
    assert "not finite" not in p.stdout or "px inside" not in p.stdout


def test_validate_passes_a_conformed_file(tmp_path):
    sample = _write(tmp_path / "sample.tif", _template(h=16, w=16), nodata=np.nan)
    bad = np.ones((16, 16), np.float32) * 0.3
    bad[7, 7] = np.nan
    fixed, _ = conform_to_template(bad, _template(h=16, w=16))
    pred = _write(tmp_path / "fixed.tif", fixed, nodata=np.nan)
    p = subprocess.run([sys.executable, str(VALIDATE), "--pred", str(pred),
                        "--sample", str(sample), "--train", str(tmp_path / "no-train.tif")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "Validation PASSED" in p.stdout


# --------------------------------------------------------------------------- the CLI
def test_sanitize_check_mode_exit_codes(tmp_path):
    sample = _write(tmp_path / "sample.tif", _template(h=16, w=16), nodata=np.nan)
    bad = np.ones((16, 16), np.float32) * 0.3
    bad[7, 7] = np.nan
    pred = _write(tmp_path / "p.tif", bad, nodata=np.nan)
    chk = subprocess.run([sys.executable, str(SANITIZE), "--pred", str(pred),
                          "--sample", str(sample)], capture_output=True, text=True)
    assert chk.returncode == 1 and "NOT CONFORMANT" in chk.stdout
    wr = subprocess.run([sys.executable, str(SANITIZE), "--pred", str(pred),
                         "--sample", str(sample), "--write"], capture_output=True, text=True)
    assert wr.returncode == 0, wr.stdout + wr.stderr
    assert (tmp_path / "sanitize.json").exists(), "the write must record before/after evidence"
    ev = json.loads((tmp_path / "sanitize.json").read_text())
    assert ev["before"]["sha256"] != ev["after"]["sha256"]
    assert ev["changes"]["filled_inside"] == 1
    chk2 = subprocess.run([sys.executable, str(SANITIZE), "--pred", str(pred),
                           "--sample", str(sample)], capture_output=True, text=True)
    assert chk2.returncode == 0 and "CONFORMANT" in chk2.stdout


def test_sanitize_missing_input_exits_2(tmp_path):
    p = subprocess.run([sys.executable, str(SANITIZE), "--pred", str(tmp_path / "nope.tif")],
                       capture_output=True, text=True)
    assert p.returncode == 2 and "MISSING" in p.stderr


# --------------------------------------------------------------------------- shipped artifact
@pytest.mark.skipif(not (SAMPLE.exists() and SHIPPED.exists()),
                    reason="data/ or the shipped artifact is not in this checkout")
def test_the_shipped_artifact_is_template_conformant():
    """The file every route hands the platform must satisfy the check the platform applies."""
    with rasterio.open(SAMPLE) as ds:
        template = ds.read(1)
        t_nodata = ds.nodata
    with rasterio.open(SHIPPED) as ds:
        field = ds.read(1)
        nodata = ds.nodata
    f = conformance_findings(field, template)
    assert f["nan_inside_px"] == 0, f"{f['nan_inside_px']} NaN px inside the scored region"
    assert f["finite_outside_px"] == 0, "outside the template must be null/nan"
    assert f["out_of_range_px"] == 0
    def _lbl(v):
        return "nan" if isinstance(v, float) and np.isnan(v) else v
    assert _lbl(nodata) == _lbl(t_nodata), "the nodata tag must match the template's"
    # and the sidecar must be truthful about the bytes on disk
    sidecar = SHIPPED.with_name("submission.sha256")
    if sidecar.exists():
        want = hashlib.sha256(SHIPPED.read_bytes()).hexdigest()
        assert sidecar.read_text().split()[0] == want


@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped artifact not in this checkout")
def test_the_sanitation_evidence_is_committed_and_two_sided():
    """A silently mutated artifact is the failure mode this repo exists to prevent."""
    ev_path = SHIPPED.with_name("sanitize.json")
    assert ev_path.exists(), "the 2026-09-25 conformance fix must leave before/after evidence"
    ev = json.loads(ev_path.read_text())
    live = hashlib.sha256(SHIPPED.read_bytes()).hexdigest()
    assert ev["after"]["sha256"] == live, "sanitize.json must describe the bytes on disk"
    assert ev["before"]["sha256"] != ev["after"]["sha256"], "the fix must be visible"
    assert ev["changes"]["filled_inside"] == 3061
    assert ev["changes"]["masked_outside"] == 1540


# --------------------------------------------------------------------------- the payload
def test_shipped_payload_meta_is_strict_json():
    """A bare NaN token in submission_meta.json breaks JSON.parse and kills the builder
    (this happened when the artifact gained nodata=nan: the browser fetch().json() threw)."""
    meta = ROOT / "docs" / "submission_meta.json"
    if not meta.exists():
        pytest.skip("the payload is not built in this checkout")
    text = meta.read_text()

    def _explode(constant):
        raise AssertionError(f"non-JSON token {constant!r} in submission_meta.json")
    json.loads(text, parse_constant=_explode)


def test_glue_suggests_unique_name_and_note():
    """The builder must offer a per-build file name and a Note the team can tell apart."""
    js = (ROOT / "docs" / "generate_submission.js").read_text()
    assert "buildIdentity" in js
    assert "gems-submission-" in js
    assert "Copy the note" in js
    assert "Note field" in js


# ------------------------------------------------------- the runner-side gate (cli_validate)
@pytest.mark.skipif(not (SAMPLE.exists() and SHIPPED.exists()),
                   reason="competition data not placed (scripts/assemble_data_bridge.py)")
def test_runner_gate_accepts_the_shipped_artifact(capsys):
    assert cli_validate(["validate-conformant", str(SHIPPED)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["conformant"] is True
    assert out["nan_inside_px"] == 0 and out["finite_outside_px"] == 0
    assert out["nodata_ok"] is True


def test_runner_gate_refuses_a_non_conformant_file(tmp_path, capsys):
    sample = _write(tmp_path / "sample.tif", _template(h=16, w=16), nodata=np.nan)
    bad = np.ones((16, 16), np.float32) * 0.3
    bad[7, 7] = np.nan                        # THE regression shape
    pred = _write(tmp_path / "bad.tif", bad, nodata=np.nan)
    rc = cli_validate(["validate-conformant", str(pred), "--sample", str(sample)])
    assert rc == 1, "the runner gate must refuse a NaN inside the valid region"
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["conformant"] is False and out["nan_inside_px"] == 1
    assert "NOT CONFORMANT" in captured.err
    assert "sanitize_submission.py" in captured.err
