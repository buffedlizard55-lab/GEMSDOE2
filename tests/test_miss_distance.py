"""The localization/detection decomposition must measure what it claims.

`scripts/measure_miss_distance.py` exists to answer a question the width sweeps cannot: are the
misses on the new-fault-like population *near* misses (a wider emitted band would convert them
into credit) or *far* misses (no plausible width reaches them, so the model needs better
detection)?  Its first real run (2026-09-17, session 10) found a 22 px median miss distance and an
interior optimum at a 16 px band - a bigger effect than any floor/thinning change measured so far -
so the measurement itself has to be pinned against geometry whose answer is known by construction.

These tests build a raster where the truth and the prediction are parallel lines at a known
separation, so every reported number has an exact expected value.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "measure_miss_distance.py"
CODE_ONLY = 2


def _write(path: Path, arr: np.ndarray, dtype: str) -> None:
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype=dtype,
                       crs="EPSG:32611", transform=from_origin(0, h * 100, 100, 100)) as dst:
        dst.write(arr.astype(dtype), 1)


def _scene(tmp: Path, truth_row: int = 32, pred_row: int = 40, n: int = 64,
           truth_codes: bool = True) -> tuple[Path, Path]:
    """Truth: a 50-px horizontal line.  Prediction: a parallel line `pred_row - truth_row` away."""
    proxy = np.zeros((n, n), np.uint8)
    proxy[truth_row, 5:55] = CODE_ONLY if truth_codes else 1
    pred = np.zeros((n, n), np.float32)
    if 0 <= pred_row < n:
        pred[pred_row, 5:55] = 1.0
    pp, pr = tmp / "proxy.tif", tmp / "pred.tif"
    _write(pp, proxy, "uint8")
    _write(pr, pred, "float32")
    return pp, pr


def _run(tmp: Path, proxy: Path, pred: Path, widths: str = "0,4,8,12,20,30",
         out: str = "out.json") -> tuple[subprocess.CompletedProcess, dict]:
    op = tmp / out
    r = subprocess.run([sys.executable, str(SCRIPT), "--pred", str(pred), "--proxy", str(proxy),
                        "--widths", widths, "--out", str(op)],
                       cwd=tmp, capture_output=True, text=True)
    return r, (json.loads(op.read_text()) if op.exists() else {})


def test_miss_distance_equals_the_known_separation(tmp_path):
    proxy, pred = _scene(tmp_path, truth_row=32, pred_row=40)          # 8 px apart
    r, d = _run(tmp_path, proxy, pred)
    assert r.returncode == 0, r.stderr
    assert d["truth_px"] == 50 and d["emitted_px"] == 50
    assert d["miss_geometry"]["percentiles_px"]["50"] == pytest.approx(8.0)
    assert d["miss_geometry"]["credited_within_R"] == 0.0              # 8 > R = 3: nothing credited
    assert d["miss_geometry"]["within_12px"] == 1.0                    # but a 12 px band reaches it
    assert d["miss_geometry"]["max_px"] == pytest.approx(8.0)


def test_width_curve_converts_near_misses_into_credit(tmp_path):
    proxy, pred = _scene(tmp_path, truth_row=32, pred_row=40)
    r, d = _run(tmp_path, proxy, pred)
    assert r.returncode == 0, r.stderr
    curve = {row["width_px"]: row for row in d["width_curve"]}
    assert curve[0]["dti"] == 0.0, "an 8 px-off line earns nothing at width 0"
    assert curve[4]["dti"] == 0.0, "4 px of band still does not reach an 8 px miss"
    assert curve[8]["dti"] > curve[4]["dti"], "an 8 px band reaches the truth"
    assert curve[8]["TP_w"] > 0
    # emission grows monotonically with the band, and the curve records it
    assert [row["emission_px"] for row in d["width_curve"]] == sorted(
        row["emission_px"] for row in d["width_curve"])
    assert d["verdict"]["best_width_px"] > 0
    assert d["verdict"]["gain_from_widening"] == pytest.approx(
        d["verdict"]["best_dti"] - d["verdict"]["shipped_width0_dti"], abs=1e-9)


def test_detection_failures_are_reported_as_unreachable(tmp_path):
    """A prediction two-thirds of the raster away is a detection failure, not a localization error."""
    proxy, pred = _scene(tmp_path, truth_row=10, pred_row=52, n=64)     # 42 px apart
    r, d = _run(tmp_path, proxy, pred, widths="0,12,20,30,40")
    assert r.returncode == 0, r.stderr
    assert d["miss_geometry"]["percentiles_px"]["50"] == pytest.approx(42.0)
    assert d["miss_geometry"]["credited_within_R"] == 0.0
    assert d["verdict"]["detection_failure_share"] == 1.0               # nothing within 12 px
    assert d["unreachable_by_widening"]["beyond_30px"] == 1.0


def test_not_every_truth_pixel_is_reachable_by_widening(tmp_path):
    """A half-covered truth: the reachable share must be the covered half, not 'all of it'."""
    proxy = np.zeros((64, 64), np.uint8)
    proxy[20, 5:30] = CODE_ONLY          # reachable: predicted line is 4 px away
    proxy[50, 5:30] = CODE_ONLY          # unreachable at 4 px of width
    pred = np.zeros((64, 64), np.float32)
    pred[24, 5:30] = 1.0
    pp, pr = tmp_path / "proxy.tif", tmp_path / "pred.tif"
    _write(pp, proxy, "uint8")
    _write(pr, pred, "float32")
    r, d = _run(tmp_path, pp, pr, widths="0,4,8")
    assert r.returncode == 0, r.stderr
    assert d["truth_px"] == 50
    assert d["miss_geometry"]["within_6px"] == pytest.approx(0.5)       # only the near line
    assert d["miss_geometry"]["within_12px"] == pytest.approx(0.5)


def test_refuses_to_report_on_an_empty_prediction(tmp_path):
    """No emitted pixels -> every miss is a detection failure by definition; say so, do not score."""
    proxy, _ = _scene(tmp_path, truth_row=32, pred_row=999, n=64)
    empty = tmp_path / "empty.tif"
    _write(empty, np.zeros((64, 64), np.float32), "float32")
    r, _ = _run(tmp_path, proxy, empty)
    assert r.returncode != 0
    assert "emits nothing" in (r.stdout + r.stderr)


def test_truth_mode_only_excludes_faults_the_labels_cover(tmp_path):
    """code 1 = already in the labels, code 2 = absent: --truth only must use code 2 alone."""
    proxy = np.zeros((64, 64), np.uint8)
    proxy[10, 5:25] = 1        # near a labelled fault
    proxy[30, 5:25] = CODE_ONLY
    pred = np.zeros((64, 64), np.float32)
    pred[30, 5:25] = 1.0
    pp, pr = tmp_path / "proxy.tif", tmp_path / "pred.tif"
    _write(pp, proxy, "uint8")
    _write(pr, pred, "float32")
    r, d = _run(tmp_path, pp, pr, widths="0,4")
    assert r.returncode == 0, r.stderr
    assert d["inputs"]["truth_mode"] == "only"
    assert d["truth_px"] == 20                      # the code-1 line is excluded
    assert d["miss_geometry"]["credited_within_R"] == 1.0


def test_oracle_ceiling_bounds_every_policy_of_that_width(tmp_path):
    """The ceiling is arithmetic about the METRIC, so it must come out exactly.

    A perfect localizer that emits the truth scores 1.0.  A perfect localizer that still writes a
    band of half-width w pays FP_w = sum over band pixels of d(x)/R (at p = 1) while TP_w = |G|, so
    the ceiling is 1/(1 + 0.2*FP_w/|G|) - independent of how large the hidden truth is, which is the
    only reason it can be quoted for a truth set we cannot see.  On a 50 px line, three of whose
    rows are within R, the numbers are exact.
    """
    proxy, pred = _scene(tmp_path, truth_row=32, pred_row=40)
    op = tmp_path / "oracle.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--pred", str(pred), "--proxy", str(proxy),
                        "--widths", "0,1,3", "--out", str(op), "--oracle"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(op.read_text())
    ceil = {row["width_px"]: row for row in d["oracle_band_ceiling"]["rows"]}
    assert ceil[0]["oracle_dti"] == 1.0, "emitting the truth itself is a perfect score"
    # Width 1 on a 50 px horizontal line, disk(1) being the 4-connected cross: the band is the two
    # neighbouring rows (50 px each, d = 1 -> fp 1/3 each) plus the two end pixels of the middle row
    # (d = 1 -> 1/3 each, the part of the cost a finite line pays that a long one does not).
    fp_per_px = (2 * 50 * (1 / 3) + 2 * (1 / 3)) / 50
    assert ceil[1]["fp_per_truth_px"] == pytest.approx(fp_per_px, abs=1e-6)
    assert ceil[1]["oracle_dti"] == pytest.approx(1 / (1 + 0.2 * fp_per_px), rel=1e-6)
    # and the ceiling must dominate the measured policy at the same width (it is an upper bound)
    measured = {row["width_px"]: row["dti"] for row in d["width_curve"]}
    for w, row in ceil.items():
        assert measured[w] <= row["oracle_dti"] + 1e-9, f"measured DTI exceeds the ceiling at {w} px"
    assert d["oracle_band_ceiling"]["truth_size_independent"] is True
    # without the flag nothing is computed, so the evidence file cannot imply a measurement happened
    r2, d2 = _run(tmp_path, proxy, pred, out="plain.json")
    assert r2.returncode == 0
    assert d2["oracle_band_ceiling"] is None
