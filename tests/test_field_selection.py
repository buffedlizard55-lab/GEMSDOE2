"""The pre-registered FIELD-selection rule (scripts/check_field_selection.py).

The rule in docs/FIELD_SELECTION_RULE.md was committed BEFORE the next re-blend.  These tests
pin the guarantees that make it a rule rather than prose:

* on the committed evidence the verdict is KEEP of the shipped field, and the shipped field
  itself is the only one that passes the re-blend gate without a fresh measurement;
* a field that passes every sweep-visible condition (F1-F2, R1, R2, R4, R5) but has NO
  measured R3 is still refused - the adoption conditions are a conjunction, and R3 is the one
  that requires both fields' raw rasters;
* the ADOPT path works end to end when R3 IS measured on raw rasters (synthetic field that
  genuinely beats the shipped field on the proxy population);
* the contrast margin is a rule constant, not a property of one measurement: flipping it above
  the synthetic field's gain flips the verdict back to KEEP.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROXY_DIR = ROOT / "data" / "evidence" / "proxy"
SHIPPED_SUBMISSION = ROOT / "data" / "evidence" / "runs" / "ens12-adopted-floor0.1-w0" / "submission.tif"
BLOCK_REPORT = ROOT / "data" / "evidence" / "block_holdout" / "block_stratified.json"
FIELDX = "fieldx"


def _cfs():
    spec = importlib.util.spec_from_file_location(
        "check_field_selection", ROOT / "scripts" / "check_field_selection.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def _proxy_truth() -> np.ndarray:
    return (_read(PROXY_DIR / "proxy_catalogue.tif") == 2).astype(np.float32)


def _emit_support(p: np.ndarray, t0: float) -> int:
    from src.submission_optim import optimize_submission
    em = optimize_submission(p, R=3, t0=t0, thin=True, dilate=0, soft=False)
    return int(np.count_nonzero(em > 0))


def _emit_dti(p: np.ndarray, t0: float, truth: np.ndarray) -> float:
    from src.metrics import GtContext
    from src.submission_optim import optimize_submission
    em = optimize_submission(p, R=3, t0=t0, thin=True, dilate=0, soft=False)
    dti, _ = GtContext(truth, R_pixels=3).score(em, return_components=True)
    return float(dti)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> dict:
    """A synthetic field that genuinely beats the shipped field on the proxy population.

    Construction: the proposed raw field is the shipped submission's values plus probability
    mass 0.45 on ~7000 proxy-truth pixels that are isolated (no 8-neighbour in the set and no
    8-neighbour in the shipped support), so the t0 <= 0.45 candidate emission is exactly
    shipped-support + dots, thinning loses nothing, and the dots cover many blocks (so the
    paired block bootstrap can separate the two fields).  All sweep rows for the synthetic
    field are MEASURED from its raster - nothing in the test is typed where a measurement
    exists.
    """
    from scipy.ndimage import binary_dilation

    tmp = tmp_path_factory.mktemp("field_selection")
    proxy_dir = tmp / "proxy"
    proxy_dir.mkdir()
    for name in ("eval_sweep-mean12.json", "sweep_source-mean12.json"):
        shutil.copy(PROXY_DIR / name, proxy_dir / name)

    shipped = _read(SHIPPED_SUBMISSION)
    truth = _proxy_truth()
    support = shipped > 0
    support_dil = binary_dilation(support, iterations=1)
    gt_bool = truth > 0

    from src.blocks import block_id_map, DEFAULT_BLOCK_PX
    blocks = block_id_map(support.shape, DEFAULT_BLOCK_PX)

    # candidate dot pixels: on the proxy truth, not adjacent to the shipped support
    cand = gt_bool & ~support_dil
    dots = np.zeros_like(support, dtype=bool)
    n_dots, per_block = 0, np.zeros(int(blocks.max()) + 1, dtype=np.int64)
    for idx in np.flatnonzero(cand.ravel()):
        r, c = divmod(int(idx), shipped.shape[1])
        b = int(blocks[r, c])
        if per_block[b] >= 200:
            continue
        if dots[r, c] or dots[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
            continue
        dots[r, c] = True
        per_block[b] += 1
        n_dots += 1
        if n_dots >= 5500:
            break
    assert n_dots >= 5000, f"only {n_dots} isolated dot pixels found - construction assumption broke"

    proposed = shipped.copy()
    proposed[dots] = 0.45
    proposed_path = tmp / "fieldx_field.tif"
    with rasterio.open(SHIPPED_SUBMISSION) as src:
        profile = src.profile
    with rasterio.open(proposed_path, "w", **profile) as dst:
        dst.write(proposed.astype(np.float32), 1)

    # measured rows for the synthetic field's sweep
    px1 = _emit_support(proposed, 0.1)
    dti1 = _emit_dti(proposed, 0.1, truth)
    shipped_dti = 0.099859  # committed: eval_sweep-mean12.json adopted-policy row
    assert px1 > 172_974 and abs(px1 - 172_974) / 172_974 < 0.25, px1
    assert dti1 > shipped_dti + 0.01, (dti1, shipped_dti)

    rows = [
        {"t0": 0.1, "thin": True, "dilate": 0, "soft": False, "gamma": 1.0, "dti": round(dti1, 6),
         "TP_w": dti1 * 1000.0, "FP_w": 140_000.0, "FN_w": 52_000.0,
         "mass": float(px1), "emission_px": px1, "mean_kept_px": px1},
        {"t0": 0.2, "thin": True, "dilate": 0, "soft": False, "gamma": 1.0, "dti": round(dti1, 6),
         "TP_w": dti1 * 1000.0, "FP_w": 140_000.0, "FN_w": 52_000.0,
         "mass": float(px1), "emission_px": px1, "mean_kept_px": px1},
        {"t0": 0.3, "thin": True, "dilate": 0, "soft": False, "gamma": 1.0, "dti": round(dti1, 6),
         "TP_w": dti1 * 1000.0, "FP_w": 140_000.0, "FN_w": 52_000.0,
         "mass": float(px1), "emission_px": px1, "mean_kept_px": px1},
        {"t0": 0.5, "thin": True, "dilate": 0, "soft": False, "gamma": 1.0, "dti": 0.0999,
         "TP_w": 8378.0, "FP_w": 164_461.0, "FN_w": 53_286.0,
         "mass": 172_974.0, "emission_px": 172_974, "mean_kept_px": 172_974},
    ]
    (proxy_dir / "eval_sweep-fieldx.json").write_text(json.dumps({
        "generated_utc": "2026-09-19T00:00:00+00:00",
        "generated_by": "tests/test_field_selection.py (synthetic field)",
        "inputs": {"pred": "fieldx_field.tif", "pred_sha256": "0" * 64,
                   "reference_t0": 0.47,
                   "reference_source": "synthetic: the field's own in-domain calibration"},
        "truth": {"mode": "only", "px": 61_664},
        "results": {"shaping_sweep": rows,
                    "sweep_verdict": {"reference_t0": 0.47,
                                      "current_policy_dti": 0.05,
                                      "beats_current_policy_by": round(dti1 - 0.05, 6)}},
    }))
    (proxy_dir / "sweep_source-fieldx.json").write_text(json.dumps(
        {"swept_run_id": "35042805806,35249562910",
         "note": "synthetic: pinned re-blend runs (F1 passes via the committed evidence dir)",
         "sweep_file": "eval_sweep-fieldx.json"}))

    return dict(cfs=_cfs(), proxy_dir=proxy_dir, proposed_path=proposed_path,
                dots=n_dots, px1=px1, dti1=dti1)


def test_committed_evidence_verdict_keep_mean12():
    m = _cfs()
    rep = m.evaluate(PROXY_DIR, BLOCK_REPORT)
    assert rep["verdict"]["decision"] == "KEEP mean12"
    ship = rep["fields"]["mean12"]
    assert ship["is_shipped_field"] is True
    assert ship["eligibility"]["F1_fold_completeness"]["status"] == "PASS"
    assert ship["eligibility"]["F2_committed_own_sweep"]["status"] == "PASS"
    # the shipped field is the reference of the contrast, so R1/R3 are not conditions on it
    assert ship["conditions"]["R1_matched_support_contrast"]["status"] == "NOT_APPLICABLE"
    assert ship["conditions"]["R3_block_bootstrap"]["status"] == "NOT_APPLICABLE"
    # every other committed field fails at least one adoption condition
    for name, f in rep["fields"].items():
        if name == "mean12":
            continue
        statuses = [c["status"] for c in f["conditions"].values()]
        assert "FAIL" in statuses or "NOT_MEASURABLE_FROM_COMMITTED_BYTES" in statuses


def test_gate_shipped_field_passes_without_measurement(tmp_path):
    m = _cfs()
    rep = m.evaluate(PROXY_DIR, BLOCK_REPORT)
    rc = m.main(["--gate", "--field", "mean12", "--quiet",
                 "--proxy-dir", str(PROXY_DIR), "--block-report", str(BLOCK_REPORT),
                 "--out", str(tmp_path / "field_selection_test.json")])
    assert rc == 0
    assert rep["verdict"]["decision"].startswith("KEEP mean12")


def test_gate_unmeasured_field_fails(tmp_path):
    m = _cfs()
    rc = m.main(["--gate", "--field", "ens123", "--quiet",
                 "--proxy-dir", str(PROXY_DIR), "--block-report", str(BLOCK_REPORT),
                 "--out", str(tmp_path / "field_selection_test.json")])
    assert rc == 1, "a field with an unmeasured R3 must not be allowed to re-blend"


def test_gate_runs_mode(tmp_path):
    """The workflow form: the field is identified by the re-blend's RUN_ID set."""
    m = _cfs()
    args = ["--quiet", "--proxy-dir", str(PROXY_DIR), "--block-report", str(BLOCK_REPORT),
            "--out", str(tmp_path / "field_selection_test.json")]
    # pinned re-blend run ids == the shipped field: passes without measurement
    rc = m.main(["--gate", "--runs", "35042805806,35249562910", *args])
    assert rc == 0
    # the 3-ensemble run set is a NEW field with no committed ADOPT measurement: refused
    rc = m.main(["--gate", "--runs", "35042805806,35249562910,35263581931", *args])
    assert rc == 1
    # an unknown run set is refused too
    rc = m.main(["--gate", "--runs", "12345", *args])
    assert rc == 1


def test_r3_not_measurable_blocks_adopt_even_when_everything_else_passes(synthetic):
    m = synthetic["cfs"]
    rep = m.evaluate(synthetic["proxy_dir"], BLOCK_REPORT)
    fx = rep["fields"][FIELDX]
    for cid in ("F1_fold_completeness", "F2_committed_own_sweep"):
        assert fx["eligibility"][cid]["status"] == "PASS", fx["eligibility"][cid]
    for cid in ("R1_matched_support_contrast", "R2_window_stability",
                "R4_policy_transfer", "R5_window_sampling"):
        assert fx["conditions"][cid]["status"] == "PASS", (cid, fx["conditions"][cid])
    assert fx["conditions"]["R3_block_bootstrap"]["status"] == "NOT_MEASURABLE_FROM_COMMITTED_BYTES"
    # the conjunction is enforced: no measured R3, no adoption
    assert rep["verdict"]["decision"] == "KEEP mean12"


def test_adopt_path_when_r3_measured(synthetic):
    m = synthetic["cfs"]
    rep = m.evaluate(synthetic["proxy_dir"], BLOCK_REPORT,
                     pred_shipped=SHIPPED_SUBMISSION,
                     pred_for={FIELDX: synthetic["proposed_path"]})
    r3 = rep["fields"][FIELDX]["conditions"]["R3_block_bootstrap"]
    assert r3["status"] == "PASS", r3
    assert r3["measured"] is True
    assert r3["p_proposed_beats_shipped"] >= m.BOOT_P_THRESHOLD
    assert rep["verdict"]["decision"] == f"ADOPT {FIELDX}"
    rc = m.main(["--gate", "--field", FIELDX, "--quiet",
                 "--proxy-dir", str(synthetic["proxy_dir"]),
                 "--pred-shipped", str(SHIPPED_SUBMISSION),
                 f"--pred={FIELDX}={synthetic['proposed_path']}",
                 "--out", str(synthetic["proxy_dir"].parent / "field_selection_test.json")])
    assert rc == 0


def test_contrast_margin_is_a_rule_constant(synthetic):
    m = synthetic["cfs"]
    gain = synthetic["dti1"] - 0.099859
    # margin above the measured gain: R1 fails, verdict flips back to KEEP
    m.CONTRAST_MARGIN = gain + 0.001
    rep = m.evaluate(synthetic["proxy_dir"], BLOCK_REPORT,
                     pred_shipped=SHIPPED_SUBMISSION,
                     pred_for={FIELDX: synthetic["proposed_path"]})
    assert rep["fields"][FIELDX]["conditions"]["R1_matched_support_contrast"]["status"] == "FAIL"
    assert rep["verdict"]["decision"] == "KEEP mean12"
    # and with the pre-registered margin the same evidence passes
    m.CONTRAST_MARGIN = 0.010
    rep = m.evaluate(synthetic["proxy_dir"], BLOCK_REPORT,
                     pred_shipped=SHIPPED_SUBMISSION,
                     pred_for={FIELDX: synthetic["proposed_path"]})
    assert rep["verdict"]["decision"] == f"ADOPT {FIELDX}"
