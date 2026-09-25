"""Shipping a MEASURED shaping policy, instead of the in-domain calibration's preference.

WHY THIS FILE EXISTS
--------------------
The pooled blend calibration maximises held-out DTI against `labels.tif` - the faults the catalogue
already contains.  The prize scores faults it does NOT contain (rules §1.1/§3.3, verified verbatim in
`data/evidence/rules_quotes.json`), and on that population the two criteria disagree in sign: on the
new-fault-like proxy population the session-12 extended sweep found its best hard candidate at floor
0.1 with width 0 px (proxy DTI 0.136452) while the shipped in-domain policy scores 0.041041 there
(`data/evidence/emission_decision.json`).  A policy measured on the scored-like population has to be
shippable, and it has to be *auditable*: the report must say that the calibration did not choose it,
must record what the calibration would have chosen instead, and must record where the measurement
lives.

The failure modes this file guards against are all quiet ones:
  * the override is accepted and then ignored (the submission is written from the calibrated floor);
  * the floor is adopted from one measurement and the width from another, producing a policy that
    nothing measured;
  * the report claims "calibrated" while the written bytes came from an adopted policy;
  * the GeoTIFF tags disagree with the report about what was applied.

Run: python -m pytest tests/test_adopted_policy.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write_tif(path: Path, arr: np.ndarray, crs: str = "EPSG:32611") -> None:
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=crs,
                       transform=from_origin(500000.0, 4200000.0, 100.0, 100.0)) as dst:
        dst.write(arr.astype(np.float32), 1)


def _synthetic_case(tmp_path: Path, n_folds: int = 3, h: int = 64) -> tuple[list[str], Path, Path, Path]:
    """Folds whose probability field has mass both above and below 0.5.

    A policy override can only be shown to matter if the two candidate floors emit different sets,
    so the field carries a sharp band (p ~ 0.95 on the trace) and a broad halo (p ~ 0.35 around it):
    floor 0.5 keeps the trace, floor 0.1 adds the halo.  That makes 'the override was ignored'
    detectable from the written pixels alone, not only from the report.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter
    gt = np.zeros((h, h), np.float32)
    for k in range(8, h - 8):
        gt[k, k] = 1.0
    rng = np.random.default_rng(7)
    folds = []
    for fi in range(n_folds):
        fd = tmp_path / f"fold-{fi}"
        fd.mkdir()
        core = gaussian_filter(gt, 0.7)
        core /= max(float(core.max()), 1e-6)                      # peak 1.0 on the trace
        halo = gaussian_filter(binary_dilation(gt > 0.5, iterations=4).astype(np.float32), 2.0)
        halo /= max(float(halo.max()), 1e-6)
        prob = np.clip(0.05 + 0.85 * core + 0.25 * halo * (core < 0.5)
                       + rng.normal(0, 0.01, (h, h)), 0.0, 1.0).astype(np.float32)
        _write_tif(fd / "prob_raw.tif", prob)
        np.savez_compressed(fd / f"heldout_mc{fi}.npz",
                            pred=prob[8:48, 8:48].astype(np.float16),
                            gt=gt[8:48, 8:48].astype(np.uint8))
        (fd / "manifest.json").write_text(json.dumps({"models": [{"file": f"m{fi}.pt", "dti": 0.4}]}))
        folds.append(str(fd))
    _write_tif(tmp_path / "sample.tif", np.zeros((h, h), np.float32))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training: {alpha: 0.2, beta: 0.8}\n"
                   "metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, "
                   "epsilon: 1.0e-7}\n")
    return folds, tmp_path / "sample.tif", tmp_path / "cfg.yaml", tmp_path / "evidence.json"


def _blend(tmp_path: Path, folds: list[str], sample: Path, cfg: Path, name: str,
           extra: list[str]) -> tuple[subprocess.CompletedProcess, dict | None, Path]:
    out = tmp_path / f"{name}.tif"
    rep = tmp_path / f"{name}_report.json"
    ens = tmp_path / f"{name}_ensemble.tif"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", *folds, "--config", str(cfg), "--sample", str(sample),
         "--out", str(out), "--report", str(rep), "--save-ensemble", str(ens),
         "--dilate-grid", "0,1,2", *extra],
        capture_output=True, text=True, timeout=900)
    report = json.loads(rep.read_text()) if rep.exists() else None
    return r, report, ens


# --------------------------------------------------------------------------- 1
def test_adopted_policy_is_what_is_written(tmp_path):
    folds, sample, cfg, evidence = _synthetic_case(tmp_path)
    evidence.write_text(json.dumps({"best_measured_candidate": {"policy": "sweep_best_t0_0.1_width0px"}}))

    # Adopt a floor the in-domain calibration would NOT pick on this field: the adopted 0.1 keeps
    # the broad halo around the trace, the calibrated ~0.36 keeps only the trace.  A test that
    # adopts a floor the calibration agrees with cannot detect an ignored flag.
    r, rep, ens = _blend(tmp_path, folds, sample, cfg, "adopted",
                         ["--shaping-t0", "0.1", "--shaping-dilate", "0", "--shaping-source", str(evidence)])
    assert r.returncode == 0, r.stdout + r.stderr
    sh = rep["shaping"]

    # (a) the report says the calibration did NOT choose this, and what it would have chosen
    assert sh["source"] == "adopted_measured_policy", sh
    assert sh["t0"] == 0.1 and sh["dilate"] == 0 and sh["thin"] is True
    assert sh["adopted"]["evidence"] == str(evidence)
    control = sh["in_domain_calibration_control"]
    assert control["t0"] is not None and control["dilate"] is not None
    assert abs(float(control["t0"]) - 0.1) > 1e-9, \
        "the fixture is not exercising the override: the calibration picked the adopted floor"
    assert "ADOPTED measured policy" in r.stdout

    # (b) the written pixels are exactly the adopted policy applied to the saved ensemble mean -
    #     recomputed here from the artifact, so 'the flag was ignored' cannot pass
    from src.submission_optim import optimize_submission
    with rasterio.open(ens) as src:
        mean = src.read(1)
    with rasterio.open(tmp_path / "adopted.tif") as src:
        written = src.read(1)
        tags = src.tags()
    want = optimize_submission(mean, R=3, t0=0.1, thin=True, hard=True, gamma=1.0, dilate=0)
    got = np.nan_to_num(written)
    assert np.count_nonzero(got) > 0, "the adopted policy emitted nothing on this field"
    assert np.array_equal(got, np.nan_to_num(want)), \
        "the written submission is not the adopted policy applied to this ensemble"

    # ... and the calibration's own policy would have written something DIFFERENT on this field,
    # so the equality above is evidence of the override and not of the two policies coinciding
    alt = optimize_submission(mean, R=3, t0=float(control["t0"]), thin=True, hard=True,
                              gamma=1.0, dilate=int(control["dilate"]))
    assert np.count_nonzero(np.nan_to_num(alt)) != np.count_nonzero(got), \
        "the adopted and calibrated policies emit the same set on this field - the fixture cannot " \
        "distinguish 'the flag was honoured' from 'the flag was ignored'"

    # (c) the GeoTIFF tags agree with the report
    assert tags["shaping_t0"] == "0.1"
    assert tags["shaping_dilate"] == "0"
    assert tags["shaping_source"] == "adopted_measured_policy"


# --------------------------------------------------------------------------- 2
def test_default_path_still_says_calibrated(tmp_path):
    """No override -> the report must not claim an adopted policy (the control path is unchanged)."""
    folds, sample, cfg, _ = _synthetic_case(tmp_path)
    r, rep, _ = _blend(tmp_path, folds, sample, cfg, "calibrated", [])
    assert r.returncode == 0, r.stdout + r.stderr
    sh = rep["shaping"]
    assert sh["source"] == "pooled_in_domain_calibration"
    assert sh["adopted"] is None
    assert sh["t0"] == sh["in_domain_calibration_control"]["t0"]
    assert sh["dilate"] == sh["in_domain_calibration_control"]["dilate"]


# --------------------------------------------------------------------------- 3
def test_joint_policy_is_enforced_at_parse_time(tmp_path):
    """A floor from one measurement with a width from another is not a measured policy."""
    folds, sample, cfg, evidence = _synthetic_case(tmp_path, n_folds=2)

    def run(extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
             "--folds", *folds, "--config", str(cfg), "--sample", str(sample),
             "--out", str(tmp_path / "x.tif"), "--report", str(tmp_path / "x.json"), *extra],
            capture_output=True, text=True, timeout=900)

    only_floor = run(["--shaping-t0", "0.1"])
    assert only_floor.returncode != 0 and "must be given together" in (only_floor.stderr + only_floor.stdout)

    only_width = run(["--shaping-dilate", "0"])
    assert only_width.returncode != 0 and "must be given together" in (only_width.stderr + only_width.stdout)

    source_only = run(["--shaping-source", str(evidence)])
    assert source_only.returncode != 0 and "--shaping-t0/--shaping-dilate" in \
        (source_only.stderr + source_only.stdout)

    # and the legal combination still runs
    both = run(["--shaping-t0", "0.5", "--shaping-dilate", "0"])
    assert both.returncode == 0, both.stdout + both.stderr
