"""Regression tests for the parallel MC-ensemble path (2026-09-15).

Covers the three pieces that only exist for the runner-scale ensemble:
  1. src.train._compact_window_subset - the calibration crop must be compact and
     fault-bearing, otherwise the shaping search scores (and pays EDT for) the
     whole raster.
  2. src.inference --raw flag parsing/wiring (no postprocess, no shaping).
  3. scripts/blend_submission.py end-to-end on two synthetic folds: grid, range,
     NaN footprint, report internals and the metric closed-form identity.

Fast: numpy/rasterio only except the inference import (torch).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- 1
def test_compact_window_subset_picks_contiguous_faulty_run():
    from src.train import _compact_window_subset

    patch = 16
    tw = [(0, j * patch) for j in range(10)]                      # one row, 10 windows
    gt = np.zeros((10, patch, patch), np.float32)
    gt[0, 8, 8] = 1.0                                             # only window 0 has fault
    sel = _compact_window_subset(tw, gt, nmax=3, patch=patch)
    assert len(sel) == 3
    assert (0, 0) in sel                                          # the fault-bearing window is in
    rows = {i for i, _ in sel}
    cols = sorted(j for _, j in sel)
    assert rows == {0}                                            # bbox stays on one row -> compact
    assert cols[-1] - cols[0] <= 2 * patch                          # contiguous, not scattered

    # no fault anywhere -> deterministic fallback to the first nmax
    sel2 = _compact_window_subset(tw, np.zeros_like(gt), nmax=3, patch=patch)
    assert sel2 == tw[:3]


# --------------------------------------------------------------------------- 2
def test_inference_raw_flag_exists():
    src = (ROOT / "src" / "inference.py").read_text()
    assert '"--raw"' in src
    assert "args.raw" in src
    # raw must skip BOTH post-processing and shaping: the `if args.raw:` branch
    # contains no postprocess call, and shaping is gated by `not args.raw`.
    raw_branch = src.split('if args.raw:', 1)[1].split("else:", 1)[0]
    assert "postprocess_pipeline" not in raw_branch
    assert 'if not args.raw and (shp.get("t0")' in src


def test_inference_uses_the_fail_loud_submission_writer():
    """The single-model inference path must have the same TIFF safeguards as blending.

    The previous ensemble fix only changed ``scripts/blend_submission.py``; direct
    ``src.inference`` still forced TILED=YES onto the striped competition template.
    Keep the two production paths on the same writer so a user following the README
    cannot recreate the 110-byte stub failure.
    """
    src = (ROOT / "src" / "inference.py").read_text()
    # the import line also carries conform_to_template since the 2026-09-25 template
    # conformance fix; both names must still come from the fail-loud writer module
    assert "from .submission_io import clean_profile, conform_to_template, write_submission" in src
    assert "write_submission(" in src
    assert 'TILED="YES"' not in src
    assert "clean_profile(source_profile" in src
    assert "conform_to_template(final, template)" in src


# --------------------------------------------------------------------------- 3
def _write_tif(path, arr, crs="EPSG:32611"):
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype="float32", crs=crs, transform=rasterio.Affine(100, 0, 0, 0, -100, 0),
                   compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def test_blend_end_to_end(tmp_path):
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):                       # diagonal fault
        gt[k, k] = 1.0
    rng = np.random.default_rng(0)

    folds = []
    for fi in range(2):
        fd = tmp_path / f"fold-{fi}"
        fd.mkdir()
        # a "model" that fires near the true fault with noise
        from scipy.ndimage import gaussian_filter, binary_dilation
        band = gaussian_filter(binary_dilation(gt > 0.5, iterations=1).astype(np.float32), 1.4)
        prob = np.clip(0.02 + 0.95 * band + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)
        prob[:4, :4] = np.nan                    # fake footprint hole
        _write_tif(fd / "prob_raw.tif", prob)
        np.savez_compressed(fd / f"heldout_mc{fi}.npz",
                            pred=prob[8:48, 8:48].astype(np.float16),
                            gt=gt[8:48, 8:48].astype(np.uint8))
        (fd / "manifest.json").write_text(json.dumps({"models": [{"file": f"m{fi}.pt", "dti": 0.5}]}))
        folds.append(str(fd))

    # the sample's validity mask is what wins (2026-09-25 template conformance): a hole
    # INSIDE it is filled - NaN there is what the platform rejects with "Predicted values
    # must be in range [0, 1]" - and finite pixels OUTSIDE it are masked to NaN.
    sample_arr = np.zeros((H, W), np.float32)
    sample_arr[:2, -2:] = np.nan                 # template says: outside the valid region
    _write_tif(tmp_path / "sample.tif", sample_arr)
    _write_tif(tmp_path / "labels.tif", gt)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""
training: {alpha: 0.2, beta: 0.8}
metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, epsilon: 1.0e-7}
""")
    out = tmp_path / "submission.tif"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", *folds, "--config", str(cfg),
         "--sample", str(tmp_path / "sample.tif"), "--labels", str(tmp_path / "labels.tif"),
         "--out", str(out), "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr

    with rasterio.open(out) as src:
        q = src.read(1)
        assert q.shape == (H, W)
        assert src.count == 1 and src.dtypes[0] == "float32" and src.crs.to_epsg() == 32611
    assert np.isfinite(q[:4, :4]).all(), \
        "a hole inside the template's valid region must be filled (platform rejects NaN there)"
    assert (q[:4, :4] == 0).all(), "the filled hole is 0.0 - no data, no predicted fault"
    assert np.isnan(q[:2, -2:]).all(), \
        "pixels outside the template's valid region must be NaN (spec: null or nan outside)"
    fin = q[np.isfinite(q)]
    assert (fin >= 0).all() and (fin <= 1).all()
    assert fin.max() == 1.0 and (fin > 0).sum() > 0            # shaped hard map fires somewhere

    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["n_folds"] == 2
    assert rep["submission"]["finite_px"] == int(np.isfinite(q).sum())
    # row0 (unshaped reference) + |grid| x (|thin=False| + |thin=True| x |dilate candidates|)
    # the grid is log-spaced and includes 0.0 (src/submission_optim.shaping_thresholds),
    # not the old 11-point linear 0.02-0.9 ramp
    from src.submission_optim import shaping_thresholds
    n_dil = len(rep["shaping"]["dilate_grid"])
    assert len(rep["shaping"]["calibration_table"]) == 1 + len(shaping_thresholds(11)) * (1 + n_dil)
    assert rep["shaping"]["dilate"] in rep["shaping"]["dilate_grid"]
    loc = rep["local_score_vs_known"]
    assert loc["closed_form_matches"] is True, "DTI != closed form -> identity regression"
    assert 0.0 <= loc["dti_known_faults"] <= 1.0
    # the blended+shaped map should beat the all-zeros baseline on this toy gt
    assert loc["TP_w"] > 0.0


def test_blend_refuses_all_zero_submission(tmp_path):
    """Guard added 2026-09-15: if shaping collapsed the whole map (a failure mode seen when a
    pre-transform flattens the distribution), the blend had to FAIL LOUDLY rather than write an
    empty submission.

    Updated 2026-09-16 after the shaping grid fix.  With the OLD linear grid (first candidate
    0.02) a map whose maximum is 0.01 was zeroed at EVERY candidate, so the run aborted; with
    the log-spaced grid used now, 0.0 is an explicit candidate and the search can choose it, so
    the same input produces a valid sparse submission instead of a collapse.  Both behaviours
    are checked: no all-zero submission may be written, and the chosen floor must sit BELOW the
    old grid's reach (0.02) - that is the defect this covers."""
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):
        gt[k, k] = 1.0
    fd = tmp_path / "fold-flat"
    fd.mkdir()
    flat = np.full((H, W), 0.01, np.float32)          # every pixel below every floor
    _write_tif(fd / "prob_raw.tif", flat)
    np.savez_compressed(fd / "heldout_mc0.npz", pred=flat.astype(np.float16),
                        gt=gt.astype(np.uint8))
    (fd / "manifest.json").write_text(json.dumps({"models": [{"file": "m.pt", "dti": 0.01}]}))
    _write_tif(tmp_path / "labels.tif", gt)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""
training: {alpha: 0.2, beta: 0.8}
metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, epsilon: 1.0e-7}
""")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", str(fd), "--config", str(cfg), "--labels", str(tmp_path / "labels.tif"),
         "--out", str(tmp_path / "sub.tif"), "--report", str(tmp_path / "rep.json")],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((tmp_path / "rep.json").read_text())
    t0 = rep["shaping"]["t0"]
    assert t0 < 0.02, f"floor {t0} is not below the old grid's first point - grid regression"
    with rasterio.open(tmp_path / "sub.tif") as src:
        q = src.read(1)
    assert np.count_nonzero(np.nan_to_num(q)) > 0, "an all-zero submission was written"
    # ... and the backstop still holds for a genuinely empty prediction
    from src.submission_io import clean_profile, write_submission
    prof = clean_profile(None, height=8, width=8)
    try:
        write_submission(tmp_path / "zero.tif", np.zeros((8, 8), np.float32), prof)
    except RuntimeError as e:
        assert "all zeros" in str(e)
    else:
        raise AssertionError("all-zero submission must never be written")

def _blend_mod():
    """Import scripts/blend_submission.py as a module (it has no import-time side effects)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "blend_submission", ROOT / "scripts" / "blend_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blend_loo_audit_and_dilate_grid(tmp_path):
    """The leave-one-fold-out audit and the emission-width grid (2026-09-16).

    `--calibrate loo` re-fits the floor on the folds that are NOT being scored, so it needs at
    least three of them; the audit must report a mean that is <= the oracle fitted on the
    scored fold itself.  `--dilate-grid` must reach the report and pick a member of the grid.
    Both are plain assertions about the artifact, not about a fixed number, so they stay true
    as the model changes.

    Session-6 update: the fold-weight rule is fitted per row but NOT scored (folds' held-out
    crops cover different regions, so averaging them is geographically meaningless).  The
    legacy weight keys stay present as None so older report readers do not KeyError.
    """
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):                                   # diagonal fault
        gt[k, k] = 1.0
    rng = np.random.default_rng(1)
    folds = []
    for fi in range(3):
        fd = tmp_path / f"fold-{fi}"
        fd.mkdir()
        from scipy.ndimage import binary_dilation, gaussian_filter
        band = gaussian_filter(binary_dilation(gt > 0.5, iterations=1).astype(np.float32), 1.4)
        prob = np.clip(0.02 + 0.95 * band + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)
        _write_tif(fd / "prob_raw.tif", prob)
        np.savez_compressed(fd / f"heldout_mc{fi}.npz",
                            pred=prob[8:48, 8:48].astype(np.float16), gt=gt[8:48, 8:48].astype(np.uint8))
        (fd / "manifest.json").write_text(json.dumps({"models": [{"file": f"m{fi}.pt", "dti": 0.4}]}))
        folds.append(str(fd))
    _write_tif(tmp_path / "sample.tif", np.zeros((H, W), np.float32))
    _write_tif(tmp_path / "labels.tif", gt)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training: {alpha: 0.2, beta: 0.8}\n"
                   "metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, epsilon: 1.0e-7}\n")
    rep_path = tmp_path / "report.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", *folds, "--config", str(cfg),
         "--sample", str(tmp_path / "sample.tif"), "--labels", str(tmp_path / "labels.tif"),
         "--out", str(tmp_path / "submission.tif"), "--report", str(rep_path),
         "--dilate-grid", "0,1,2", "--calibrate", "loo", "--loo-grid", "3"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(rep_path.read_text())
    loo = rep["aggregation_calibration"]
    assert loo is not None and loo["summary"]["n_folds"] == 3
    assert len(loo["rows"]) == 3
    sm = loo["summary"]
    assert sm["mean_dti_self_best_oracle"] >= sm["mean_dti_loo"] - 1e-9, sm
    assert abs((sm["mean_dti_loo"] - sm["mean_dti_unshaped"]) - sm["gain_loo_over_unshaped"]) < 1e-9
    assert "acceptance" in sm and "weight_rule_gain" in sm
    # session 6: no LOO score for the weight rule (cross-geography averaging removed);
    # the keys stay as None, the per-row fitted weights stay as weights_loo (summing to 1)
    assert sm["weight_rule_gain"] is None
    assert sm["mean_dti_loo_weighted"] is None and sm["mean_dti_loo_equal_weighted_mean"] is None
    assert "weight_rule_note" in sm
    for r in loo["rows"]:
        assert r["weights_loo"] is not None
        assert abs(sum(r["weights_loo"]) - 1.0) < 1e-6
    sh = rep["shaping"]
    assert sh["dilate_grid"] == [0, 1, 2] and sh["dilate"] in (0, 1, 2)


def test_calibrate_shaping_without_heldout_crops_returns_full_tuple():
    """Regression (session 6): with no usable folds calibrate_shaping returned 4 values while
    every caller unpacks 5 (t0, thin, mean, table, dilate) -> ValueError."""
    mod = _blend_mod()
    from src.submission_optim import shaping_thresholds
    out = mod.calibrate_shaping([{"pred_crop": None, "gt_crop": None}], R=3,
                                thresholds=shaping_thresholds(5), alpha=0.2, beta=0.8)
    assert len(out) == 5, f"expected a 5-tuple, got {len(out)} values"
    t0, thin, mean_dti, table, dil = out
    assert (t0, thin, table, dil) == (0.3, True, [], 0)
    assert mean_dti != mean_dti  # nan


def test_pre_transform_does_not_mutate_fold_crops():
    """Regression (session 6): with --calibrate loo the pre-transform (e.g. Frangi) was applied
    once by calibrate_shaping and AGAIN by loo_aggregation to the same arrays, because both
    mutated the shared fold dicts.  Calibration must copy, never mutate."""
    mod = _blend_mod()
    from src.submission_optim import shaping_thresholds
    rng = np.random.default_rng(3)
    folds = []
    for i in range(3):
        gt = np.zeros((24, 24), np.float32)
        gt[10:14, 5:20] = 1.0
        folds.append(dict(dir=f"f{i}", pred_crop=rng.random((24, 24)).astype(np.float32),
                          gt_crop=gt, mean_dti=0.2))
    before = [f["pred_crop"].copy() for f in folds]

    def pre(a):
        return np.clip(a + 0.1, 0, 1).astype(np.float32)

    thr = shaping_thresholds(3)
    mod.calibrate_shaping(folds, 3, thr, 0.2, 0.8, pre=pre)
    loo = mod.loo_aggregation(folds, 3, thr, 0.2, 0.8, pre=pre)
    assert loo is not None and loo["summary"]["n_folds"] == 3
    for f, b in zip(folds, before):
        assert np.array_equal(f["pred_crop"], b), "calibration mutated the fold's pred_crop"


def test_weights_dti_falls_back_to_equal_loudly(tmp_path):
    """--weights dti with a fold lacking positive held-out DTI used to fall back to equal
    weights silently.  A requested weighting that does not happen must say so (session 6)."""
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):
        gt[k, k] = 1.0
    fd = tmp_path / "fold-nodti"
    fd.mkdir()
    prob = np.clip(gt + 0.05, 0, 1).astype(np.float32)
    _write_tif(fd / "prob_raw.tif", prob)
    np.savez_compressed(fd / "heldout_mc0.npz", pred=prob.astype(np.float16),
                        gt=gt.astype(np.uint8))
    (fd / "manifest.json").write_text(json.dumps({"models": [{"file": "m.pt", "dti": 0.0}]}))
    _write_tif(tmp_path / "labels.tif", gt)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training: {alpha: 0.2, beta: 0.8}\n"
                   "metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, epsilon: 1.0e-7}\n")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", str(fd), "--config", str(cfg), "--labels", str(tmp_path / "labels.tif"),
         "--out", str(tmp_path / "sub.tif"), "--report", str(tmp_path / "rep.json"),
         "--weights", "dti"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING" in r.stdout and "equal weights" in r.stdout, r.stdout
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["fold_weights"] == "equal"


def test_generate_dummy_submission_is_seeded(tmp_path):
    """--seed (session 6): two runs with the same seed must produce identical bytes (rules
    §3.2/§3.5 reproducibility); different seeds must differ."""
    spec = importlib.util.spec_from_file_location(
        "generate_dummy_submission", ROOT / "scripts" / "generate_dummy_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a, b, c = (tmp_path / f"d{i}.tif" for i in range(3))
    mod.generate_dummy(str(a), width=64, height=64, seed=7)
    mod.generate_dummy(str(b), width=64, height=64, seed=7)
    mod.generate_dummy(str(c), width=64, height=64, seed=8)
    assert a.read_bytes() == b.read_bytes(), "same seed produced different files"
    assert a.read_bytes() != c.read_bytes(), "different seeds produced identical files"



def test_save_ensemble_writes_the_pre_shaping_input(tmp_path):
    """--save-ensemble must persist the exact map shaping consumed.

    Why it matters: the proxy-catalogue sweep (scripts/eval_proxy_catalogue.py --sweep) asks
    policy questions about the emission width on real faults the labels lack.  If it re-derived
    the "ensemble" from the shaped submission it would score the CURRENT policy against
    alternatives - the comparison would be rigged.  So the blend has to hand over its input.
    """
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):
        gt[k, k] = 1.0
    rng = np.random.default_rng(3)
    from scipy.ndimage import binary_dilation, gaussian_filter

    folds = []
    for fi in range(2):
        fd = tmp_path / f"fold-{fi}"
        fd.mkdir()
        band = gaussian_filter(binary_dilation(gt > 0.5, iterations=1).astype(np.float32), 1.4)
        prob = np.clip(0.02 + 0.9 * band + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)
        prob[:4, :4] = np.nan
        _write_tif(fd / "prob_raw.tif", prob)
        np.savez_compressed(fd / f"heldout_mc{fi}.npz",
                            pred=prob[8:48, 8:48].astype(np.float16),
                            gt=gt[8:48, 8:48].astype(np.uint8))
        (fd / "manifest.json").write_text(json.dumps({"models": [{"file": f"m{fi}.pt", "dti": 0.5}]}))
        folds.append((str(fd), prob))

    _write_tif(tmp_path / "sample.tif", np.zeros((H, W), np.float32))
    _write_tif(tmp_path / "labels.tif", gt)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training: {alpha: 0.2, beta: 0.8}\n"
                   "metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, "
                   "epsilon: 1.0e-7}\n")
    ens_path = tmp_path / "ensemble_mean.tif"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", *[f for f, _ in folds], "--config", str(cfg),
         "--sample", str(tmp_path / "sample.tif"), "--labels", str(tmp_path / "labels.tif"),
         "--out", str(tmp_path / "submission.tif"), "--report", str(tmp_path / "report.json"),
         "--save-ensemble", str(ens_path)],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ens_path.exists(), "blend did not write the ensemble mean"

    expect = np.nanmean(np.stack([p for _, p in folds]), axis=0)
    with rasterio.open(ens_path) as src:
        ens = src.read(1)
        assert src.count == 1 and src.dtypes[0] == "float32" and src.crs.to_epsg() == 32611
        assert (src.height, src.width) == (H, W)
    # The blend zero-fills outside the footprint before shaping (blend_submission: "0 outside
    # footprint for now") and re-applies NaN only when writing the submission, so the ensemble
    # raster is finite where the submission is NaN.  That is the map shaping actually consumes, so
    # the sweep scores exactly the pixels the pipeline scored.  Pinned here so the two files cannot
    # drift apart silently.
    assert np.isfinite(ens).all() and (ens[:4, :4] == 0).all(), "footprint hole: 0 in, NaN out"
    ok = np.isfinite(expect)
    probe = ok.copy()
    probe[:4, :4] = False                       # nanmean of an all-NaN block is not defined
    assert np.allclose(ens[probe], expect[probe], atol=1e-6), "ensemble raster != mean of fold maps"

    # shaping is lossy by construction, so the pre-shaping map must carry at least as much mass
    with rasterio.open(tmp_path / "submission.tif") as src:
        sub = src.read(1)
    assert float(np.nansum(ens)) >= float(np.nansum(sub)) - 1e-6


def test_min_dilate_restricts_the_pooled_search_space(tmp_path):
    """`--min-dilate` is how a MEASURED emission width gets shipped.

    The pooled search maximises in-domain held-out DTI, and on the faults the model trained on the
    narrowest band always wins (0.1903 at width 0 vs 0.0908 at width 6, data/evidence/
    emission_decision.json).  The scored population is the opposite regime: on faults the labels
    lack, the shipped skeleton's MEDIAN miss distance is 22 px and every plausible scored-truth
    size above ~2,000 km prefers a band (data/evidence/proxy/miss_distance-ensemble1.json).
    Shipping that requires constraining the search space, not overruling the calibration - so the
    option must (a) be a no-op by default, (b) drop narrower candidates, (c) be recorded.
    """
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):
        gt[k, k] = 1.0
    from scipy.ndimage import binary_dilation, gaussian_filter
    rng = np.random.default_rng(2)
    folds = []
    for fi in range(2):
        fd = tmp_path / f"fold-{fi}"
        fd.mkdir()
        band = gaussian_filter(binary_dilation(gt > 0.5, iterations=1).astype(np.float32), 1.4)
        prob = np.clip(0.02 + 0.95 * band + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)
        _write_tif(fd / "prob_raw.tif", prob)
        np.savez_compressed(fd / f"heldout_mc{fi}.npz",
                            pred=prob[8:48, 8:48].astype(np.float16),
                            gt=gt[8:48, 8:48].astype(np.uint8))
        (fd / "manifest.json").write_text(json.dumps({"models": [{"file": f"m{fi}.pt", "dti": 0.4}]}))
        folds.append(str(fd))
    _write_tif(tmp_path / "sample.tif", np.zeros((H, W), np.float32))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training: {alpha: 0.2, beta: 0.8}\n"
                   "metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, "
                   "epsilon: 1.0e-7}\n")

    def blend(extra, name):
        rep = tmp_path / f"{name}.json"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
             "--folds", *folds, "--config", str(cfg),
             "--out", str(tmp_path / f"{name}.tif"), "--report", str(rep),
             "--dilate-grid", "0,2,3", *extra],
            capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout + r.stderr
        return json.loads(rep.read_text())["shaping"], r.stdout

    default, _ = blend([], "default")
    assert default["min_dilate"] == 0
    assert default["dilate_grid"] == [0, 2, 3] and default["dilate"] in (0, 2, 3)

    narrow, out = blend(["--min-dilate", "3"], "narrow")
    assert narrow["min_dilate"] == 3
    assert narrow["dilate_grid"] == [3]
    assert "restricted to widths (3,)" in out
    # The unshaped row may NOT win once a width has been asked for: it used to seed the search, so
    # on the catalogue (where the raw soft map can beat every band) a run told to ship a 3 px band
    # would have returned dilate=0 - a silent no-op.  The seed now loses by construction.
    assert narrow["dilate"] == 3, narrow
    assert narrow["calibration_table"][0]["excluded_from_search"] is True
    excluded = {r["dilate"] for r in narrow["calibration_table"] if r.get("dilate") is not None}
    assert not (excluded & {1, 2}), f"a width below --min-dilate entered the table: {excluded}"

    # a min-dilate outside the grid still yields a usable search space rather than an empty one
    outside, _ = blend(["--min-dilate", "5"], "outside")
    assert outside["dilate_grid"] == [5] and outside["dilate"] == 5
