"""Verification suite for the GEMS Prize repo.

Every test here is a *falsifiable check of a claim made in this repository*.  Run:

    python -m pytest tests -q          (or: python tests/test_metric.py)

Anti-hallucination policy: nothing in the repo may assert a number the code cannot reproduce.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (compute_distance_weighted_tversky, dti_bounds, kernel_offsets,  # noqa: E402
                          score_arrays_blocked)


def _brute(pred, gt, R=3, alpha=0.2, beta=0.8, eps=1e-7):
    """Literal transcription of the four formulas on the problem page (slow, obvious)."""
    H, W = pred.shape
    P = [(y, x, float(pred[y, x])) for y in range(H) for x in range(W) if pred[y, x] > 0]
    G = [(y, x) for y in range(H) for x in range(W) if gt[y, x] > 0.5]
    k = lambda d: max(1.0 - d / R, 0.0)
    TP = 0.0
    for (gy, gx) in G:
        best = 0.0
        for (y, x, p) in P:
            d = math.hypot(y - gy, x - gx)
            if d <= R + 1e-12:
                best = max(best, p * k(d))
        TP += best
    FN = len(G) - TP
    FP = 0.0
    for (y, x, p) in P:
        bk = 0.0
        for (gy, gx) in G:
            bk = max(bk, k(math.hypot(y - gy, x - gx)))
        FP += p * (1.0 - bk)
    return TP / (TP + alpha * FP + beta * FN + eps), (TP, FP, FN)


# ---------------------------------------------------------------------------------------
def test_metric_equals_brute_force_definition():
    """src/metrics.py == the literal formulas (6 random + 3 structured rasters)."""
    rng = np.random.default_rng(3)
    cases = []
    for _ in range(6):
        gt = (rng.random((24, 30)) < 0.08).astype(float)
        cases.append((rng.random((24, 30)) * (rng.random((24, 30)) < 0.3), gt))
    line = np.zeros((20, 20)); line[:, 9] = 1
    cases.append(((line * 0.7 + (np.roll(line, 1, axis=1) * 0.4)), line))
    diag = np.zeros((20, 20))
    for i in range(20):
        diag[i, min(19, i)] = 1
    cases.append((np.clip(diag + np.roll(diag, 2, axis=1), 0, 1), diag))
    cases.append((np.ones((8, 8)), np.zeros((8, 8))))          # no GT at all
    for pred, gt in cases:
        a = compute_distance_weighted_tversky(pred, gt)
        b, _ = _brute(pred, gt)
        assert abs(a - b) < 1e-9, (a, b)


def test_blocked_equals_dense():
    """The memory-safe blocked scorer must be numerically identical (no approximation)."""
    rng = np.random.default_rng(11)
    gt = np.zeros((180, 95)); gt[30:90, 50] = 1; gt[120:150, 10:70] = 1
    pred = (rng.random((180, 95)) < 0.06) * rng.random((180, 95))
    d0, c0 = compute_distance_weighted_tversky(pred, gt, return_components=True)
    d1, c1 = score_arrays_blocked(pred, gt, block=48)
    assert abs(d0 - d1) < 1e-9
    assert all(abs(x - y) < 1e-6 for x, y in zip(c0, c1)), (c0, c1)


def test_official_worked_example_arithmetic():
    """Problem page 'Scoring example': TP_w=3.00, FP_w=1.89, FN_w=2.00 -> TI_w = 0.60."""
    v = 3.00 / (3.00 + 0.2 * 1.89 + 0.8 * 2.00)
    assert round(v, 2) == 0.60, v
    # our formula reading with eps=0 -> 0.6027 which the page rounds to 0.60
    assert abs(dti_bounds(3.00, 1.89, 5) - 0.6027) < 2e-4      # |G| = 3 TP + 2 FN = 5 px
    # the configuration our raster search did find (FP_w = 2.00) also gives exactly 0.60
    assert abs(dti_bounds(3.0, 2.0, 5) - 0.60) < 1e-9


def test_tp_plus_fn_equals_number_of_gt_pixels():
    rng = np.random.default_rng(5)
    gt = (rng.random((40, 40)) < 0.05).astype(float)
    pred = rng.random((40, 40)) * (rng.random((40, 40)) < 0.4)
    _, (tp, fp, fn) = compute_distance_weighted_tversky(pred, gt, return_components=True)
    assert abs(tp + fn - gt.sum()) < 1e-9


def test_kernel_values_and_radius():
    offs = {(dy, dx): k for dy, dx, k in kernel_offsets(3)}
    assert offs[(0, 0)] == 1.0
    assert abs(offs[(0, 1)] - 2 / 3) < 1e-12
    assert abs(offs[(1, 1)] - (1 - math.sqrt(2) / 3)) < 1e-12
    assert offs[(0, 3)] == 0.0
    assert (0, 4) not in offs and (3, 3) not in offs     # outside Euclidean radius
    assert len(offs) == 29                                # count of px in a d<=3 disc


def test_recall_is_worth_four_times_precision():
    """alpha=0.2 / beta=0.8 -> dropping a fault line hurts more than inventing one."""
    gt = np.zeros((40, 40)); gt[10:30, 20] = 1
    perfect = compute_distance_weighted_tversky(gt, gt)
    extra = gt.copy(); extra[10:30, 30] = 1
    missing = gt.copy(); missing[10:20, 20] = 0
    assert perfect > 0.9999
    assert compute_distance_weighted_tversky(extra, gt) > compute_distance_weighted_tversky(missing, gt)


def test_within_R_offsets_are_credited_but_beyond_R_are_not():
    gt = np.zeros((40, 40)); gt[10:30, 20] = 1
    near = np.zeros_like(gt); near[10:30, 22] = 1.0     # 2 px away  <= R
    far = np.zeros_like(gt); far[10:30, 24] = 1.0       # 4 px away   > R
    # hand-derived for the 'near' case: 20 GT px, each credited k=1-2/3=1/3 ->
    #   TP=20/3, FN=40/3, FP=40/3  ->  DTI = 1/(1 + 0.2*2 + 0.8*2) = 1/3
    d_near, (tp, fpv, fn) = compute_distance_weighted_tversky(near, gt, return_components=True)
    assert abs(tp - 20 / 3) < 1e-9 and abs(fpv - 40 / 3) < 1e-9 and abs(fn - 40 / 3) < 1e-9
    assert abs(d_near - 1 / 3) < 1e-7        # residual = the official epsilon in the denominator
    assert compute_distance_weighted_tversky(far, gt) == 0.0


def test_nan_outside_and_out_of_range_are_handled():
    gt = np.zeros((30, 30)); gt[5:25, 15] = 1
    p = gt.astype(float).copy(); p[:, :6] = np.nan
    assert np.isfinite(compute_distance_weighted_tversky(p, gt))
    p2 = gt.astype(float) * 4.0 - 1.0
    assert np.isfinite(compute_distance_weighted_tversky(p2, gt))


# ---------------------------------------------------------------------------------------
def test_dw_loss_is_the_metric():
    """Differentiable loss (torch) must equal 1 - DTI of src/metrics.py for the same patch."""
    import torch
    from src.losses import DistanceWeightedTverskyLoss, fp_weights_np

    rng = np.random.default_rng(2)
    gt = np.zeros((64, 64)); gt[10:50, 30] = 1; gt[20:26, 40:48] = 1
    pred = (rng.random((2, 1, 64, 64)) < 0.2) * rng.random((2, 1, 64, 64)).astype(np.float32)
    pred[0] = pred[0] * gt                                 # one sample only on the fault line
    logits = torch.from_numpy(np.log(np.clip(pred, 1e-4, 1 - 1e-4) / (1 - np.clip(pred, 1e-4, 1 - 1e-4)))).float()
    targets = torch.from_numpy(np.stack([gt, np.roll(gt, 3, axis=1)])).float()

    fpw = torch.from_numpy(np.stack([fp_weights_np(gt), fp_weights_np(np.roll(gt, 3, axis=1))])).float()
    loss = DistanceWeightedTverskyLoss(R=3)(logits, targets, fp_weight=fpw)

    # per-sample fp weights are each sample's own 1-max_g k(d), so the batch-mean loss must
    # equal the mean of (1 - DTI) computed by the numpy reference metric.
    ref = []
    for b in range(2):
        db = compute_distance_weighted_tversky(torch.sigmoid(logits[b, 0]).numpy(),
                                               targets[b].numpy(), R_pixels=3)
        ref.append(1.0 - db)
    assert abs(float(loss) - float(np.mean(ref))) < 1e-6, (float(loss), ref)
    assert 0.0 <= float(loss) <= 2.0
    # gradient must flow and be non-zero exactly where it matters
    logits.requires_grad_(True)
    l = DistanceWeightedTverskyLoss(R=3)(logits, targets, fp_weight=fpw)
    l.backward()
    g = logits.grad.abs().sum(dim=(2, 3))
    assert float(g[0]) > 0 and torch.isfinite(logits.grad).all()


def test_fp_weight_map_definition():
    """fpw = 1 - max_g k(d(x,g)); must be 0 on GT, 1/3 one pixel away, 1 beyond R."""
    from src.losses import fp_weights_np
    gt = np.zeros((30, 30)); gt[15, :] = 1
    w = fp_weights_np(gt)
    assert w[15, 10] == 0.0
    assert abs(w[14, 10] - 1 / 3) < 1e-6
    assert abs(w[13, 10] - 2 / 3) < 1e-6
    assert w[10, 10] == 1.0
    assert w[:, 0][0] == 1.0


def test_patches_have_no_label_leakage():
    """make_patches must zero the test windows in the training arrays (reference design)."""
    from src.dataset import make_patches
    rng = np.random.default_rng(0)
    H = W = 256
    X = rng.random((H, W, 3)).astype(np.float32)
    y = np.zeros((H, W), np.float32)
    y[100:150, 50:60] = 1
    y[10:40, 200:210] = 1
    res = make_patches(X, y, patch_size=128, train_step=64, test_proportion=0.5, seed=1)
    tw = res["summary"]["test_windows"]
    p = 128
    test_mask = np.zeros((H, W), bool)
    for (i, j) in tw:
        test_mask[i:i + p, j:j + p] = True
    # every training window must have zero labels inside the test region
    train_label_sum = float(res["y_train"].sum())
    # labels visible to training == labels outside the test windows
    outside = float((y * ~test_mask).sum())
    # training labels can only be a subset of the outside-region labels (windows are cropped)
    assert train_label_sum <= outside + 1e-6, (train_label_sum, outside)
    assert train_label_sum > 0
    # and test labels are untouched / present
    assert float(res["y_test"].sum()) == float((y * test_mask).sum())


def test_fp_weight_map_does_not_leak_heldout_labels():
    """The FP-weight map handed to the loss must know nothing about held-out faults.

    fpw = 1 - max_g k(d(x,g)) is a statement "a fault lies within R px of here".  It used
    to be computed from the labels *before* the test windows were zeroed, so wherever a
    training window overlapped the test grid (allowed up to 25%) the loss was told not to
    penalise predictions exactly on the hidden faults.  Regression test for that fix.
    """
    from scipy.ndimage import distance_transform_edt

    from src.dataset import make_patches
    rng = np.random.default_rng(3)
    H = W = 256
    X = rng.random((H, W, 3)).astype(np.float32)
    y = np.zeros((H, W), np.float32)
    y[40:44, 20:120] = 1          # a fault in the upper-left quadrant
    y[150:250, 150:154] = 1       # and one in the lower-right
    res = make_patches(X, y, patch_size=128, train_step=64, test_proportion=0.5, seed=1)
    s = res["summary"]
    p = s["patch"]
    Hp, Wp = s["H"], s["W"]
    test_mask = np.zeros((Hp, Wp), bool)
    for (i, j) in s["test_windows"]:
        test_mask[i:i + p, j:j + p] = True

    # reference built from TRAIN-ONLY labels, with the held-out region at full FP weight
    yp = np.pad(y, ((0, Hp - H), (0, Wp - W)))
    ytr = yp.copy()
    ytr[test_mask] = 0.0
    d = distance_transform_edt(~(ytr > 0.5))
    ref = 1.0 - np.maximum(1.0 - d / 3.0, 0.0)
    ref[test_mask] = 1.0

    fpw = res["fpw_train"]
    assert fpw.shape[0] == len(s["train_windows"])
    for k, (i, j) in enumerate(s["train_windows"]):
        err = float(np.abs(fpw[k] - ref[i:i + p, j:j + p]).max())
        assert err < 1e-6, (k, (i, j), err)
        # no train patch may carry fpw < 1 inside the held-out region
        leaked = int(((fpw[k] < 1.0) & test_mask[i:i + p, j:j + p]).sum())
        assert leaked == 0, (k, leaked)


def test_shaping_grid_reaches_low_contrast_optima():
    """The floor search must be able to pick t0 below 0.02.

    The full-raster smoke run pinned t0 at 0.02 - the first point of the old
    linspace(0.02, 0.9) grid - which means the grid, not the data, chose the optimum.  An
    under-trained model emits a low-contrast field whose entire useful range sits below
    that floor, so the old grid zeroes the submission outright.
    """
    from src.submission_optim import optimize_submission, shaping_thresholds

    grid = shaping_thresholds(15)
    assert grid[0] == 0.0                                   # explicit "no floor" reference
    assert (grid[1:] > 0).all() and grid.max() <= 0.9
    assert grid[grid > 0].min() <= 1e-3, grid               # must reach well below 0.02
    assert (np.diff(grid) > 0).all()                        # sorted, unique

    H = W = 96
    gt = np.zeros((H, W), np.float32)
    gt[48, 10:86] = 1.0
    p = np.full((H, W), 0.002, np.float32)                  # diffuse low-contrast background
    p[48, 10:86] = 0.011                                    # a weak but correct fault signal

    def best(thresholds):
        out = -1.0
        for t in thresholds:
            for thin in (False, True):
                q = optimize_submission(p, R=3, t0=float(t), thin=thin)
                out = max(out, compute_distance_weighted_tversky(q, gt, R_pixels=3))
        return out

    old = best(np.linspace(0.02, 0.9, 15))
    new = best(grid)
    assert old == 0.0, old            # old grid wipes the whole field: every value <= 0.02
    assert new > 0.5, new             # new grid recovers the fault


def test_augmentation_is_label_consistent():
    """A flip must map the fault line to the flipped label line (x, y, w stay aligned)."""
    from src.dataset import FaultDataset
    X = np.zeros((1, 16, 16, 2), np.float32)
    X[0, :, 4] = 1.0
    y = np.zeros((1, 16, 16), np.float32); y[0, :, 4] = 1.0
    w = np.ones((1, 16, 16), np.float32)
    ds = FaultDataset(X, y, w, train=True, augment=True, noise_std=0.0, rand_crop_scale=(0.75, 1.0), seed=0)
    ok = 0
    for _ in range(60):
        x, yy, ww = ds[0]
        prof_x = x[0].sum(axis=0)
        prof_y = yy.sum(axis=0)
        if np.argmax(prof_x) == np.argmax(prof_y):
            ok += 1
        assert x.shape == (2, 16, 16) and yy.shape == (16, 16)
        assert set(np.unique(yy)) <= {0.0, 1.0}
    assert ok > 0, "augmentation never produced an identity/consistent sample"


def test_norm_stats_roundtrip():
    from src.dataset import apply_norm_stats, fit_norm_stats
    X = np.random.default_rng(1).normal(5, 2, (40, 40, 3)).astype(np.float32)
    X[0, 0] = np.nan
    s = fit_norm_stats(X)
    a = apply_norm_stats(X, s, mode="clip_zscore")
    b = apply_norm_stats(X, s, mode="clip_zscore")
    assert np.array_equal(a, b) and a.shape == X.shape
    assert np.isfinite(a).all() and (a.min() >= 0 and a.max() <= 1)
    m = apply_norm_stats(X, s, mode="minmax")
    assert m[..., 0][np.isnan(X[..., 0])].tolist() == [0.0]


def test_submission_writing_helpers(tmp_path=None):
    """validate_submission must accept a file written by the exact spec of the data page."""
    import rasterio
    from rasterio.transform import from_origin
    arr = np.zeros((50, 60), np.float32); arr[20:30, 10:40] = 0.5
    arr[:, :5] = np.nan
    p = Path(tempfile_name()) if False else None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "s.tif"
        with rasterio.open(f, "w", driver="GTiff", height=50, width=60, count=1, dtype="float32",
                           crs="EPSG:32611", transform=from_origin(500000, 4300000, 100, 100),
                           nodata=float("nan")) as dst:
            dst.write(arr, 1)
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
                            "--pred", str(f), "--sample", str(f), "--train", str(f)],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "PASSED" in r.stdout, r.stdout + r.stderr


def tempfile_name():  # pragma: no cover - helper kept for readability of the test above
    return ""




# ---------------------------------------------------------------------------------------
# submission shaping (src/submission_optim.py) - the measured "metric calculus"
# ---------------------------------------------------------------------------------------
def test_floor_reduces_fp_mass_without_losing_credit():
    """Zeroing sub-floor probability removes FP mass; a p=1 pixel ON the fault still earns
    full TP credit (max over the R-neighbourhood), so DTI must not decrease here."""
    from src.submission_optim import floor_sharpen
    gt = np.zeros((40, 40)); gt[10:30, 20] = 1
    p = gt.astype(np.float32) * 0.9 + 0.3                       # on-fault 0.9, background 0.3
    q = floor_sharpen(p, t0=0.5, hard=True)
    d0, (_, fp0, _) = compute_distance_weighted_tversky(p, gt, return_components=True)
    d1, (_, fp1, _) = compute_distance_weighted_tversky(q, gt, return_components=True)
    assert fp1 == 0.0 and fp0 > 100.0
    assert d1 > d0, (d0, d1)
    assert abs(d1 - 1.0) < 1e-6                                  # TP=|G|, FP=0 -> DTI=1


def test_thinning_wide_band_onto_a_narrow_fault_raises_dti():
    """The realistic case: the model fires on a 5-px-wide corridor, the official label is a
    ~1-px line.  TP_w is a MAX over the R-neighbourhood, so extra parallel pixels add nothing
    to TP while each adds 0.2*(1-k) to FP.  Thinning to a distance-R dominating set therefore
    *raises* DTI.  (Counter-case, measured here too: thinning a thick GT band costs credit -
    see docs/results.html 'metric calculus' - so we thin predictions, never labels.)"""
    from src.submission_optim import dominant_thin
    from scipy.ndimage import distance_transform_edt

    gt = np.zeros((60, 60), np.float32); gt[10:40, 25] = 1.0            # 1-px-wide fault line
    band = np.zeros_like(gt); band[10:40, 23:28] = 1.0                  # 5-px corridor, contains it
    sel = dominant_thin(band, R=3, p=band)

    # (a) dominance: every band pixel still has a kept pixel within R
    d = distance_transform_edt(sel == 0)
    assert (d[band > 0.5] <= 3).all(), "thinning broke R-dominance"
    # (b) mass collapses, credit does not
    d_band, (tp_b, fp_b, _fn_b) = compute_distance_weighted_tversky(band, gt, return_components=True)
    d_sel, (tp_s, fp_s, _fn_s) = compute_distance_weighted_tversky(sel, gt, return_components=True)
    assert sel.sum() < 0.6 * band.sum(), (sel.sum(), band.sum())
    # FP mass collapses (>=90% here); TP may drop a little because a discrete skeleton can
    # sit 1 px off the labelled line (credit k=2/3 instead of 1) - bounded, and worth it.
    assert fp_s < 0.5 * fp_b, (fp_b, fp_s)
    assert tp_s >= 0.9 * tp_b, (tp_b, tp_s)
    assert d_sel > d_band, (d_band, d_sel)
    # (c) a thick GT band is the counter-example: thinning *predictions* to the skeleton keeps
    #     only partial credit there, which is why t0/thin are chosen by held-out search rather
    #     than fixed.  Assert the direction so a future change that ignores this fails loudly.
    gt_thick = np.zeros((60, 60), np.float32); gt_thick[:, 25:30] = 1.0
    sel_thin = dominant_thin(gt_thick.copy(), R=3, p=gt_thick)
    assert compute_distance_weighted_tversky(gt_thick, gt_thick) > compute_distance_weighted_tversky(sel_thin, gt_thick)


def test_marginal_rule_predict_only_within_R():
    """Derived rule (src/submission_optim.py docstring): adding probability v at pixel x
    raises DTI iff k(d) > alpha * DTI.  Verified numerically on a concrete map."""
    alpha, beta = 0.2, 0.8
    gt = np.zeros((50, 50)); gt[10:40, 25] = 1
    p = gt.astype(float).copy()                                  # perfect prediction: DTI=1
    d0 = compute_distance_weighted_tversky(p, gt)
    for col, expected_gain in [(27, False), (28, False)]:        # 2-3 px away, GT already max
        q = p.copy(); q[:, col] = 1.0
        d1 = compute_distance_weighted_tversky(q, gt)
        # credit at those GT pixels is already 1 (p=1 on them), so this is pure FP cost
        assert d1 < d0, (col, d0, d1)
    # a GT pixel that is NOT covered: adding v within R must raise DTI (k > alpha*DTI holds
    # because DTI is far below 1 there)
    gt2 = np.zeros((50, 50)); gt2[10:40, 25] = 1
    p2 = np.zeros_like(gt2); p2[10:40, 22] = 1.0                 # 3 px away, k=0
    d_a = compute_distance_weighted_tversky(p2, gt2)
    p3 = p2.copy(); p3[10:40, 24] = 1.0                          # 1 px away, k=2/3
    d_b = compute_distance_weighted_tversky(p3, gt2)
    assert d_b > d_a and d_a == 0.0, (d_a, d_b)


def test_manifest_is_written_by_training(tmp_path=None):
    """Regression: training must write manifest.json (models + calibrated shaping) and inference
    must read it.  An earlier refactor silently dropped the write, so inference reused a stale
    shaping dict - which is why this is checked at the source level, not only on artifacts
    (artifacts are gitignored and may not exist on a fresh machine)."""
    import re as _re
    src_train = (ROOT / "src" / "train.py").read_text()
    src_infer = (ROOT / "src" / "inference.py").read_text()
    assert _re.search(r"manifest\.json", src_train), "src/train.py no longer names manifest.json"
    # the write itself (currently `(out_dir/"manifest.json").write_text(json.dumps(manifest,...))`)
    assert _re.search(r"json\.dumps?\(\s*manifest", src_train), \
        "src/train.py builds a manifest but no longer dumps it - the silent-drop regression"
    assert "manifest.json" in src_infer or "MANIFEST" in src_infer.upper(), \
        "src/inference.py no longer reads the manifest, so calibrated shaping would be ignored"
    for key in ("R_pixels", "band_names", "models", "feature_grid"):
        assert f'"{key}"' in src_train or key in src_train, f"manifest lost the {key} field"
    out = ROOT / "outputs_recon" / "manifest.json"
    if out.exists():                       # if a run is present, its contents must be valid too
        m = json.loads(out.read_text())
        assert m["models"], "manifest has no model entries"
        assert all("arch" in e and "encoder" in e and "in_channels" in e for e in m["models"])
        assert "R_pixels" in m and m["n_bands"] > 0
        if "shaping" in m:
            assert 0.0 <= m["shaping"]["t0"] <= 1.0 and isinstance(m["shaping"]["thin"], bool)


def test_postprocess_version_shims_behave_identically():
    """skimage renamed square()->footprint_rectangle (0.25) and min_size->max_size (0.26).
    Both shims must keep the previous semantics exactly, or post-processing silently changes
    the submitted mask while still looking fine."""
    import numpy as np
    from skimage.morphology import closing
    from src.postprocess import morphological_close, filter_small_faults
    try:
        from skimage.morphology import footprint_rectangle as rect
    except ImportError:                                   # pragma: no cover
        from skimage.morphology import square as _sq
        rect = lambda shape: _sq(shape[0])
    rng = np.random.default_rng(0)
    for k in (3, 5, 7):
        x = rng.random((64, 64)) > 0.7
        assert np.array_equal(closing(x, rect((k, k))), morphological_close(x, k)), k
    # closing fills a 1-px hole but does not grow an isolated pixel
    ring = np.zeros((7, 7), bool); ring[2:5, 2:5] = True; ring[3, 3] = False
    assert morphological_close(ring, 3).sum() == 9
    assert morphological_close(np.array([[True]]), 3).sum() == 1
    # "smaller than min_length" semantics, not "smaller or equal"
    blobs = np.zeros((12, 12), np.uint8)
    blobs[0:2, 0:2] = 1      # 4 px  -> removed at min_length=5
    blobs[5:8, 5:8] = 1      # 9 px  -> kept
    assert filter_small_faults(blobs, min_length=5).sum() == 9
    assert filter_small_faults(blobs, min_length=3).sum() == 13   # 4-px blob kept at 3


def test_line_geometry_dti_values():
    """Pins the numbers in docs/methodology.html's 'how the metric behaves' table.

    A 20-px, 1-px-wide line is the ground truth; alpha=0.2, beta=0.8.  These are the values the
    documentation quotes, so they are asserted here instead of being quoted from memory - an
    earlier revision of that paragraph (0.91 / 0.61) had never been produced by this code.
    """
    H = W = 40
    gt = np.zeros((H, W), dtype=bool)
    gt[10:30, 20] = True
    cases = {0: 1.0, 1: 2 / 3, 2: 1 / 3, 3: 0.0, 4: 0.0}
    for off, want in cases.items():
        pred = np.zeros((H, W), dtype=bool)
        if off == 0:
            pred[:] = gt
        else:
            pred[10:30, 20 + off] = True
        got = compute_distance_weighted_tversky(pred.astype(np.float32), gt)
        assert abs(got - want) < 1e-4, f"offset {off} px: DTI={got:.4f}, expected {want:.4f}"
    # a 5-px band: full recall, and 40 extra px at weights (1-k) -> FP_w = 40
    band = np.zeros((H, W), dtype=bool)
    band[10:30, 18:23] = True
    dti, (tp, fp, fn) = compute_distance_weighted_tversky(
        band.astype(np.float32), gt, return_components=True)
    assert abs(tp - 20.0) < 1e-6 and abs(fn) < 1e-9, (tp, fn)
    assert abs(fp - 40.0) < 1e-6, fp
    assert abs(dti - 20 / (20 + 0.2 * 40)) < 1e-6, dti
    # the tolerance is *within* R, not *up to* R: k vanishes exactly at d = R
    offs = {(dy, dx): k for dy, dx, k in kernel_offsets(3)}
    assert abs(offs[(0, 1)] - 2 / 3) < 1e-12          # 1 px off keeps 2/3 of the credit
    assert offs[(0, 3)] == 0.0                        # 3 px off keeps nothing
    assert (0, 4) not in offs and (3, 3) not in offs  # outside the Euclidean radius
    # nothing predicted at all -> 0, and it must not be NaN
    assert compute_distance_weighted_tversky(np.zeros((H, W), np.float32), gt) == 0.0


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
