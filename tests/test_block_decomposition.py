"""Per-block decomposition of the official metric must be EXACT, not an approximation.

`src.metrics.score_within_mask` exists so a spatial block hold-out can report where a score
comes from (which 51 km block earns the DTI) instead of only the single global number.  Two
failure modes would be invisible in the global number and fatal to the per-block one:

* cropping a block and scoring it alone (under-credits truth near the boundary, over-penalises
  predictions whose supporting truth sits just outside) - the reason this function keeps the
  global geometry;
* a decomposition whose parts do not sum to the whole - which would mean the per-block table
  and the published score describe different predictions.

Both are pinned here against the definition, written out independently of the implementation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, EPS, GtContext,  # noqa: E402
                         compute_distance_weighted_tversky_reference, score_within_mask)


def _case(seed=0, shape=(48, 56), n_gt=40, density=0.35):
    rng = np.random.default_rng(seed)
    gt = (rng.random(shape) < 0.06).astype(np.float64)
    pred = np.zeros(shape)
    mask = rng.random(shape) < density
    pred[mask] = rng.random(int(mask.sum()))
    return pred, gt


def _brute_force_terms(pred, gt, mask, R=3, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    """The definition, written out per pixel, restricted to `mask` but with global geometry."""
    H, W = gt.shape
    offs = [(dy, dx, max(1.0 - math.hypot(dy, dx) / R, 0.0))
            for dy in range(-R, R + 1) for dx in range(-R, R + 1) if math.hypot(dy, dx) <= R]
    gy, gx = np.nonzero(gt > 0.5)
    TP = 0.0
    n_gt = 0
    for y, x in zip(gy, gx):
        if not mask[y, x]:
            continue
        n_gt += 1
        best = 0.0
        for dy, dx, k in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                best = max(best, k * pred[yy, xx])
        TP += best
    FN = n_gt - TP
    FP = 0.0
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or pred[y, x] <= 0:
                continue
            bestk = 0.0
            for gy2, gx2 in zip(gy, gx):
                d = math.hypot(y - gy2, x - gx2)
                bestk = max(bestk, max(1.0 - d / R, 0.0))
            FP += pred[y, x] * (1.0 - bestk)
    return dict(TP_w=TP, FP_w=FP, FN_w=FN, n_gt=n_gt,
                dti=(TP / (TP + alpha * FP + beta * FN + EPS)) if n_gt else None)


# --------------------------------------------------------------------------------------
# correctness against the definition
# --------------------------------------------------------------------------------------
def test_single_block_matches_a_bruteforce_reading_of_the_definition():
    pred, gt = _case(seed=1)
    mask = np.zeros(gt.shape, bool)
    mask[8:32, 12:40] = True
    ctx = GtContext(gt)
    got = score_within_mask(pred, ctx, mask)
    exp = _brute_force_terms(pred, gt, mask)
    for k in ("TP_w", "FP_w", "FN_w"):
        assert got[k] == pytest.approx(exp[k], rel=1e-9, abs=1e-12), f"{k}: {got[k]} != {exp[k]}"
    assert got["n_gt"] == exp["n_gt"]
    assert got["dti"] == pytest.approx(exp["dti"], rel=1e-12)


def test_whole_grid_mask_reproduces_the_global_score():
    pred, gt = _case(seed=2)
    ctx = GtContext(gt)
    all_mask = np.ones(gt.shape, bool)
    got = score_within_mask(pred, ctx, all_mask)
    dti, (TP, FP, FN) = ctx.score(pred, return_components=True)
    assert got["TP_w"] == pytest.approx(TP, rel=1e-12)
    assert got["FP_w"] == pytest.approx(FP, rel=1e-12)
    assert got["FN_w"] == pytest.approx(FN, rel=1e-12)
    assert got["dti"] == pytest.approx(dti, rel=1e-12)


# --------------------------------------------------------------------------------------
# additivity over a partition (the property a per-block table depends on)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("block", [8, 16, 24])
def test_parts_sum_to_the_whole_over_a_block_partition(block):
    pred, gt = _case(seed=3, shape=(48, 48))
    ctx = GtContext(gt)
    credit = ctx.credit_vector(pred)
    H, W = gt.shape
    tot = dict(TP_w=0.0, FP_w=0.0, FN_w=0.0, n_gt=0, pred_px=0, pred_mass=0.0)
    for y0 in range(0, H, block):
        for x0 in range(0, W, block):
            m = np.zeros(gt.shape, bool)
            m[y0:y0 + block, x0:x0 + block] = True
            r = score_within_mask(pred, ctx, m, credit=credit)
            tot["TP_w"] += r["TP_w"]
            tot["FP_w"] += r["FP_w"]
            tot["FN_w"] += r["FN_w"]
            tot["n_gt"] += r["n_gt"]
            tot["pred_px"] += r["pred_px"]
            tot["pred_mass"] += r["pred_mass"]
    dti, (TP, FP, FN) = ctx.score(pred, return_components=True)
    assert tot["TP_w"] == pytest.approx(TP, rel=1e-9, abs=1e-9)
    assert tot["FP_w"] == pytest.approx(FP, rel=1e-9, abs=1e-9)
    assert tot["FN_w"] == pytest.approx(FN, rel=1e-9, abs=1e-9)
    assert tot["n_gt"] == ctx.n_gt
    pos = np.nan_to_num(pred) > 0
    assert tot["pred_px"] == int(pos.sum())
    assert tot["pred_mass"] == pytest.approx(float(pred[pos].sum()), rel=1e-9)
    # and recombining the parts reproduces the published index
    recomposed = tot["TP_w"] / (tot["TP_w"] + DEFAULT_ALPHA * tot["FP_w"] + DEFAULT_BETA * tot["FN_w"] + EPS)
    assert recomposed == pytest.approx(dti, rel=1e-12)


def test_overlapping_masks_double_count_so_the_caller_must_partition():
    """Guard against a silent misuse: masks are summed, they are not de-duplicated."""
    pred, gt = _case(seed=4)
    ctx = GtContext(gt)
    m1 = np.zeros(gt.shape, bool); m1[:, : gt.shape[1] // 2 + 4] = True
    m2 = np.zeros(gt.shape, bool); m2[:, gt.shape[1] // 2 - 4:] = True
    a = score_within_mask(pred, ctx, m1)["n_gt"]
    b = score_within_mask(pred, ctx, m2)["n_gt"]
    assert a + b > ctx.n_gt, "the overlap must be visible, i.e. masks are not de-duplicated"


# --------------------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------------------
def test_block_with_no_truth_reports_dti_none_but_keeps_its_false_positives():
    pred, gt = _case(seed=5)
    ctx = GtContext(gt)
    empty = np.zeros(gt.shape, bool)
    gy, gx = np.nonzero(gt > 0.5)
    empty[gy, gx] = False                     # keep every truth pixel out
    # choose a region that provably contains no truth
    row_has = np.any(gt > 0.5, axis=1)
    free = np.where(~row_has)[0]
    if len(free):
        m = np.zeros(gt.shape, bool)
        m[free, :] = True
        if (m & (gt > 0.5)).sum() == 0:
            r = score_within_mask(pred, ctx, m)
            assert r["dti"] is None and r["scoreable"] is False
            assert r["n_gt"] == 0 and r["TP_w"] == 0.0 and r["FN_w"] == 0.0
            assert r["FP_w"] >= 0.0


def test_empty_prediction_scores_zero_in_every_block():
    _, gt = _case(seed=6)
    ctx = GtContext(gt)
    m = np.ones(gt.shape, bool)
    r = score_within_mask(np.zeros(gt.shape), ctx, m)
    assert r["TP_w"] == 0.0 and r["pred_px"] == 0 and r["dti"] == pytest.approx(0.0, abs=1e-9)


def test_nan_prediction_is_treated_as_the_spec_says():
    """NaN outside the footprint -> no credit, no penalty (problem page #submission-format)."""
    pred, gt = _case(seed=7)
    ctx = GtContext(gt)
    m = np.ones(gt.shape, bool)
    with_nan = pred.copy()
    with_nan[0:4, 0:4] = np.nan
    r_nan = score_within_mask(with_nan, ctx, m)
    r_zero = score_within_mask(np.nan_to_num(with_nan), ctx, m)
    assert r_nan["TP_w"] == pytest.approx(r_zero["TP_w"], rel=1e-12)
    assert r_nan["FP_w"] == pytest.approx(r_zero["FP_w"], rel=1e-12)


def test_mask_shape_mismatch_is_rejected():
    pred, gt = _case(seed=8)
    ctx = GtContext(gt)
    with pytest.raises(ValueError):
        score_within_mask(pred, ctx, np.ones((gt.shape[0] - 1, gt.shape[1]), bool))


def test_credit_reuse_gives_identical_numbers():
    """The caller scores many candidates per block; reusing one credit vector must be exact."""
    pred, gt = _case(seed=9)
    ctx = GtContext(gt)
    credit = ctx.credit_vector(pred)
    m = np.zeros(gt.shape, bool); m[10:30, 10:30] = True
    a = score_within_mask(pred, ctx, m)
    b = score_within_mask(pred, ctx, m, credit=credit)
    assert a == b


def test_reference_implementation_agrees_on_the_whole_grid():
    pred, gt = _case(seed=10)
    ctx = GtContext(gt)
    dti_ref, comps_ref = compute_distance_weighted_tversky_reference(pred, gt, return_components=True)
    r = score_within_mask(pred, ctx, np.ones(gt.shape, bool))
    assert r["dti"] == pytest.approx(dti_ref, rel=1e-12)
    assert (r["TP_w"], r["FP_w"], r["FN_w"]) == pytest.approx(comps_ref, rel=1e-12)
