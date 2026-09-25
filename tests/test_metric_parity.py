"""Parity between the fast metric path (GtContext) and the original full-raster formulation.

WHY.  The fast path was introduced on 2026-09-16 to make the leave-one-fold-out calibration audit
practical (the general implementation ran for more than 90 minutes on a runner, because it rebuilt a
49-offset credit map over the whole raster for every candidate floor, then read only the ~1% of
pixels that are labels).  A speed-up in the *scoring function* is the one change in this repository
that could silently corrupt every number at once, so equivalence is tested rather than assumed:
random predictions, random labels, several radii, NaN/Inf/huge values, all-zero labels and all-one
predictions.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (GtContext, compute_distance_weighted_tversky,          # noqa: E402
                         compute_distance_weighted_tversky_reference)


def _cases():
    rng = np.random.default_rng(7)
    yield ("random", rng.random((48, 61)).astype(np.float32), (rng.random((48, 61)) > 0.97))
    yield ("sparse", rng.random((40, 40)).astype(np.float32) * 0.05,
           np.zeros((40, 40), bool))
    yield ("dense-labels", rng.random((33, 29)).astype(np.float32),
           np.ones((33, 29), bool))
    yield ("empty-pred", np.zeros((25, 25), np.float32), (rng.random((25, 25)) > 0.8))
    yield ("all-ones", np.ones((21, 21), np.float32), (rng.random((21, 21)) > 0.9))
    x = rng.random((30, 30)).astype(np.float32)
    x[0, 0] = np.nan
    x[1, 1] = np.inf
    x[2, 2] = -np.inf
    x[3, 3] = 5.0                                     # out of range: must be clipped
    x[4, 4] = -3.0
    yield ("nan-inf-out-of-range", x, (rng.random((30, 30)) > 0.93))
    yield ("single-label-pixel-corner", rng.random((16, 16)).astype(np.float32),
           np.eye(16, dtype=bool))


def test_fast_path_matches_the_full_raster_reference():
    for name, pred, gt in _cases():
        for R in (1, 2, 3, 5):
            for alpha, beta in ((0.2, 0.8), (0.5, 0.5), (0.9, 0.1)):
                fast = compute_distance_weighted_tversky(pred, gt, R_pixels=R, alpha=alpha, beta=beta)
                slow = compute_distance_weighted_tversky_reference(pred, gt, R_pixels=R,
                                                                   alpha=alpha, beta=beta)
                assert abs(fast - slow) < 1e-9, (name, R, alpha, beta, fast, slow)


def test_components_match_too():
    """TP_w / FP_w / FN_w must agree, not just the ratio (they are reported as evidence)."""
    rng = np.random.default_rng(11)
    pred = rng.random((44, 37)).astype(np.float32)
    gt = (rng.random((44, 37)) > 0.95)
    for R in (1, 3, 4):
        f = compute_distance_weighted_tversky(pred, gt, R_pixels=R, return_components=True)[1]
        s = compute_distance_weighted_tversky_reference(pred, gt, R_pixels=R,
                                                        return_components=True)[1]
        assert np.allclose(f, s, atol=1e-9), (R, f, s)


def test_reusing_one_context_matches_per_call_scoring():
    """The audit scores hundreds of candidates against one label crop; that reuse must be exact."""
    rng = np.random.default_rng(3)
    gt = (rng.random((50, 50)) > 0.96)
    ctx = GtContext(gt, R_pixels=3)
    for _ in range(5):
        pred = (rng.random((50, 50)) > 0.5).astype(np.float32) * rng.random((50, 50))
        assert abs(ctx.score(pred) - compute_distance_weighted_tversky(pred, gt)) < 1e-12


def test_shape_mismatch_is_still_loud():
    import pytest
    with pytest.raises(ValueError):
        GtContext(np.zeros((10, 10), bool), R_pixels=3).score(np.zeros((9, 9)))
