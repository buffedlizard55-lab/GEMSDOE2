"""The windowed greedy fill in `dominant_thin` must equal the full-raster formulation.

WHY.  Per candidate shaping evaluation, `dominant_thin` used to compute a whole-raster Euclidean
distance transform *per iteration* of its greedy fill (up to 40 iterations).  That was the dominant
cost of the leave-one-fold-out calibration audit, which spent over 90 minutes on a runner without
finishing on 2026-09-16.  The distance transform and the density filter are now restricted to the
bounding box of the still-uncovered pixels plus an R+1 halo - exact for the only question the loop
asks ("is this pixel farther than R from the kept set?"), because any kept pixel within R of the
window lies inside the halo.

A faster thinning that kept a different pixel set would change every published DTI, so the
equivalence is pinned here against a literal re-implementation of the original body.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.ndimage import distance_transform_edt, uniform_filter        # noqa: E402
from skimage.morphology import skeletonize                              # noqa: E402

from src.submission_optim import dominant_thin                          # noqa: E402


def _reference_dominant_thin(mask, R=3, p=None, max_extra=40):
    """The pre-2026-09-16 implementation, verbatim in structure."""
    m = mask > 0.5
    if not m.any():
        return m.astype(np.float32)
    sel = skeletonize(m)
    if not sel.any():
        sel = m.copy()
    sel = sel & m
    if sel.any():
        for _ in range(max_extra):
            d = distance_transform_edt(~sel)
            uncovered = m & (d > R)
            if not uncovered.any():
                break
            dens = uniform_filter(uncovered.astype(np.float64), size=2 * R + 1)
            cand = m & ~sel
            score = dens + 1e-3 * p if p is not None else dens
            score = np.where(cand, score, -np.inf)
            iy, ix = np.unravel_index(np.argmax(score), score.shape)
            sel[iy, ix] = True
    return sel.astype(np.float32)


def _random_mask(rng, H, W):
    m = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(1, 6))):
        if rng.random() < 0.5:                       # blob
            y, x = int(rng.integers(0, H)), int(rng.integers(0, W))
            h, w = int(rng.integers(1, 12)), int(rng.integers(1, 25))
            m[y:y + h, x:x + w] = 1
        else:                                        # line
            y0, x0 = int(rng.integers(0, H)), int(rng.integers(0, W))
            m[y0:y0 + 1, x0:min(W, x0 + 20)] = 1
    return m


def test_windowed_greedy_fill_matches_the_full_raster_version():
    rng = np.random.default_rng(0)
    for trial in range(12):
        H, W = int(rng.integers(20, 90)), int(rng.integers(20, 90))
        m = _random_mask(rng, H, W)
        p = rng.random((H, W)).astype(np.float32) if trial % 3 == 0 else None
        for R in (1, 3):
            a = dominant_thin(m, R=R, p=p)
            b = _reference_dominant_thin(m, R=R, p=p)
            assert int(np.count_nonzero(a != b)) == 0, f"trial {trial}, R={R}"


def test_no_mask_and_empty_selection_are_handled():
    assert dominant_thin(np.zeros((10, 10), np.float32)).sum() == 0
    # a single pixel survives unchanged
    m = np.zeros((10, 10), np.float32)
    m[5, 5] = 1
    assert dominant_thin(m, R=3).sum() == 1


def test_thinning_still_dominates_every_original_pixel():
    """The contract: every pixel of the input mask stays within R of a kept pixel."""
    rng = np.random.default_rng(5)
    m = _random_mask(rng, 64, 64)
    kept = dominant_thin(m, R=3) > 0.5
    d = distance_transform_edt(~kept)
    assert (d[m > 0.5] <= 3 + 1e-9).all()
