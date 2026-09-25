"""The emission band: hard vs ramp, and why the difference is a measurement, not a preference.

Session 10 measured that *widening* the emitted band is the single largest shaping gain available on
the new-fault-like population (0.0247 -> 0.0713 proxy DTI from a 0 px skeleton to a 16 px band), and
that 74 % of the scored-like truth is more than 12 px away from any emitted pixel.  Widening raises a
second question that the width sweep cannot answer: what VALUES should be written into the band?

    hard band (previous behaviour) : 1.0 everywhere inside the band
    ramp  (src.submission_optim.soft_band) : 1.0 on the skeleton, decaying linearly to 0 at the edge

Both have the SAME support, so scoring them against the same truth isolates the values from the
geometry (scripts/eval_proxy_catalogue.py --soft-band does exactly that).  Per the official metric a
pixel beyond R = 3 px of a scored fault is charged alpha = 0.2 per unit of probability, while a truth
pixel is credited through max_x p(x) k(d(x,g)) - so the ramp charges the far ring less and also
credits it less.  Which effect wins is an empirical question; these tests pin the operator's
mathematics so that the measurement is of the metric, not of a coding accident.

Run: python -m pytest tests/test_shaping_band.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_optim import (dilate_mask, floor_sharpen, optimize_submission,  # noqa: E402
                                  soft_band)


def _skeleton(h: int = 40, w: int = 40) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[10:30, 20] = 1.0
    return m


def test_band_is_one_on_the_skeleton_and_zero_outside_the_width():
    m = _skeleton()
    b = soft_band(m, width=5)
    assert np.all(b[m > 0.5] == 1.0), "the skeleton itself must stay at full probability"
    # a pixel farther than the band width from the skeleton is exactly 0 (not small-but-nonzero:
    # FP_w sums probability mass, see the module docstring)
    assert b[10, 20 + 6] == 0.0 and b[10, 20 - 6] == 0.0
    # the skeleton runs for rows 10..29, so 6 px past its tip is outside a 5 px band
    assert b[4, 20] == 0.0 and b[35, 20] == 0.0
    assert b[5, 20] == np.float32(1.0 - 5 / 6), "the band edge keeps a positive value"


def test_band_values_are_linear_in_the_distance():
    m = _skeleton()
    width = 4
    b = soft_band(m, width=width)
    # along row 10 (the skeleton row) the distance to the skeleton column 20 grows by 1 px per step;
    # the denominator is width + 1 so that the outermost ring keeps a positive value while the
    # support stays exactly the hard band's (see soft_band's docstring)
    assert b[10, 20] == 1.0
    for k in range(1, width + 1):
        assert b[10, 20 + k] == np.float32(1.0 - k / (width + 1)), f"not linear at {k} px"
    assert b[10, 20 + width + 1] == 0.0


def test_band_support_equals_the_hard_dilation_support():
    """Same support, different values - that is what makes the comparison a clean A/B."""
    m = _skeleton()
    for width in (1, 2, 3, 6):
        ramp = soft_band(m, width=width)
        hard = dilate_mask(m, radius=width)
        assert np.array_equal(ramp > 0, hard > 0.5), f"support differs at width {width}"


def test_band_is_inside_the_hard_band_and_never_exceeds_one():
    m = _skeleton()
    ramp = soft_band(m, width=6, gamma=1.0)
    hard = dilate_mask(m, radius=6)
    assert ramp.max() <= 1.0 and ramp.min() >= 0.0
    assert np.all(ramp <= hard + 1e-6), "a ramp value may never exceed the hard band's 1.0"


def test_gamma_sharpens_the_ramp_without_moving_its_support():
    m = _skeleton()
    g1 = soft_band(m, width=6, gamma=1.0)
    g2 = soft_band(m, width=6, gamma=2.0)
    assert np.array_equal(g1 > 0, g2 > 0)
    interior = (g1 > 0) & (g1 < 1)
    assert np.all(g2[interior] <= g1[interior] + 1e-6)
    assert g2[interior].mean() < g1[interior].mean(), "gamma > 1 must reduce the ramp's mass"


def test_zero_width_and_empty_mask_are_identity():
    m = _skeleton()
    assert np.array_equal(soft_band(m, width=0), m.astype(np.float32))
    empty = np.zeros_like(m)
    assert soft_band(empty, width=4).sum() == 0.0


def test_optimize_submission_soft_flag_is_the_same_support_as_the_hard_band():
    """The flag the sweep uses; a mismatch here would silently compare different supports."""
    p = np.zeros((40, 40), dtype=np.float32)
    p[10:30, 19:22] = 0.4                   # a low-confidence shoulder that the floor removes
    p[10:30, 20] = 0.9                      # the ridge (assigned AFTER the shoulder on purpose)
    hard = optimize_submission(p, t0=0.5, thin=True, dilate=4, soft=False)
    ramp = optimize_submission(p, t0=0.5, thin=True, dilate=4, soft=True)
    assert np.array_equal(hard > 0, ramp > 0), "soft=True must not change the support"
    assert ramp.max() <= 1.0
    # the hard band is uniform 1.0; the ramp is not
    assert hard[hard > 0].min() == 1.0
    assert ramp[ramp > 0].min() < 1.0


def test_floor_sharpen_zeros_the_background_exactly():
    p = np.array([[0.5, 0.0], [0.25, 1.0]], dtype=np.float32)
    q = floor_sharpen(p, t0=0.3, hard=True)
    assert q[0, 1] == 0.0 and q[1, 0] == 0.0 and q[0, 0] == 1.0 and q[1, 1] == 1.0
