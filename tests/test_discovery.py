"""Tests for the discovery diagnostics (src/discovery.py).

Why these matter: the competition scores the NEW fault dataset (rules §1.1), which is
disjoint from the only labels we can see (existing faults, rules §3.3).  A model that
reproduces the visible catalog scores ~1.0 on catalog-DTI and ~0 on the competition target.
`novel_fraction` and `candidate_new_faults` are the only numbers in this repo that can tell
those two situations apart, so they have to be right.

Run: python tests/test_discovery.py   (or python -m pytest tests -q)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.discovery import (candidate_new_faults, credit_map_known, discovery_report,  # noqa: E402
                           novel_mass)
from src.metrics import compute_distance_weighted_tversky  # noqa: E402

R = 3


def _blob(shape, y, x, h=1, w=5):
    a = np.zeros(shape, np.float32)
    a[y:y + h, x:x + w] = 1.0
    return a


def test_novel_mass_is_zero_when_the_prediction_is_the_catalog():
    cat = _blob((40, 40), 20, 10)
    out = novel_mass(cat, cat, R=R)
    assert abs(out["novel_mass"]) < 1e-9, out
    assert abs(out["novel_fraction"]) < 1e-9
    assert abs(out["prob_mass"] - cat.sum()) < 1e-9


def test_novel_mass_matches_the_metric_fp_weight_definition():
    """novel_mass(p, G) must equal FP_w from the official metric when the catalog is the GT."""
    cat = _blob((40, 40), 10, 10)
    pred = np.zeros((40, 40), np.float32)
    pred[5:9, 5:9] = 0.8          # far from the catalog
    pred[10:12, 10:14] = 0.5      # on top of the catalog
    dti, (tp, fp, fn) = compute_distance_weighted_tversky(pred, cat, R_pixels=R,
                                                         return_components=True)
    out = novel_mass(pred, cat, R=R)
    assert abs(out["novel_mass"] - fp) < 1e-9, (out["novel_mass"], fp)


def test_mass_within_R_of_the_catalog_is_not_novel():
    cat = _blob((40, 40), 20, 10, w=1)          # single cataloged pixel at (20, 10)
    pred = np.zeros((40, 40), np.float32)
    pred[20, 10 + R] = 1.0        # exactly R away -> k(d)=0 -> fully novel (weight 1)
    pred[20, 10 + R - 1] = 1.0    # R-1 away -> k=1/3 -> partially novel (weight 2/3)
    out = novel_mass(pred, cat, R=R)
    assert abs(out["novel_mass"] - (1.0 + 2 / 3)) < 1e-9, out
    assert abs(out["catalog_adjacent_mass"] - 1 / 3) < 1e-9
    assert abs(out["novel_fraction"] - (1 + 2 / 3) / 2) < 1e-9


def test_credit_map_is_the_triangular_kernel_of_the_nearest_catalog_pixel():
    cat = np.zeros((21, 21), np.float32)
    cat[10, 10] = 1.0
    m = credit_map_known(cat, R=R)
    assert abs(m[10, 10] - 1.0) < 1e-12
    assert abs(m[10, 11] - 2 / 3) < 1e-9
    assert abs(m[10, 12] - 1 / 3) < 1e-9
    assert m[10, 13] == 0.0
    assert m[0, 0] == 0.0


def test_candidate_new_faults_are_the_components_away_from_the_catalog():
    cat = _blob((60, 60), 30, 5, h=1, w=10)                  # known fault
    pred = np.zeros((60, 60), np.float32)
    pred[30, 5:15] = 1.0                                     # the known fault (touching)
    pred[5, 40:50] = 1.0                                     # a discovery
    out = candidate_new_faults(pred, cat, R=R, thr=0.5, min_px=3)
    assert out["n_components"] == 2, out["n_components"]
    assert out["n_novel_components"] == 1, out["n_novel_components"]
    assert out["novel_px"] == 10, out["novel_px"]
    assert out["novel_bboxes"][0]["length_px"] == 10
    # a discovery 1 px from the catalog is NOT novel (within R)
    pred2 = np.zeros((60, 60), np.float32)
    pred2[29, 5:15] = 1.0
    assert candidate_new_faults(pred2, cat, R=R, thr=0.5)["n_novel_components"] == 0
    # min_px filters speckle
    pred3 = np.zeros((60, 60), np.float32)
    pred3[5, 40] = 1.0
    assert candidate_new_faults(pred3, cat, R=R, thr=0.5, min_px=3)["n_novel_components"] == 0
    assert candidate_new_faults(pred3, cat, R=R, thr=0.5, min_px=1)["n_novel_components"] == 1


def test_novel_component_px_matches_bboxes_largest_first():
    """Regression (session 6): novel_bboxes is largest-first and min_px-filtered, but
    novel_component_px used to be the last 25 component ids in label order, unfiltered —
    the two fields could disagree.  Both now derive from the same boxes."""
    cat = _blob((60, 60), 30, 5, h=1, w=10)
    pred = np.zeros((60, 60), np.float32)
    pred[5, 40:52] = 1.0                                     # 12 px discovery
    pred[10, 40:46] = 1.0                                    # 6 px discovery
    pred[15, 40:42] = 1.0                                    # 2 px speckle (below min_px)
    out = candidate_new_faults(pred, cat, R=R, thr=0.5, min_px=3)
    assert out["novel_component_px"] == [b["px"] for b in out["novel_bboxes"]]
    assert out["novel_component_px"] == [12, 6]
    assert out["novel_component_px"] == sorted(out["novel_component_px"], reverse=True)
    assert sum(out["novel_component_px"]) == out["novel_px"]


def test_discovery_report_is_json_safe_and_separates_the_two_universes():
    cat = _blob((50, 50), 25, 5, h=1, w=20)
    copycat = cat.copy()                       # perfect catalog reproduction ...
    discovery = np.zeros((50, 50), np.float32)
    discovery[5, 5:25] = 1.0                   # ... plus a disjoint fault
    a = discovery_report(discovery, cat, R=R, thresholds=(0.1, 0.5))
    b = discovery_report(copycat, cat, R=R, thresholds=(0.1, 0.5))
    import json
    json.dumps(a)
    # catalog recall: the discovery-only prediction misses the catalog entirely
    assert a["catalog_recall_R_t0.5"] == 0.0
    assert b["catalog_recall_R_t0.5"] == 1.0
    # catalog DTI: the copy scores ~1.0 while proposing nothing new; the discovery scores 0
    assert compute_distance_weighted_tversky(copycat, cat, R_pixels=R) > 0.99
    assert compute_distance_weighted_tversky(discovery, cat, R_pixels=R) == 0.0
    assert b["novel_mass"]["novel_fraction"] < 1e-9
    assert a["novel_mass"]["novel_fraction"] > 0.99
    assert a["per_threshold"]["0.5"]["novel_px"] == 20


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
