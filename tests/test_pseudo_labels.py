"""Pseudo-label training signal (src/dataset.py, configs/config_pseudo_labels.yaml).

The SGMC proxy trace is the only remaining external-catalogue route (the QFaults transfer
experiment was refused on a total overlap with the training labels).  Training on it is
legitimate (problem page #external-datasets) ONLY if the gain is measured under a spatial
hold-out so it cannot be label leakage.  These tests pin the leakage design in
`make_patches`:

* baseline invariance - an all-False pseudo mask (or none) leaves every array byte-identical,
  so a no-pseudo run and a pseudo run differ in exactly one controlled way;
* pseudo pixels become label mass in TRAINING windows only; test windows are untouched;
* the FP-weight map credits pseudo pixels in the training region (they are faults the loss
  should credit) but NOT in the held-out region (the same leakage rule as the labels);
* the partition and the window selection keep using the original labels alone, so the
  pseudo run and the baseline run train on the SAME windows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_pseudo_mask, make_patches  # noqa: E402


def test_load_pseudo_mask_on_committed_proxy_catalogue():
    mask = load_pseudo_mask(str(ROOT / "data" / "evidence" / "proxy" / "proxy_catalogue.tif"),
                            code=2)
    # pinned to the committed measurement (data/evidence/proxy/proxy_stats.json)
    assert mask.shape == (3730, 3292)
    assert int(mask.sum()) == 61_664


def test_load_pseudo_mask_rejects_wrong_code():
    with pytest.raises(ValueError, match="no pixels with code"):
        load_pseudo_mask(str(ROOT / "data" / "evidence" / "proxy" / "proxy_catalogue.tif"),
                         code=7)


def _grid(n=1024, c=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n, c)).astype(np.float32)
    return X


def test_baseline_invariance_with_empty_pseudo_mask():
    """pseudo=None and pseudo=all-False must be byte-identical - the experiment is off."""
    X = _grid()
    y = np.zeros(X.shape[:2], np.float32)
    rng = np.random.default_rng(0)
    y[rng.integers(0, 512, 200), rng.integers(0, 512, 200)] = 1.0
    kw = dict(patch_size=256, train_step=128, seed=0, neg_fraction=0.35, R_pixels=3,
              holdout="random", test_proportion=0.3)
    base = make_patches(X, y, pseudo=None, **kw)
    off = make_patches(X, y, pseudo=np.zeros_like(y, dtype=bool), pseudo_weight=1.0, **kw)
    for key in ("X_train", "y_train", "fpw_train", "X_test", "y_test"):
        assert np.array_equal(base[key], off[key]), key
    assert base["summary"]["train_windows"] == off["summary"]["train_windows"]
    assert base["summary"]["pseudo"]["enabled"] is False
    assert off["summary"]["pseudo"]["enabled"] is False


def test_pseudo_applied_to_training_windows_only():
    """Spatial hold-out: pseudo mass enters training windows, never test windows or the
    held-out region of the FP map."""
    from src.blocks import assign_folds, block_id_map, block_table

    n = 1024
    X = _grid(n, c=2, seed=1)
    y = np.zeros((n, n), np.float32)
    rng = np.random.default_rng(1)
    y[rng.integers(0, n, 400), rng.integers(0, n, 400)] = 1.0

    block_px, patch, block_folds, fold = 512, 256, 2, 0
    valid = np.isfinite(X).any(axis=-1)
    tab = block_table((n, n), block_px, valid=valid, labels=y > 0.5)
    fold_of = assign_folds(tab, block_folds, seed=0, mode="balanced")
    blocks = block_id_map((n, n), block_px)
    held = np.isin(blocks, [b["block_id"] for b in tab
                            if fold_of.get(b["block_id"]) == fold])
    train = ~held
    assert held.any() and train.any()

    # pseudo pixels: a few on training-region labels, a few OFF any label in the training
    # region (the actual pseudo signal), and a few inside the HELD-OUT region (leakage probe)
    pseudo = np.zeros((n, n), dtype=bool)
    off_label = train & (y == 0)
    idx = np.flatnonzero(off_label.ravel())
    rng2 = np.random.default_rng(2)
    pick = rng2.choice(idx, size=30, replace=False)
    pseudo.ravel()[pick] = True
    # held-out probe pixels: off-label AND > R px from any label, so the "fpw == 1.0" check
    # below cannot be confounded by a real label's kernel
    from scipy.ndimage import distance_transform_edt
    d_lab = distance_transform_edt(~(y > 0.5))
    held_off = held & (y == 0) & (d_lab > 3)
    pick2 = np.random.default_rng(3).choice(np.flatnonzero(held_off.ravel()), size=20,
                                            replace=False)
    pseudo.ravel()[pick2] = True

    kw = dict(patch_size=patch, train_step=patch, seed=0, neg_fraction=0.35, R_pixels=3,
              holdout="spatial_blocks", block_px=block_px, block_folds=block_folds,
              block_fold=fold, block_mode="balanced", block_seed=0)
    base = make_patches(X, y, pseudo=None, **kw)
    pl = make_patches(X, y, pseudo=pseudo, pseudo_weight=1.0, **kw)

    # 1) identical windows - the experiment changed nothing about the training set
    assert base["summary"]["train_windows"] == pl["summary"]["train_windows"]
    assert base["summary"]["holdout"] == pl["summary"]["holdout"]
    # identical features and baseline labels everywhere
    assert np.array_equal(base["X_train"], pl["X_train"])
    assert np.array_equal(base["X_test"], pl["X_test"])
    assert np.array_equal(base["y_test"], pl["y_test"]), \
        "pseudo pixels leaked into the TEST labels - the held-out measurement is corrupted"

    # 2) training labels carry the pseudo mass, and only there: for every window the diff
    #    against the baseline is exactly 1.0 on (pseudo & label-empty) pixels and 0 elsewhere
    assert pl["summary"]["pseudo"]["enabled"] is True
    assert pl["summary"]["pseudo"]["train_px_added"] > 0
    for k, (i, j) in enumerate(pl["summary"]["train_windows"]):
        yb, yp = base["y_train"][k], pl["y_train"][k]
        expect = pseudo[i:i + patch, j:j + patch] & (yb <= 0.5)
        got = (yp != yb)
        assert (got == expect).all(), f"window {k}: pseudo mass applied to the wrong pixels"
        assert (yp[expect] == 1.0).all()

    # 3) FP weight: a prediction exactly on a training-region pseudo pixel is credited as a
    #    fault (fpw = 0, the kernel at distance 0); outside the pseudo pixels the FP map is
    #    unchanged.  Held-out pseudo pixels cannot leak through the FP map because the
    #    training windows do not touch the held-out blocks or their collar at all - which is
    #    asserted directly below (the held-region fpw is never handed to the loss).
    fpw_base = np.zeros((n, n), np.float32)
    fpw_pl = np.zeros((n, n), np.float32)
    for k, (i, j) in enumerate(pl["summary"]["train_windows"]):
        fpw_base[i:i + patch, j:j + patch] = base["fpw_train"][k]
        fpw_pl[i:i + patch, j:j + patch] = pl["fpw_train"][k]
    train_pseudo_px = pseudo & train
    on_pseudo = np.flatnonzero(train_pseudo_px.ravel())
    assert len(on_pseudo) > 0
    assert (fpw_pl.ravel()[on_pseudo] == 0.0).all(), \
        "pseudo pixels must remove the false-positive penalty in the training region"
    # the fpw kernel reaches R px, so the map may legitimately change within R px of a
    # pseudo pixel (their kernel now covers those pixels); farther away it must be exact
    reach = distance_transform_edt(~train_pseudo_px)
    no_pseudo = reach > 3
    assert no_pseudo.sum() > 1_000_000
    assert np.allclose(fpw_base[no_pseudo], fpw_pl[no_pseudo], atol=1e-6), \
        "FP map changed beyond the pseudo pixels' R-px kernel reach"

    # 4) the held-out probe: no training window touches the held blocks or their R-px collar,
    #    so the held-out pseudo pixels (20 of them, deliberately placed) enter no training
    #    signal at all - labels, FP map or window selection
    from src.blocks import held_out_mask
    excl = held_out_mask((n, n), block_px, fold_of, fold, buffer_px=3)
    assert (pseudo & held).sum() > 0, "test setup: the leakage probe must exist"
    for (i, j) in pl["summary"]["train_windows"]:
        assert not excl[i:i + patch, j:j + patch].any(), \
            "a training window touches the held-out blocks/collar - held-out pseudo pixels " \
            "would enter the training signal"


def test_pseudo_weight_below_half_raises():
    X = _grid(512, c=1, seed=2)
    y = np.zeros((512, 512), np.float32)
    y[100:101, 100:101] = 1.0
    pseudo = np.zeros((512, 512), dtype=bool)
    pseudo[200, 200] = True
    with pytest.raises(ValueError, match="pseudo_weight"):
        make_patches(X, y, pseudo=pseudo, pseudo_weight=0.3,
                     patch_size=256, train_step=128, seed=0)


def test_pseudo_shape_mismatch_raises():
    X = _grid(512, c=1, seed=3)
    y = np.zeros((512, 512), np.float32)
    with pytest.raises(ValueError, match="pseudo mask"):
        make_patches(X, y, pseudo=np.ones((300, 300), dtype=bool), pseudo_weight=1.0,
                     patch_size=256, train_step=128, seed=0)


def test_config_pins_the_experiment():
    """The pseudo config differs from its baseline in exactly the pseudo keys."""
    import yaml
    a = yaml.safe_load((ROOT / "configs" / "config_pseudo_labels.yaml").read_text())
    b = yaml.safe_load((ROOT / "configs" / "config_block_holdout.yaml").read_text())
    assert a["data"]["pseudo_label_path"] == "data/evidence/proxy/proxy_catalogue.tif"
    assert a["data"]["pseudo_code"] == 2
    assert a["training"]["pseudo_weight"] == 1.0
    a["data"].pop("pseudo_label_path")
    a["data"].pop("pseudo_code")
    a["training"].pop("pseudo_weight")
    # everything else - partition, geometry, schedule - identical to the baseline
    assert a == b, "the pseudo config must differ from the baseline ONLY in the pseudo keys"
