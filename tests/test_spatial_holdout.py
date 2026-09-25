"""Spatial block hold-out inside the patcher: the leakage properties that make it a hold-out.

`src/dataset.make_patches` grew a second hold-out design (2026-09-18; next-step item
"Spatial block-holdout" from EXECUTIVE_SUMMARY.md §11 / docs/DISCOVERY_PLAN.md §3(a)).
These tests pin the two things that would silently invalidate the experiment if they broke:

1. **the random path is unchanged** - three committed ensembles (folds 0-17) were trained by
   it, so its behaviour is part of the shipped submission's provenance;
2. **the block path really holds geography out** - no training window may touch the held-out
   blocks or their collar, every scored window must lie wholly inside them, the folds must
   partition the grid, and a fold's geography must not depend on the fold's RNG seed.

Synthetic grid: 512x512 with an 8-px NaN border (the real raster is 3292x3730 with 57.92 %
NaN outside the GeoDAWN footprint), 4 fault traces, 64-px patches at a 32-px step and 128-px
blocks - i.e. blocks are 2x2 patches, the same ratio the real config uses (512-px blocks,
256-px patches).  Block size == patch size is deliberately NOT used: a window then fits inside
a single block, and any held-out neighbour disqualifies it, which measured 4 surviving
training windows out of 49 (a degenerate fold, not a leakage check).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import assign_folds, block_table, held_out_mask, scored_mask  # noqa: E402
from src.dataset import make_patches                                          # noqa: E402

SHAPE = (512, 512)
PATCH = 64
BLOCK = 128
FOLDS = 4


def _stack(shape=(SHAPE[0], SHAPE[1], 3), seed=0, nan_border=8):
    """Synthetic feature stack + fault labels with a NaN border (like the real footprint)."""
    rng = np.random.default_rng(seed)
    H, W = shape[0], shape[1]
    X = rng.normal(size=shape).astype(np.float32)
    X[:nan_border, :, :] = np.nan
    X[-nan_border:, :, :] = np.nan
    y = np.zeros((H, W), np.float32)
    y[64:68, 40:480] = 1.0            # long E-W trace
    y[120:490, 300:304] = 1.0         # long N-S trace
    y[250:254, 60:400] = 1.0
    y[400:470, 150:154] = 1.0
    return X, y


BASE = dict(patch_size=PATCH, train_step=32, test_proportion=0.3, seed=0,
            neg_fraction=0.35, R_pixels=3, min_px_per_patch=3)
BLK = dict(BASE, holdout="spatial_blocks", block_px=BLOCK, block_folds=FOLDS, block_fold=0,
           block_buffer_px=3, block_mode="balanced", block_seed=0)


def _partition(X, y, fold=0, mode="balanced", buffer_px=3, block_px=BLOCK, n_folds=FOLDS):
    """The partition make_patches must be using (same defaults, same derivation)."""
    valid = np.isfinite(X).any(-1)
    tab = block_table(valid.shape, block_px, valid=valid, labels=y > 0.5)
    fold_of = assign_folds(tab, n_folds, seed=0, mode=mode)
    return (tab, fold_of,
            scored_mask(valid.shape, block_px, fold_of, fold),
            held_out_mask(valid.shape, block_px, fold_of, fold, buffer_px=buffer_px))


# --------------------------------------------------------------------------------------
# 1. the random path is unchanged
# --------------------------------------------------------------------------------------
def test_random_holdout_is_the_default():
    X, y = _stack()
    s = make_patches(X, y, **BASE)["summary"]
    assert s["holdout"] == {"mode": "random", "test_proportion": pytest.approx(0.3)}
    assert s["n_test"] > 0 and s["n_train"] > 0


def test_random_holdout_is_reproducible_and_seed_sensitive():
    X, y = _stack()
    a = make_patches(X, y, **BASE)["summary"]
    b = make_patches(X, y, **BASE)["summary"]
    assert a["test_windows"] == b["test_windows"], "same seed must give the same split"
    c = make_patches(X, y, **{**BASE, "seed": 7})["summary"]
    assert a["test_windows"] != c["test_windows"], "different seeds must give different splits"


def test_random_holdout_golden_counts():
    """Pinned so any change to the reference-style split is a deliberate, reviewed act.

    Values measured 2026-09-18 on the synthetic grid documented in this file's docstring.
    """
    X, y = _stack()
    s = make_patches(X, y, **BASE)["summary"]
    assert (s["n_test"], s["n_pos"], s["n_train"], s["H"], s["W"]) == (19, 36, 55, 512, 512)


def test_random_holdout_leaves_blocks_partly_held_out():
    """The adjacency leak the block path exists to remove must still be visible here."""
    X, y = _stack()
    s = make_patches(X, y, **BASE)["summary"]
    per_block = {}
    for (i, j) in [tuple(t) for t in s["test_windows"]]:
        per_block[(i // BLOCK, j // BLOCK)] = per_block.get((i // BLOCK, j // BLOCK), 0) + 1
    windows_per_block = (BLOCK // PATCH) ** 2
    partial = [b for b, n in per_block.items() if 0 < n < windows_per_block]
    assert partial, f"random hold-out unexpectedly tiled whole blocks: {per_block}"


def test_random_path_zeroes_its_own_test_windows():
    X, y = _stack()
    res = make_patches(X, y, **BASE)
    s = res["summary"]
    glob = np.zeros(X.shape[:2], np.float32)
    for patch, (i, j) in zip(res["y_train"], [tuple(t) for t in s["train_windows"]]):
        np.maximum(glob[i:i + PATCH, j:j + PATCH], patch, out=glob[i:i + PATCH, j:j + PATCH])
    held = np.zeros(X.shape[:2], bool)
    for (i, j) in [tuple(t) for t in s["test_windows"]]:
        held[i:i + PATCH, j:j + PATCH] = True
    assert not (glob[held] > 0.5).any(), "random-path zeroing regressed"


def test_invalid_holdout_mode_is_rejected():
    X, y = _stack()
    with pytest.raises(ValueError):
        make_patches(X, y, **{**BASE, "holdout": "spatial"})


# --------------------------------------------------------------------------------------
# 2. the block path holds geography out
# --------------------------------------------------------------------------------------
def test_every_scored_window_lies_wholly_inside_the_held_out_blocks():
    X, y = _stack()
    res = make_patches(X, y, **BLK)
    s = res["summary"]
    assert s["holdout"]["mode"] == "spatial_blocks"
    _, _, sc, _ = _partition(X, y)
    for (i, j) in [tuple(t) for t in s["test_windows"]]:
        assert sc[i:i + PATCH, j:j + PATCH].all(), f"window {(i, j)} leaves the held-out blocks"
    assert s["n_test"] > 0, "a fold with no scored window cannot select anything"
    # and the scored windows tile exactly the valid part of the held-out blocks
    covered = np.zeros(X.shape[:2], bool)
    for (i, j) in [tuple(t) for t in s["test_windows"]]:
        covered[i:i + PATCH, j:j + PATCH] = True
    valid = np.isfinite(X).any(-1)
    assert covered.sum() >= int((sc & valid).sum()) - 4 * PATCH, \
        "scored windows should cover the held-out blocks' valid area"


def test_no_training_window_touches_the_held_out_blocks_or_their_collar():
    X, y = _stack()
    res = make_patches(X, y, **BLK)
    s = res["summary"]
    _, _, _, excl = _partition(X, y)
    trains = [tuple(t) for t in s["train_windows"]]
    assert trains, "no training window survived the collar - the fold cannot train"
    for (i, j) in trains:
        assert not excl[i:i + PATCH, j:j + PATCH].any(), \
            f"training window {(i, j)} reaches into the held-out collar"
    # the collar must actually have removed windows the un-collared path would have kept
    no_collar = make_patches(X, y, **{**BLK, "block_buffer_px": 0})["summary"]["n_train"]
    assert s["n_train"] <= no_collar, \
        f"a 3 px collar kept MORE windows than none ({s['n_train']} > {no_collar})"


def test_training_labels_from_held_out_geography_are_zeroed():
    """Reconstruct the global training label map from the returned patches: no pixel inside a
    held-out block may carry a positive label."""
    X, y = _stack()
    res = make_patches(X, y, **BLK)
    _, _, sc, _ = _partition(X, y)
    glob = np.zeros(X.shape[:2], np.float32)
    for patch, (i, j) in zip(res["y_train"], [tuple(t) for t in res["summary"]["train_windows"]]):
        np.maximum(glob[i:i + PATCH, j:j + PATCH], patch, out=glob[i:i + PATCH, j:j + PATCH])
    assert not (glob[sc] > 0.5).any(), "a held-out label survived into the training signal"
    assert (glob > 0.5).sum() > 0, "the fold kept no positive labels at all"
    # the held-out truth itself is still present in the returned test labels
    assert (res["y_test"] > 0.5).sum() > 0, "fold 0 has no held-out fault pixels to score"


def test_fp_weight_is_a_valid_probability_weight_map():
    """fpw < 1 means "a visible training fault is within R px"; it must stay in [0, 1] and be 0
    exactly on the labels the fold was allowed to see."""
    X, y = _stack()
    res = make_patches(X, y, **BLK)
    fpw, y_tr = res["fpw_train"], res["y_train"]
    assert fpw.shape == y_tr.shape
    assert float(fpw.min()) >= 0.0 and float(fpw.max()) <= 1.0
    assert float(fpw[y_tr > 0.5].max()) == pytest.approx(0.0, abs=1e-6)
    assert float(fpw[y_tr < 0.5].max()) == pytest.approx(1.0, abs=1e-6), \
        "far-from-fault pixels must carry full false-positive weight"


def test_folds_partition_the_grid_and_are_pairwise_disjoint():
    X, y = _stack()
    seen = np.zeros(SHAPE, bool)
    total_scored = 0
    for f in range(FOLDS):
        res = make_patches(X, y, **{**BLK, "block_fold": f})
        assert res["summary"]["n_test"] > 0, f"fold {f} scored nothing"
        assert res["summary"]["n_train"] > 0, f"fold {f} has no training windows"
        _, _, sc, _ = _partition(X, y, fold=f)
        assert not (sc & seen).any(), f"fold {f} re-scores geography another fold held out"
        seen |= sc
        total_scored += int(sc.sum())
    assert total_scored == int(seen.sum())


def test_block_geography_does_not_depend_on_the_fold_rng_seed():
    """Fold identity is its geography: `seed` drives augmentation/negative sampling only."""
    X, y = _stack()
    a = make_patches(X, y, **BLK)["summary"]["test_windows"]
    b = make_patches(X, y, **{**BLK, "seed": 123})["summary"]["test_windows"]
    assert a == b, "the held-out blocks must be fixed by block_seed, not by the fold seed"
    assert make_patches(X, y, **BLK)["summary"]["holdout"]["fold"] == \
        make_patches(X, y, **{**BLK, "seed": 123})["summary"]["holdout"]["fold"]


def test_balanced_mode_spreads_fault_mass_more_evenly_than_contiguous():
    """The measured reason to default to `balanced` for selection (and to report both)."""
    X, y = _stack()
    masses = {}
    for mode in ("balanced", "contiguous"):
        m = []
        for f in range(FOLDS):
            ho = make_patches(X, y, **{**BLK, "block_fold": f, "block_mode": mode})["summary"]["holdout"]
            m.append(ho["fold"]["fault_px"])
        assert sum(m) == int(((y > 0.5) & np.isfinite(X).any(-1)).sum()), \
            f"{mode}: held-out fault mass must sum to the labelled total"
        masses[mode] = m
    spread = lambda v: (max(v) - min(v)) / max(1.0, float(np.mean(v)))  # noqa: E731
    assert spread(masses["balanced"]) <= spread(masses["contiguous"]), \
        f"balanced {masses['balanced']} should be tighter than contiguous {masses['contiguous']}"


def test_contiguous_block_mode_keeps_the_fold_in_one_piece():
    X, y = _stack()
    ho = make_patches(X, y, **{**BLK, "block_mode": "contiguous"})["summary"]["holdout"]
    assert ho["block_mode"] == "contiguous"
    nbx = SHAPE[1] // BLOCK
    cells = {(b // nbx, b % nbx) for b in ho["fold"]["block_ids"]}
    seen, stack = set(), [next(iter(cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in cells and nb not in seen:
                stack.append(nb)
    assert seen == cells, "contiguous mode must hold out one connected super-region"


def test_collar_width_is_reported_and_monotone():
    X, y = _stack()
    ho0 = make_patches(X, y, **{**BLK, "block_buffer_px": 0})["summary"]["holdout"]
    assert ho0["buffer_px"] == 0 and ho0["fold"]["collar_px"] == 0
    ho_default = make_patches(X, y, **{**BLK, "block_buffer_px": None})["summary"]["holdout"]
    assert ho_default["buffer_px"] == BASE["R_pixels"], "the default collar is the metric radius R"
    n3 = make_patches(X, y, **{**BLK, "block_buffer_px": 3})["summary"]["n_train"]
    n16 = make_patches(X, y, **{**BLK, "block_buffer_px": 16})["summary"]["n_train"]
    assert n16 <= n3, f"a wider collar must not add training windows ({n16} > {n3})"


def test_block_fold_out_of_range_is_rejected():
    X, y = _stack()
    with pytest.raises(ValueError):
        make_patches(X, y, **{**BLK, "block_fold": FOLDS})


def test_holdout_record_carries_the_partition_for_reproduction():
    """Rules §3.5: the assets must reproduce the result, so the geography is recorded."""
    X, y = _stack()
    ho = make_patches(X, y, **BLK)["summary"]["holdout"]
    part = ho["partition"]
    assert part["block_px"] == BLOCK and part["n_folds"] == FOLDS and part["buffer_px"] == 3
    assert part["block_shape"] == [SHAPE[0] // BLOCK, SHAPE[1] // BLOCK]
    assert part["mode"] == "balanced"
    assert sum(e["blocks"] for e in part["per_fold"]) + part["empty_blocks_excluded"] == part["n_blocks"]
    fold = ho["fold"]
    assert fold["scored_px"] > 0
    assert fold["excluded_px"] >= fold["scored_px"]
    assert fold["collar_px"] == fold["excluded_px"] - fold["scored_px"]
    assert fold["fault_px"] > 0
