"""Spatial block hold-out partition: the properties the selection signal depends on.

Every property below is one that, if it silently broke, would corrupt a model-selection
decision rather than crash:

* determinism      - a fold's identity is its geography; it must be reproducible from config
* contiguity       - a fold must be whole blocks, not a scatter (that is the point of blocks)
* balance          - a fold with no faults cannot select anything
* buffer           - training must not touch labels within R of the held-out truth
* footprint        - 57.92 % of the grid is NaN; balance on valid pixels, not on area
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (DEFAULT_BLOCK_PX, assign_folds, block_id_map, block_shape,  # noqa: E402
                        block_slices, block_table, describe_partition, held_out_mask,
                        scored_mask, super_grid)


def _synthetic(shape=(256, 320), block_px=64, fault_stripes=True):
    """A small grid with a NaN collar and fault mass concentrated in two bands."""
    H, W = shape
    rng = np.random.default_rng(7)
    valid = np.ones(shape, bool)
    valid[:, :32] = False                       # pretend footprint edge
    valid[-16:, :] = False
    labels = np.zeros(shape, bool)
    if fault_stripes:
        labels[40:44, 40:280] = True            # a long E-W trace
        labels[150:260, 200:204] = True         # a long N-S trace
    return valid, labels


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def test_block_shape_tiles_partial_blocks():
    assert block_shape((3730, 3292), 512) == (8, 7)          # ceil, last block partial
    assert block_shape((512, 512), 512) == (1, 1)
    assert block_shape((513, 512), 512) == (2, 1)
    with pytest.raises(ValueError):
        block_shape((10, 10), 0)


def test_block_slices_cover_the_grid_exactly_once():
    shape, bp = (300, 260), 64
    covered = np.zeros(shape, int)
    for (r0, r1, c0, c1) in block_slices(shape, bp):
        assert r1 <= shape[0] and c1 <= shape[1] and r1 > r0 and c1 > c0
        covered[r0:r1, c0:c1] += 1
    assert covered.min() == 1 and covered.max() == 1, "blocks must tile without gaps or overlap"
    assert len(block_slices(shape, bp)) == int(np.prod(block_shape(shape, bp)))


def test_block_id_map_matches_slices():
    shape, bp = (300, 260), 64
    ids = block_id_map(shape, bp)
    for bid, (r0, r1, c0, c1) in enumerate(block_slices(shape, bp)):
        assert set(np.unique(ids[r0:r1, c0:c1]).tolist()) == {bid}


def test_block_table_counts_valid_and_fault_pixels():
    valid, labels = _synthetic()
    tab = block_table(valid.shape, 64, valid=valid, labels=labels)
    assert sum(t["valid_px"] for t in tab) == int(valid.sum())
    # faults are ANDed with the footprint, so a trace over NaN cannot be scored
    assert sum(t["fault_px"] for t in tab) == int((labels & valid).sum())
    assert all(t["fault_km"] == pytest.approx(t["fault_px"] * 0.1, abs=1e-6) for t in tab)


def test_block_table_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        block_table((100, 100), 32, valid=np.ones((99, 100), bool))
    with pytest.raises(ValueError):
        block_table((100, 100), 32, labels=np.ones((100, 99), bool))


# --------------------------------------------------------------------------------------
# fold assignment
# --------------------------------------------------------------------------------------
def test_assignment_is_deterministic_for_a_seed():
    valid, labels = _synthetic(shape=(512, 512), block_px=64)
    tab = block_table(valid.shape, 64, valid=valid, labels=labels)
    a = assign_folds(tab, n_folds=4, seed=11)
    b = assign_folds(tab, n_folds=4, seed=11)
    assert a == b, "the same config must give the same geography (rules 3.5 reproducibility)"
    c = assign_folds(tab, n_folds=4, seed=12)
    assert set(c) == set(a), "the block set must not depend on the seed"


def test_every_scored_block_gets_exactly_one_fold():
    valid, labels = _synthetic(shape=(512, 512), block_px=64)
    tab = block_table(valid.shape, 64, valid=valid, labels=labels)
    fold_of = assign_folds(tab, n_folds=4, seed=3)
    assert len(fold_of) == len(tab)
    for t in tab:
        f = fold_of[t["block_id"]]
        if t["valid_px"] > 0:
            assert 0 <= f < 4
        else:
            assert f == -1, "footprint-empty blocks are unscoreable and must be excluded"


def test_folds_are_balanced_on_fault_mass():
    """The greedy must beat both extremes: no empty fold, and no fold holding everything."""
    valid, labels = _synthetic(shape=(512, 512), block_px=64)
    tab = block_table(valid.shape, 64, valid=valid, labels=labels)
    fold_of = assign_folds(tab, n_folds=4, seed=5)
    mass = [sum(t["fault_px"] for t in tab if fold_of[t["block_id"]] == f) for f in range(4)]
    total = sum(mass)
    assert total > 0
    assert min(mass) > 0, f"a fold with no fault pixels cannot select a model: {mass}"
    assert max(mass) < total, "one fold holding every fault is not a hold-out"
    # greedy multiway partition: worst fold within 2x the ideal share
    assert max(mass) <= 2.0 * total / 4 + 1, f"unbalanced: {mass}"


def test_valid_pixel_balance_across_folds():
    valid, labels = _synthetic(shape=(512, 512), block_px=64)
    tab = block_table(valid.shape, 64, valid=valid, labels=labels)
    fold_of = assign_folds(tab, n_folds=4, seed=5)
    vp = [sum(t["valid_px"] for t in tab if fold_of[t["block_id"]] == f) for f in range(4)]
    assert min(vp) > 0
    assert max(vp) <= 2.0 * sum(vp) / 4 + 1, f"footprint area unbalanced: {vp}"


def test_assign_folds_rejects_bad_fold_count():
    tab = block_table((64, 64), 32)
    with pytest.raises(ValueError):
        assign_folds(tab, n_folds=0)


# --------------------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------------------
def test_held_out_mask_is_a_union_of_whole_blocks():
    shape, bp = (256, 256), 64
    tab = block_table(shape, bp)
    fold_of = assign_folds(tab, n_folds=4, seed=1)
    m = scored_mask(shape, bp, fold_of, 0)
    ids = block_id_map(shape, bp)
    for bid in np.unique(ids):
        sel = ids == bid
        frac = m[sel].mean()
        assert frac in (0.0, 1.0), f"block {bid} is {frac:.2f} held out - blocks must be atomic"


def test_buffer_dilates_the_collar_and_never_shrinks_the_truth():
    shape, bp = (256, 256), 64
    tab = block_table(shape, bp)
    fold_of = assign_folds(tab, n_folds=4, seed=1)
    sc = scored_mask(shape, bp, fold_of, 0)
    buf = held_out_mask(shape, bp, fold_of, 0, buffer_px=3)
    assert buf[sc].all(), "the collar must contain every scored pixel"
    assert buf.sum() > sc.sum(), "a positive buffer must exclude a collar of training pixels"
    # the collar is exactly 3 px thick: 4 px away from the truth is untouched
    from scipy.ndimage import binary_dilation
    assert buf.sum() == binary_dilation(sc, structure=np.ones((7, 7), bool)).sum()
    assert held_out_mask(shape, bp, fold_of, 0, buffer_px=0).sum() == sc.sum()


def test_folds_are_disjoint_on_their_scored_blocks():
    shape, bp = (256, 256), 64
    tab = block_table(shape, bp)
    fold_of = assign_folds(tab, n_folds=4, seed=2)
    masks = [scored_mask(shape, bp, fold_of, f) for f in range(4)]
    total = sum(int(m.sum()) for m in masks)
    assert int(np.logical_or.reduce(masks).sum()) == total, "scored blocks must not overlap"


def test_empty_fold_raises_instead_of_reporting_zero():
    shape, bp = (256, 256), 64
    tab = block_table(shape, bp)
    fold_of = {t["block_id"]: -1 for t in tab}
    with pytest.raises(ValueError):
        scored_mask(shape, bp, fold_of, 0)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def test_describe_partition_is_derived_from_the_assignment():
    valid, labels = _synthetic(shape=(512, 512), block_px=64)
    tab = block_table(valid.shape, 64, valid=valid, labels=labels)
    fold_of = assign_folds(tab, n_folds=4, seed=5)
    rep = describe_partition(valid.shape, 64, 4, 5, tab, fold_of, buffer_px=3,
                             labels=labels, valid=valid)
    assert rep["n_blocks"] == len(tab)
    assert sum(e["fault_px"] for e in rep["per_fold"]) == sum(t["fault_px"] for t in tab)
    assert sum(e["valid_px"] for e in rep["per_fold"]) == sum(t["valid_px"] for t in tab)
    assert all(e["collar_excluded_px"] > 0 for e in rep["per_fold"])
    assert rep["balance"]["fault_px_spread"] >= 0.0
    assert rep["block_km"] == pytest.approx(6.4)


def test_describe_partition_on_the_real_grid_shape():
    """The competition grid: 3292x3730 at 100 m, 512-px blocks -> 8x7 = 56 blocks."""
    shape = (3730, 3292)
    tab = block_table(shape, DEFAULT_BLOCK_PX)
    assert len(tab) == 56
    fold_of = assign_folds(tab, n_folds=4, seed=0)
    rep = describe_partition(shape, DEFAULT_BLOCK_PX, 4, 0, tab, fold_of)
    assert rep["block_shape"] == [8, 7]
    assert rep["block_km"] == pytest.approx(51.2)
    assert sum(e["blocks"] for e in rep["per_fold"]) == 56


# --------------------------------------------------------------------------------------
# contiguous mode (geographic extrapolation rather than block interpolation)
# --------------------------------------------------------------------------------------
def test_super_grid_prefers_aspect_matching_2d_cuts():
    assert super_grid(8, 7, 4) == ("grid", 2, 2)
    kind, a, b = super_grid(8, 7, 5)          # prime: bands along the longer axis
    assert kind == "bands" and a == "row" and b == 5
    assert super_grid(4, 10, 6)[0] == "grid"


def test_contiguous_folds_are_connected_regions():
    shape, bp = (512, 512), 64
    tab = block_table(shape, bp)
    fold_of = assign_folds(tab, n_folds=4, seed=0, mode="contiguous")
    nbx = max(t["block_col"] for t in tab) + 1
    for f in range(4):
        cells = {(t["block_row"], t["block_col"]) for t in tab if fold_of[t["block_id"]] == f}
        assert cells, f"fold {f} is empty"
        # flood fill over 4-adjacency: one component == contiguous super-region
        seen, stack = set(), [next(iter(cells))]
        while stack:
            r, c = stack.pop()
            if (r, c) in seen:
                continue
            seen.add((r, c))
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen:
                    stack.append(nb)
        assert seen == cells, f"fold {f} is not contiguous: {len(seen)} of {len(cells)} reachable"


def test_contiguous_folds_partition_every_block():
    shape, bp = (512, 512), 64
    tab = block_table(shape, bp)
    fold_of = assign_folds(tab, n_folds=4, seed=0, mode="contiguous")
    assert len(fold_of) == len(tab)
    assert sorted(fold_of.values()) == sorted([0] * 16 + [1] * 16 + [2] * 16 + [3] * 16)
    masks = [scored_mask(shape, bp, fold_of, f) for f in range(4)]
    assert int(np.logical_or.reduce(masks).sum()) == shape[0] * shape[1]


def test_contiguous_is_deterministic_and_seed_independent():
    tab = block_table((512, 512), 64)
    assert assign_folds(tab, 4, seed=0, mode="contiguous") == assign_folds(tab, 4, seed=99, mode="contiguous")


def test_bad_mode_is_rejected():
    tab = block_table((128, 128), 64)
    with pytest.raises(ValueError):
        assign_folds(tab, n_folds=2, mode="scatter")


def test_describe_partition_records_the_mode():
    tab = block_table((256, 256), 64)
    fold_of = assign_folds(tab, 4, seed=0, mode="contiguous")
    rep = describe_partition((256, 256), 64, 4, 0, tab, fold_of, mode="contiguous")
    assert rep["mode"] == "contiguous"
    assert "contiguous" in rep["assignment"]
