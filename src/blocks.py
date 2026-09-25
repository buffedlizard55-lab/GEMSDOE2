"""Spatial block hold-out: contiguous geographic folds over the competition grid.

WHY THIS MODULE EXISTS
----------------------
Every model-selection number this repository produced before 2026-09-18 came from a
*random-window* hold-out: `src/dataset.make_patches` scatters 256-px test windows over the
whole raster (the reference solution's design, cells 11-12).  Two consequences were measured,
not assumed:

1. **Adjacency leak.**  A held-out window is surrounded by training windows whose labels were
   used to build the false-positive weight map and to fit the encoder.  Fault traces continue
   across the window boundary, so "held-out DTI" partly measures interpolation between
   neighbours that saw the same structure.
2. **The two populations disagree on sign.**  In-domain DTI (catalogue labels) *falls* when the
   emission is widened (0.1903 -> 0.0908, `docs/METRIC_STRATEGY.md`) while the new-fault-like
   proxy population *rises* (+0.11, `data/evidence/proxy/eval_sweep.json`).  Choosing between
   them by argument is how a project talks itself into a wrong policy.

`docs/DISCOVERY_PLAN.md` §3(a) names the fix: split the grid into LARGE blocks, train with one
block fully excluded, and measure on that block.  Both metrics are then computed on geography
the model has never seen, so a policy that wins on unseen catalogue faults AND unseen
proxy-only faults is winning on the thing that actually generalises - and the sign conflict
becomes a measurement on one population instead of a choice between two.

WHAT IS GUARANTEED HERE (each property has a test in tests/test_blocks.py)
--------------------------------------------------------------------------
* **Deterministic.**  Same shape + block size + fold count + seed -> identical assignment.  A
  fold's identity is its held-out geography, so it must be reproducible from the config alone
  (Official Rules §3.5: assets must "sufficiently reproduce the winning results").
* **Contiguous.**  A fold is a union of whole rectangular blocks - never a scatter of pixels.
* **Balanced.**  Blocks are dealt greedily by descending fault-pixel mass to the currently
  lightest fold (fault px first, valid px as tie-break), so no fold is a fault-free desert and
  none carries half the labels.  A seeded shuffle breaks exact ties, which keeps the assignment
  free of a row-major geographic bias without making it non-reproducible.
* **Buffered.**  `held_out_mask(..., buffer_px=k)` additionally marks the k-pixel collar AROUND
  the held-out blocks.  Training windows that touch the collar are excluded, so no training
  window can read a label within R of the held-out truth (the metric's kernel reaches R = 3 px,
  and the FP weight map is an EDT over the training labels).  The scored truth stays the exact
  blocks; only training is pushed back.
* **Footprint-aware.**  57.92 % of the grid is NaN outside the GeoDAWN survey footprint
  (`data/evidence/rasters.json`), so blocks are balanced on *valid* pixels, not on area.

No network, no torch, no rasterio: numpy + scipy only, so the partition is testable in any
environment and identical on a runner and in the sandbox.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_BLOCK_PX",
    "DEFAULT_N_FOLDS",
    "DEFAULT_BUFFER_PX",
    "block_shape",
    "block_slices",
    "block_id_map",
    "block_table",
    "assign_folds",
    "super_grid",
    "scored_mask",
    "held_out_mask",
    "describe_partition",
]

DEFAULT_BLOCK_PX = 512      # 51.2 km squares at the official 100 m resolution
DEFAULT_N_FOLDS = 4
DEFAULT_BUFFER_PX = 3       # == the metric's R in pixels (300 m / 100 m)
KM_PER_PIXEL = 0.1          # 100 m resolution, verified in data/evidence/rasters.json


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def block_shape(shape_hw: Sequence[int], block_px: int = DEFAULT_BLOCK_PX) -> Tuple[int, int]:
    """Number of block rows/cols needed to tile (H, W); the last block in each axis is partial."""
    if block_px <= 0:
        raise ValueError(f"block_px must be positive, got {block_px}")
    H, W = int(shape_hw[0]), int(shape_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"degenerate grid {shape_hw}")
    return int(np.ceil(H / block_px)), int(np.ceil(W / block_px))


def block_slices(shape_hw: Sequence[int], block_px: int = DEFAULT_BLOCK_PX) -> List[Tuple[int, int, int, int]]:
    """Row-major list of (row0, row1, col0, col1) half-open slices, one per block."""
    H, W = int(shape_hw[0]), int(shape_hw[1])
    nby, nbx = block_shape((H, W), block_px)
    out = []
    for by in range(nby):
        for bx in range(nbx):
            out.append((by * block_px, min(H, (by + 1) * block_px),
                        bx * block_px, min(W, (bx + 1) * block_px)))
    return out


def block_id_map(shape_hw: Sequence[int], block_px: int = DEFAULT_BLOCK_PX) -> np.ndarray:
    """(H, W) int32 array whose value at each pixel is its block id (row-major index).

    Built by broadcasting per-axis index vectors, so a 3292x3730 grid costs two small arrays
    plus one (H, W) int32 (49 MB) rather than a python loop over 12 M pixels.
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    nby, nbx = block_shape((H, W), block_px)
    rows = (np.arange(H, dtype=np.int32) // block_px)[:, None]          # (H,1)
    cols = (np.arange(W, dtype=np.int32) // block_px)[None, :]          # (1,W)
    return (rows * nbx + cols).astype(np.int32)


def block_table(shape_hw: Sequence[int], block_px: int = DEFAULT_BLOCK_PX,
                valid: Optional[np.ndarray] = None,
                labels: Optional[np.ndarray] = None) -> List[dict]:
    """Per-block statistics used to balance the folds and to report what each fold contains.

    `valid`  - boolean/finite mask of the data footprint (NaN outside the survey area)
    `labels` - binary fault labels (the training catalogue)

    Both are optional so the partition can be built from geometry alone (e.g. in unit tests).
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    if valid is not None and valid.shape != (H, W):
        raise ValueError(f"valid mask {valid.shape} != grid {(H, W)}")
    if labels is not None and labels.shape != (H, W):
        raise ValueError(f"labels {labels.shape} != grid {(H, W)}")
    nby, nbx = block_shape((H, W), block_px)
    out: List[dict] = []
    for bid, (r0, r1, c0, c1) in enumerate(block_slices((H, W), block_px)):
        area = (r1 - r0) * (c1 - c0)
        row: dict = dict(block_id=bid, block_row=bid // nbx, block_col=bid % nbx,
                         row0=int(r0), row1=int(r1), col0=int(c0), col1=int(c1),
                         area_px=int(area))
        if valid is not None:
            v = valid[r0:r1, c0:c1]
            row["valid_px"] = int(np.count_nonzero(v))
            row["valid_frac"] = float(row["valid_px"] / area) if area else 0.0
            if labels is not None:
                row["fault_px"] = int(np.count_nonzero(labels[r0:r1, c0:c1] & np.asarray(v, bool)))
            else:
                row["fault_px"] = 0
        elif labels is not None:
            row["valid_px"] = int(area)
            row["valid_frac"] = 1.0
            row["fault_px"] = int(np.count_nonzero(labels[r0:r1, c0:c1]))
        else:
            row["valid_px"] = int(area)
            row["valid_frac"] = 1.0
            row["fault_px"] = 0
        row["fault_km"] = round(row["fault_px"] * KM_PER_PIXEL, 3)
        out.append(row)
    assert len(out) == nby * nbx, "block table size disagrees with block_shape"
    return out


# --------------------------------------------------------------------------------------
# fold assignment
# --------------------------------------------------------------------------------------
def super_grid(nby: int, nbx: int, n_folds: int) -> Tuple[str, object, int]:
    """How to cut an (nby, nbx) block grid into `n_folds` CONTIGUOUS super-regions.

    Returns ("grid", sy, sx) when n_folds factorises into two integers > 1 - the pair whose
    super-cell aspect is closest to the block grid's aspect is chosen, so a 8x7 block grid cut
    into 4 becomes 2x2 rather than 1x4 strips.  Otherwise ("bands", axis, n_folds): contiguous
    bands along the longer axis, which works for prime fold counts.
    """
    pairs = [(sy, n_folds // sy) for sy in range(2, n_folds)
             if n_folds % sy == 0 and n_folds // sy >= 2]
    if pairs:
        sy, sx = min(pairs, key=lambda p: abs(math.log((nby / p[0]) / (nbx / p[1]))))
        return ("grid", int(sy), int(sx))
    return ("bands", "row" if nby >= nbx else "col", int(n_folds))


def _assign_contiguous(table: Sequence[dict], n_folds: int) -> Dict[int, int]:
    """Contiguous super-region assignment (see `assign_folds(mode="contiguous")`)."""
    nby = max(int(t["block_row"]) for t in table) + 1
    nbx = max(int(t["block_col"]) for t in table) + 1
    kind, a, b = super_grid(nby, nbx, n_folds)
    fold_of: Dict[int, int] = {}
    if kind == "grid":
        row_of = {int(r): i for i, g in enumerate(np.array_split(np.arange(nby), int(a))) for r in g}
        col_of = {int(c): i for i, g in enumerate(np.array_split(np.arange(nbx), int(b))) for c in g}
        for t in table:
            fold_of[int(t["block_id"])] = row_of[int(t["block_row"])] * int(b) + col_of[int(t["block_col"])]
    else:
        n = nby if a == "row" else nbx
        key = "block_row" if a == "row" else "block_col"
        idx_of = {int(v): i for i, g in enumerate(np.array_split(np.arange(n), int(b))) for v in g}
        for t in table:
            fold_of[int(t["block_id"])] = idx_of[int(t[key])]
    if max(fold_of.values()) >= n_folds:
        raise AssertionError(f"contiguous assignment produced {max(fold_of.values()) + 1} folds, expected {n_folds}")
    for t in table:                                        # footprint-empty blocks are unscoreable
        if int(t.get("valid_px", 0)) == 0:
            fold_of[int(t["block_id"])] = -1
    return fold_of


def assign_folds(table: Sequence[dict], n_folds: int = DEFAULT_N_FOLDS,
                 seed: int = 0, mode: str = "balanced") -> Dict[int, int]:
    """Deterministic block -> fold assignment.  Two modes, because they answer two questions.

    ``mode="balanced"`` (default) - blocks are dealt to folds so that each fold carries a
    comparable share of the fault mass and of the survey footprint.  The held-out blocks are
    scattered over the region, so this measures generalisation to *unseen 51 km blocks* with
    the lowest variance.  It is the mode to select models with.

    ``mode="contiguous"`` - the block grid is cut into ``n_folds`` contiguous super-regions
    (rows x cols as close to the block grid's aspect as the fold count allows) and each fold
    holds out one whole super-region.  This is the stricter test: the model must extrapolate to
    geography it has never seen adjacent to its training data, which is qualitatively closer to
    "find faults nobody has mapped" than interpolation between scattered blocks.  Folds cannot
    be mass-balanced by construction; `describe_partition` reports the resulting spread so the
    imbalance is a number, not a surprise.

    Empty blocks (no valid pixels at all) get fold -1 in both modes: they are outside the survey
    footprint and can never be scored, so handing them to a fold would only dilute its
    statistics.

    Balanced algorithm: seeded shuffle (tie-break only) -> sort by descending fault mass, then
    descending valid px -> deal each block to the fold with the smallest NORMALISED load

        load(f) = fault_load(f)/fault_target + valid_load(f)/valid_target

    where each target is that resource's total share (total/n_folds).  Normalising both
    resources matters and was measured, not assumed: a greedy keyed on absolute fault mass
    first (fault_load, then valid_load) made the fold that missed the few mass-rich blocks a
    SINK for every remaining zero-fault block - on a synthetic 512x512/64-px grid with two
    fault stripes the valid-px split came out [8192, 8192, 6144, 215552], i.e. one fold held
    90 % of the footprint.  tests/test_blocks.py::test_valid_pixel_balance_across_folds pins
    the fix.  This is the standard greedy multiway-number-partition (LPT) heuristic on two
    normalised resources; O(B log B), no solver.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}")
    if mode not in ("balanced", "contiguous"):
        raise ValueError(f"mode must be 'balanced' or 'contiguous', got {mode!r}")
    if mode == "contiguous":
        return _assign_contiguous(table, n_folds)
    rng = np.random.default_rng(seed)
    order = np.arange(len(table))
    rng.shuffle(order)                                   # deterministic tie-break only
    scored = [i for i in order if int(table[i].get("valid_px", 0)) > 0]
    scored.sort(key=lambda i: (-int(table[i].get("fault_px", 0)),
                               -int(table[i].get("valid_px", 0)),
                               int(table[i]["block_id"])))
    fault_total = sum(int(table[i].get("fault_px", 0)) for i in scored)
    valid_total = sum(int(table[i].get("valid_px", 0)) for i in scored)
    fault_target = max(1.0, fault_total / n_folds)
    valid_target = max(1.0, valid_total / n_folds)
    fault_load = [0] * n_folds
    valid_load = [0] * n_folds
    fold_of: Dict[int, int] = {}
    for i in scored:
        f = min(range(n_folds),
                key=lambda k: (fault_load[k] / fault_target + valid_load[k] / valid_target, k))
        fold_of[int(table[i]["block_id"])] = f
        fault_load[f] += int(table[i].get("fault_px", 0))
        valid_load[f] += int(table[i].get("valid_px", 0))
    for i in order:
        fold_of.setdefault(int(table[i]["block_id"]), -1)   # footprint-empty blocks
    return fold_of


def held_out_mask(shape_hw: Sequence[int], block_px: int, fold_of: Dict[int, int],
                  fold: int, buffer_px: int = DEFAULT_BUFFER_PX) -> np.ndarray:
    """Boolean (H, W): True on the collar AROUND fold `fold`'s blocks (training exclusion).

    Two masks matter and they are deliberately different:
      * `scored_mask`  - the exact blocks of this fold (what the metric is computed on)
      * `held_out_mask`- scored_mask dilated by `buffer_px` (where training may not look)

    buffer_px = 0 returns the scored mask itself.
    """
    ids = block_id_map(shape_hw, block_px)
    held_ids = np.array(sorted(b for b, f in fold_of.items() if f == fold), dtype=np.int64)
    if held_ids.size == 0:
        raise ValueError(f"fold {fold} holds out no blocks - nothing to measure")
    scored = np.isin(ids, held_ids)
    if buffer_px <= 0:
        return scored
    from scipy.ndimage import binary_dilation
    b = int(buffer_px)
    struct = np.ones((2 * b + 1, 2 * b + 1), bool)
    return binary_dilation(scored, structure=struct)


def scored_mask(shape_hw: Sequence[int], block_px: int, fold_of: Dict[int, int],
                fold: int) -> np.ndarray:
    """Boolean (H, W): exactly the blocks held out for `fold` (no collar)."""
    return held_out_mask(shape_hw, block_px, fold_of, fold, buffer_px=0)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def describe_partition(shape_hw: Sequence[int], block_px: int, n_folds: int, seed: int,
                       table: Sequence[dict], fold_of: Dict[int, int],
                       buffer_px: int = DEFAULT_BUFFER_PX,
                       labels: Optional[np.ndarray] = None,
                       valid: Optional[np.ndarray] = None,
                       mode: str = "balanced") -> dict:
    """Machine-readable description of a partition - committed as evidence, never retyped.

    Every number here is computed from `table`/`fold_of`, so a report cannot disagree with the
    partition it describes.  `per_fold` includes the collar-excluded pixel count, which is the
    quantity that proves training was pushed back from the held-out geography.
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    per_fold = []
    for f in range(n_folds):
        rows = [t for t in table if fold_of.get(t["block_id"]) == f]
        fault_px = sum(int(r["fault_px"]) for r in rows)
        valid_px = sum(int(r["valid_px"]) for r in rows)
        entry = dict(
            fold=f,
            blocks=len(rows),
            block_ids=[int(r["block_id"]) for r in rows],
            valid_px=int(valid_px),
            fault_px=int(fault_px),
            fault_km=round(fault_px * KM_PER_PIXEL, 3),
            valid_frac_of_grid=round(valid_px / float(H * W), 6),
            fault_frac_of_scored=round(fault_px / max(1, sum(int(t["fault_px"]) for t in table)), 6),
        )
        if labels is not None or valid is not None:
            try:
                collar = held_out_mask((H, W), block_px, fold_of, f, buffer_px=buffer_px)
                sc = scored_mask((H, W), block_px, fold_of, f)
                entry["collar_excluded_px"] = int(np.count_nonzero(collar) - np.count_nonzero(sc))
            except ValueError:                                  # a fold with no blocks
                entry["collar_excluded_px"] = 0
        per_fold.append(entry)
    scored_fault = [e["fault_px"] for e in per_fold]
    return dict(
        grid=dict(height=H, width=W),
        block_px=int(block_px),
        block_shape=list(block_shape((H, W), block_px)),
        n_blocks=int(len(table)),
        n_folds=int(n_folds),
        seed=int(seed),
        buffer_px=int(buffer_px),
        resolution_m=100.0,
        block_km=round(block_px * KM_PER_PIXEL, 3),
        mode=mode,
        assignment=(("greedy LPT: blocks dealt by descending fault mass to the fold with the "
                     "smallest normalised load (fault/target + valid/target)") if mode == "balanced"
                    else "contiguous super-regions of the block grid (see super_grid)"),
        per_fold=per_fold,
        balance=dict(
            fault_px_min=int(min(scored_fault)) if scored_fault else 0,
            fault_px_max=int(max(scored_fault)) if scored_fault else 0,
            fault_px_spread=round((max(scored_fault) - min(scored_fault)) / max(1, float(np.mean(scored_fault))), 6)
            if scored_fault else 0.0,
        ),
        empty_blocks_excluded=int(sum(1 for f in fold_of.values() if f == -1)),
    )
