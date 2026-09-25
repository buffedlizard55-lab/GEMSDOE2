"""Dataset utilities for the GEMS Prize.

Responsibilities
----------------
* load the competition GeoTIFF stack(s) (all documented name variants are accepted; see
  data/README.md - the problem page, the reference solution and the Dropbox mirrors each
  use a different file name)
* normalisation with *persisted* statistics (train and inference must use the same numbers;
  Official Rules 3.5 requires assets that "sufficiently reproduce the winning results")
* patching that follows the reference solution's anti-leakage design: test windows are cut
  out of the global raster FIRST, zeroed globally, and only then are (overlapping) training
  windows sampled - so no training window can read a test label
* augmentation (flips / 90-degree rotations / noise) applied identically to x, y and the
  false-positive weight map
* per-patch precomputation of the metric's FP weight map (1 - max_g k(d(x,g))), cropped from
  the GLOBAL distance transform so predictions at a patch border are not over-penalised

Verified sources
----------------
- reference solution (patchify/unpatchify + global zeroing idea):
  https://github.com/drivendataorg/gems-prize-reference-solution  (cells 9-12)
- feature/label description + CRS/resolution:
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#datasets
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio

# torch is a TRAINING dependency, not a data dependency.  The documented fast path
# (EXECUTIVE_SUMMARY.md §0: assemble_data_bridge -> prepare_data -> validate_submission)
# must run on a machine that only has numpy/scipy/rasterio/scikit-image installed - a 2.5 GB
# torch install is not a prerequisite for placing or validating a GeoTIFF.  Measured defect
# 2026-09-18: with torch absent, `python scripts/prepare_data.py` died on this import, so the
# submission guide's own quickstart failed on a clean environment.  The stub keeps module
# import cheap and turns the failure into a precise, actionable error at the point torch is
# actually needed (constructing a FaultDataset).
try:
    from torch.utils.data import Dataset as _TorchDataset
except ModuleNotFoundError:                                   # pragma: no cover - env dependent
    class _TorchDataset:
        """Placeholder base so this module imports without torch."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "FaultDataset requires torch. Install the training dependencies "
                "(`pip install -r requirements.txt`) - or, if you only need to place and "
                "validate competition data, use scripts/assemble_data_bridge.py, "
                "scripts/prepare_data.py and scripts/validate_submission.py, which do not."
            )

FEATURE_NAME_CANDIDATES = (
    "training_features.tif",              # problem page
    "numeric_features.tif",               # reference solution
    "gems-geodawn-numerical-features.tif",  # Dropbox mirror on the data tab
    "features.tif",
)
LABEL_NAME_CANDIDATES = ("labels.tif", "existing_faults.tif", "faults.tif")   # page / mirror / -
SAMPLE_NAME_CANDIDATES = ("sample_submission.tif", "example_submission.tif")


def resolve_path(explicit: Optional[str], candidates: Sequence[str], subdirs=("", "reconstructed")) -> str:
    """Return first existing path among `explicit` then `<dir>/<candidate>` for dir in subdirs."""
    if explicit:
        if os.path.exists(explicit):
            return explicit
        base = Path(explicit).name
        for d in subdirs:
            p = Path("data") / d / base if d else Path("data") / base
            if p.exists():
                return str(p)
    for d in subdirs:
        for c in candidates:
            p = (Path("data") / d / c) if d else (Path("data") / c)
            if p.exists():
                return str(p)
    # reconstructed variants
    for d in subdirs:
        for c in candidates:
            p = (Path("data") / d / f"recon_{c}") if d else None
            if p and p.exists():
                return str(p)
    raise FileNotFoundError(
        f"none of {list(candidates)} found under data/ (checked subdirs {list(subdirs)}). "
        "Run: bash scripts/download_competition_data.sh"
    )


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_stack(path: str) -> Tuple[np.ndarray, dict, List[dict]]:
    """-> (H,W,C) float32 array with nodata->NaN, meta dict, per-band tag dicts."""
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        meta["crs"] = src.crs
        meta["transform"] = src.transform
        data = src.read().astype(np.float32)          # (C,H,W)
        nodata = src.nodata
        tags = []
        for i in range(1, src.count + 1):
            t = dict(src.tags(i) or {})
            tags.append(t)
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    data[data < -1e30] = np.nan                        # reference solution convention
    return np.moveaxis(data, 0, -1), meta, tags         # (H,W,C)


def load_labels(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        meta["crs"] = src.crs
        meta["transform"] = src.transform
        y = src.read(1).astype(np.float32)
        if src.nodata is not None:
            y[y == src.nodata] = 0
    y[~np.isfinite(y)] = 0
    y = (y > 0.5).astype(np.float32)          # reference: `y_orig[y_orig < 1] = 0`
    return y, meta


def load_pseudo_mask(path: str, code: int = 2) -> np.ndarray:
    """Coded fault raster -> binary mask of the pixels carrying `code`.

    The standard input is `data/evidence/proxy/proxy_catalogue.tif` (0 = no trace,
    1 = trace within R px of a training label, 2 = trace the labels do NOT contain -
    the 61,664 px SGMC proxy population).  Pseudo-labels are external-catalogue
    information the competition explicitly allows (problem page #external-datasets,
    rules PDF 3.2); the leakage safety is enforced in `make_patches`, not here: only
    TRAINING windows receive pseudo pixels as label mass, the block partition and the
    window selection keep using the original labels alone, so a pseudo run and its
    baseline run train on exactly the same windows and the measured difference is the
    pseudo signal, nothing else.
    """
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path}: expected a single-band coded raster, got {src.count} bands")
        a = src.read(1)
    mask = (a == int(code))
    if not mask.any():
        raise ValueError(f"{path}: no pixels with code {code} - nothing to pseudo-label")
    return mask


def band_names(tags: List[dict], n: int) -> List[str]:
    out = []
    for i in range(n):
        t = tags[i] if i < len(tags) else {}
        out.append(t.get("description") or t.get("name") or t.get("LONG_NAME") or f"band_{i}")
    return out


# --------------------------------------------------------------------------------------
# normalisation (stats persisted so train == inference == reproduction)
# --------------------------------------------------------------------------------------
def fit_norm_stats(X: np.ndarray, clip_percentile=(1.0, 99.0)) -> dict:
    """Per-channel (lo, hi, mean, std) on finite values, after percentile clipping."""
    stats = {"clip_percentile": list(clip_percentile), "channels": []}
    for c in range(X.shape[-1]):
        ch = X[..., c]
        v = ch[np.isfinite(ch)]
        if v.size == 0:
            stats["channels"].append(dict(lo=0.0, hi=1.0, mean=0.0, std=1.0, n=0))
            continue
        lo, hi = (float(np.percentile(v, clip_percentile[0])), float(np.percentile(v, clip_percentile[1])))
        if not hi > lo:
            lo, hi = float(v.min()), float(v.max())
        sel = v[(v >= lo) & (v <= hi)]
        mean = float(sel.mean()) if sel.size else 0.0
        std = float(sel.std()) if sel.size and sel.std() > 1e-8 else 1.0
        stats["channels"].append(dict(lo=lo, hi=hi, mean=mean, std=std, n=int(v.size)))
    return stats


def apply_norm_stats(X: np.ndarray, stats: dict, mode: str = "clip_zscore") -> np.ndarray:
    """(H,W,C) -> (H,W,C) float32 in ~[0,1]; NaNs become 0 (model sees 'no data')."""
    out = np.zeros(X.shape, dtype=np.float32)
    for c, ch in enumerate(stats["channels"]):
        v = X[..., c].astype(np.float32)
        lo, hi = ch["lo"], ch["hi"]
        if hi <= lo:
            continue
        v = np.clip(v, lo, hi)
        if mode == "minmax":
            out[..., c] = (v - lo) / (hi - lo)
        else:  # clip_zscore -> 0..1, centred, robust to outliers
            out[..., c] = np.clip(0.5 + 0.25 * (v - ch["mean"]) / ch["std"], 0.0, 1.0)
        bad = ~np.isfinite(X[..., c])
        if bad.any():
            out[..., c][bad] = 0.0
    return out


def save_norm_stats(path, stats):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)


def load_norm_stats(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------------------
def _windows(shape_hw, patch: int, step: int):
    H, W = shape_hw
    ys = list(range(0, max(1, H - patch + 1), step))
    xs = list(range(0, max(1, W - patch + 1), step))
    if ys and ys[-1] + patch < H:
        ys.append(H - patch)
    if xs and xs[-1] + patch < W:
        xs.append(W - patch)
    return [(y, x) for y in ys for x in xs]


def make_patches(
    X: np.ndarray,
    y: np.ndarray,
    patch_size: int = 128,
    train_step: int = 64,
    test_proportion: float = 0.3,
    seed: int = 0,
    neg_fraction: float = 0.35,
    R_pixels: int = 3,
    min_px_per_patch: int = 3,
    holdout: str = "random",
    block_px: int = 512,
    block_folds: int = 4,
    block_fold: int = 0,
    block_buffer_px: Optional[int] = None,
    block_mode: str = "balanced",
    block_seed: int = 0,
    pseudo: Optional[np.ndarray] = None,
    pseudo_weight: float = 1.0,
):
    """Leakage-free split + overlapping training windows.

    Returns dict with:
      X_tr, y_tr, fpw_tr (lists of arrays), X_te, y_te, test_origin (row, col of each test
      window in the padded global grid), n_gt.

    Order of operations (matches reference notebook cells 11-12 semantics):
      1. non-overlapping test grid -> choose test windows -> ZERO them in the global arrays
      2. re-patchify the *zeroed* global arrays with `train_step` overlap
      3. keep windows that contain >= min_px_per_patch fault pixels, plus `neg_fraction`
         of empty windows (hard negatives make the model calibrate, unlike the reference,
         which trains only on fault-containing windows)

    TWO HOLD-OUT DESIGNS (`holdout=`)
    ---------------------------------
    ``"random"`` (default, unchanged behaviour) - the reference solution's Monte-Carlo split:
      `test_proportion` of the valid non-overlapping windows are drawn with the fold's RNG.
      Reproducible, but the held-out windows are *scattered*, so every one of them is
      surrounded by training windows that saw the same fault traces (adjacency leak).

    ``"spatial_blocks"`` - `docs/DISCOVERY_PLAN.md` §3(a): the grid is cut into `block_px`
      blocks (512 px = 51.2 km at the official 100 m resolution) and whole blocks are held
      out (`src/blocks.py`).  Differences from the random path, each deliberate:

      * test windows are the grid windows **fully inside** the held-out blocks - never a
        partial window, so no scored pixel is also a training pixel;
      * training windows are dropped if they touch the held-out blocks **or their
        `block_buffer_px` collar** (default = `R_pixels` = 3, the metric's own kernel reach),
        which is stricter than the random path's 25 % overlap rule.  Without the collar a
        training window one pixel from the boundary could read a label whose fault continues
        into the held-out block;
      * the block partition is seeded by `block_seed` (default 0), **not** by the fold's
        `seed`: fold identity is its geography, and every fold of one experiment must agree on
        which blocks belong to which fold (Official Rules §3.5 reproducibility).

    `block_mode` selects "balanced" (scattered blocks, mass-balanced - lowest-variance model
    selection) or "contiguous" (one whole super-region per fold - geographic extrapolation).

    ``pseudo`` (2026-09-19) - an optional binary mask of pseudo-label pixels (external
    catalogue, e.g. the SGMC proxy population, `load_pseudo_mask`).  With
    ``pseudo_weight > 0`` those pixels become positives (value ``pseudo_weight``) in the
    training labels.  Deliberately narrow, each for a measured reason:

      * the block partition (`block_table` -> `assign_folds`) and the pos/neg WINDOW
        SELECTION keep using the original labels only: the pseudo run and its baseline
        run train on exactly the same windows, so the measured DTI difference is the
        pseudo signal and not a different training set;
      * the global FP-weight map is computed on original labels UNION pseudo, both
        already zeroed in the test region (the same leakage fix as the labels): a
        prediction on a pseudo pixel is a true positive the loss should credit, not a
        full-weight false positive - but a pseudo pixel inside the held-out region must
        not reduce the FP weight of training geography;
      * test windows are untouched: the held-out measurement (and with it the leakage
        reading) is against the labels, exactly as in the baseline.
    """
    if holdout not in ("random", "spatial_blocks"):
        raise ValueError(f"holdout must be 'random' or 'spatial_blocks', got {holdout!r}")
    rng = np.random.default_rng(seed)
    H, W, C = X.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    Xp = np.pad(X, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    yp = np.pad(y, ((0, pad_h), (0, pad_w)), constant_values=0)
    Hp, Wp = yp.shape

    # ---- 1. test windows on a non-overlapping grid --------------------------------
    grid = _windows((Hp, Wp), patch_size, patch_size)
    holdout_info = None
    exclude_mask = None
    if holdout == "spatial_blocks":
        from .blocks import (DEFAULT_BUFFER_PX, assign_folds, block_table, describe_partition,
                             held_out_mask, scored_mask)
        buf = int(R_pixels if block_buffer_px is None else block_buffer_px)
        valid_orig = np.isfinite(X).any(axis=-1)
        labels_orig = y > 0.5
        tab = block_table((H, W), block_px, valid=valid_orig, labels=labels_orig)
        fold_of = assign_folds(tab, block_folds, seed=block_seed, mode=block_mode)
        scored = scored_mask((H, W), block_px, fold_of, block_fold)
        excluded = held_out_mask((H, W), block_px, fold_of, block_fold, buffer_px=buf)
        pad_bool = ((0, pad_h), (0, pad_w))
        scored_p = np.pad(scored, pad_bool, constant_values=False)
        exclude_mask = np.pad(excluded, pad_bool, constant_values=False)
        test_windows = [(i, j) for (i, j) in grid
                        if scored_p[i:i + patch_size, j:j + patch_size].all()
                        and np.isfinite(Xp[i:i + patch_size, j:j + patch_size]).any()]
        test_windows = sorted(test_windows)
        test_mask = scored_p
        fold_rows = [t for t in tab if fold_of.get(t["block_id"]) == block_fold]
        holdout_info = dict(
            mode="spatial_blocks", block_px=int(block_px), block_folds=int(block_folds),
            block_fold=int(block_fold), block_seed=int(block_seed), block_mode=block_mode,
            buffer_px=buf,
            partition=describe_partition((H, W), block_px, block_folds, block_seed, tab,
                                         fold_of, buffer_px=buf, mode=block_mode),
            fold=dict(blocks=len(fold_rows),
                      block_ids=[int(t["block_id"]) for t in fold_rows],
                      fault_px=int(sum(t["fault_px"] for t in fold_rows)),
                      valid_px=int(sum(t["valid_px"] for t in fold_rows)),
                      scored_px=int(scored.sum()),
                      excluded_px=int(excluded.sum()),
                      collar_px=int(excluded.sum() - scored.sum())),
        )
    else:
        valid = [(i, j) for (i, j) in grid if np.isfinite(Xp[i:i + patch_size, j:j + patch_size]).any()]
        n_test = int(round(test_proportion * len(valid)))
        test_windows = sorted(rng.choice(len(valid), size=n_test, replace=False).tolist())
        test_windows = [valid[i] for i in test_windows]
        test_mask = np.zeros((Hp, Wp), bool)
        for (i, j) in test_windows:
            test_mask[i:i + patch_size, j:j + patch_size] = True

    X_test = np.stack([Xp[i:i + patch_size, j:j + patch_size] for (i, j) in test_windows]) if test_windows \
        else np.zeros((0, patch_size, patch_size, C), np.float32)
    y_test = np.stack([yp[i:i + patch_size, j:j + patch_size] for (i, j) in test_windows]) if test_windows \
        else np.zeros((0, patch_size, patch_size), np.float32)

    # ---- 2. zero the test region globally, THEN extract overlapping train windows --
    # In-place on the padded arrays: the held-out copies were already extracted above, and
    # a .copy() here used to be a THIRD full-stack allocation (X + Xp + Xtr_src ~ 2.9 GB at
    # the 19-band GeoDAWN grid), which OOM-killed the 3.9 GB dev sandbox on 2026-09-17
    # (measured anon-rss 3.8 GB). Semantics unchanged: Xtr_src is the padded stack with the
    # test region zeroed, exactly what the .copy() produced.
    Xtr_src, ytr_src = Xp, yp
    Xtr_src[test_mask] = 0.0
    ytr_src[test_mask] = 0.0

    # ---- pseudo-labels (external catalogue, e.g. the SGMC proxy population) ----------
    # Applied to training windows ONLY, and never to the partition or the window
    # selection (see the docstring).  `pseudo_src` is the padded, test-region-zeroed
    # mask; None means the experiment is off and the function is byte-identical to the
    # baseline path (pinned by tests/test_pseudo_labels.py).
    pseudo_src = None
    if pseudo is not None:
        if pseudo.shape != (H, W):
            raise ValueError(f"pseudo mask {pseudo.shape} != grid {(H, W)}")
        if pseudo_weight > 0:
            if pseudo_weight < 0.5:
                raise ValueError(
                    f"pseudo_weight {pseudo_weight} < 0.5 would be silently dropped by the "
                    f"label binarisation in FaultDataset - use >= 0.5 (1.0 = hard pseudo-label)")
            if not (pseudo > 0.5).any():
                pseudo_src = None     # mask provided but empty: nothing to do, pure baseline
            else:
                pseudo_p = np.pad(pseudo > 0.5, ((0, pad_h), (0, pad_w)), constant_values=False)
                pseudo_src = pseudo_p & ~test_mask

    # global FP weight map = 1 - max_g k(d(x,g)), computed on the TRAINING labels ONLY
    # (UNION the training-region pseudo pixels: a prediction on them is a fault the loss
    # should credit, not a full-weight false positive).
    #
    # LEAKAGE FIX (2026-09-15): this EDT used to run on `yp`, i.e. *before* the held-out
    # windows were zeroed.  fpw is handed to the loss as "how much a prediction here counts
    # as a false positive", so a value < 1 is a direct statement that a fault lies within
    # R px.  Computing it on unzeroed labels therefore wrote the held-out fault positions
    # into the training signal: a train window may legitimately overlap the test grid by up
    # to 25% (see the filter below), and inside that sliver the loss was told not to
    # penalise predictions exactly where the hidden faults are.  Measured on a synthetic
    # 256x256 case with a fault placed only inside the test window: 196 test-region pixels
    # carried fpw < 1 before the fix, 0 after.  Held-out DTI was consequently optimistic.
    from scipy.ndimage import distance_transform_edt
    fp_src = (ytr_src > 0.5)
    if pseudo_src is not None:
        fp_src = fp_src | pseudo_src
    if fp_src.any():
        d2gt = distance_transform_edt(~fp_src)
        fpw_global = 1.0 - np.maximum(1.0 - d2gt / float(R_pixels), 0.0)
    else:
        fpw_global = np.ones((Hp, Wp), np.float32)
    # inside the held-out region there is no training label to speak of; treat any
    # prediction there as a full-weight false positive rather than inferring from
    # neighbouring train faults that bled across the boundary.
    fpw_global[test_mask] = 1.0

    cand = _windows((Hp, Wp), patch_size, train_step)
    pos, neg = [], []
    for (i, j) in cand:
        if exclude_mask is not None:
            # SPATIAL BLOCKS: a training window may not touch the held-out blocks OR their
            # buffer collar at all.  The collar exists because the metric's kernel reaches
            # R px: a window one pixel from the boundary would otherwise be trained on labels
            # whose fault trace continues into the scored geography.
            if exclude_mask[i:i + patch_size, j:j + patch_size].any():
                continue
        elif test_mask[i:i + patch_size, j:j + patch_size].mean() > 0.25:
            continue                       # mostly-test window: skip outright (random hold-out)
        n_fault = int((ytr_src[i:i + patch_size, j:j + patch_size] > 0.5).sum())
        (pos if n_fault >= min_px_per_patch else neg).append((i, j))
    keep = list(pos)
    if neg and neg_fraction > 0:
        n_keep = int(round(neg_fraction * len(pos) / max(1e-6, 1 - neg_fraction)))
        n_keep = min(n_keep, len(neg))
        keep += [neg[a] for a in rng.choice(len(neg), size=n_keep, replace=False)]
    keep = sorted(keep)

    # PSEUDO LABELS APPLIED HERE - after selection, so the training set is identical to the
    # baseline run's: the pos/neg split above used the original labels alone, and only the
    # label VALUES inside the selected windows change.
    pseudo_train_px, pseudo_windows = 0, 0
    if pseudo_src is not None:
        added = pseudo_src & (ytr_src <= 0.5)
        pseudo_train_px = int(added.sum())
        if pseudo_train_px:
            ytr_src = ytr_src.copy()
            ytr_src[added] = pseudo_weight
        pseudo_windows = int(sum(1 for (i, j) in keep
                                 if (pseudo_src[i:i + patch_size, j:j + patch_size]).any()))

    X_train = np.stack([Xtr_src[i:i + patch_size, j:j + patch_size] for (i, j) in keep]) if keep \
        else np.zeros((0, patch_size, patch_size, C), np.float32)
    y_train = np.stack([ytr_src[i:i + patch_size, j:j + patch_size] for (i, j) in keep]) if keep \
        else np.zeros((0, patch_size, patch_size), np.float32)
    fpw_train = np.stack([fpw_global[i:i + patch_size, j:j + patch_size] for (i, j) in keep]) if keep \
        else np.zeros((0, patch_size, patch_size), np.float32)

    summary = dict(
        n_train=len(keep), n_test=len(test_windows), n_pos=len(pos), n_neg_kept=len(keep) - len(pos),
        patch=patch_size, train_step=train_step, seed=int(seed), H=Hp, W=Wp, C=C,
        test_windows=[(int(i), int(j)) for (i, j) in test_windows],
        train_windows=[(int(i), int(j)) for (i, j) in keep],
        # Which geography was held out is part of a fold's IDENTITY, so it is recorded rather
        # than implied (Official Rules 3.5: assets must reproduce the result).  For the random
        # hold-out the window list already pins it; for spatial blocks the partition is derived
        # from (block_px, block_folds, block_fold, block_seed, block_mode) and echoed here so a
        # reader cannot confuse a block fold with a random split of the same seed.
        holdout=holdout_info if holdout_info is not None else dict(
            mode="random", test_proportion=float(test_proportion)),
        pseudo=(dict(enabled=False) if pseudo_src is None else
                dict(enabled=True, weight=float(pseudo_weight),
                     mask_px=int((pseudo > 0.5).sum()),
                     train_px_added=pseudo_train_px,
                     train_windows_affected=pseudo_windows,
                     note=("pseudo pixels become label mass in training windows only; partition "
                           "and window selection used the original labels alone, so this run is "
                           "compared against the no-pseudo baseline on identical windows"))),
    )
    return dict(X_train=X_train, y_train=y_train, fpw_train=fpw_train,
                X_test=X_test, y_test=y_test, summary=summary,
                # unpadded training-label grid (H, W): the union_population selection signal in
                # src/train.py builds the combined truth from exactly these pixels
                labels=y)


# --------------------------------------------------------------------------------------
# torch dataset
# --------------------------------------------------------------------------------------
class FaultDataset(_TorchDataset):
    """X: (N,H,W,C) normalised, y: (N,H,W), fpw: (N,H,W).  Augment = flips/rot90/noise.

    Augmentation is geometry-preserving-with-label (90-degree rotations and flips keep
    faults valid; the reference solution uses RandomResizedCrop + RandomRotation(30), which
    we additionally support via `rand_crop_scale`).
    """

    def __init__(self, X, y, fpw=None, train=False, augment=True, noise_std=0.01,
                 rand_crop_scale=(0.75, 1.0), seed=None):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.fpw = np.ones_like(self.y, np.float32) if fpw is None else np.ascontiguousarray(fpw, np.float32)
        self.train = train
        self.augment = augment and train
        self.noise_std = noise_std
        self.rand_crop_scale = rand_crop_scale
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, y, w = self.X[idx], self.y[idx], self.fpw[idx]
        if self.augment:
            r = self._rng
            # random-resized-crop (zoom into a sub-window, then re-pad to patch size):
            # teaches scale invariance; faults are 1-px wide at 100 m so we keep s >= 0.75
            s_lo, s_hi = self.rand_crop_scale
            if r.random() < 0.5 and s_hi < 1.0 + 1e-6 and s_lo < 1.0:
                s = float(r.uniform(s_lo, s_hi))
                p = x.shape[0]
                cp = max(16, int(round(p * s)))
                if cp < p:
                    i = int(r.integers(0, p - cp + 1))
                    j = int(r.integers(0, p - cp + 1))
                    x, y, w = x[i:i + cp, j:j + cp], y[i:i + cp, j:j + cp], w[i:i + cp, j:j + cp]
                    k = int(round((p - cp) / 2))
                    x = np.pad(x, ((k, p - cp - k), (k, p - cp - k), (0, 0)))
                    y = np.pad(y, ((k, p - cp - k), (k, p - cp - k)))
                    w = np.pad(w, ((k, p - cp - k), (k, p - cp - k)), constant_values=1.0)
            if r.random() < 0.5:
                x, y, w = x[:, ::-1], y[:, ::-1], w[:, ::-1]
            if r.random() < 0.5:
                x, y, w = x[::-1], y[::-1], w[::-1]
            k = int(r.integers(0, 4))
            if k:
                x, y, w = (np.rot90(a, k, axes=(0, 1)) for a in (x, y, w))
            if self.noise_std > 0:
                x = x + self._rng.normal(0, self.noise_std, size=x.shape).astype(np.float32)
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)), dtype=np.float32)
        # flips/rot90 leave negative strides, which torch.from_numpy rejects
        y = np.ascontiguousarray((y > 0.5).astype(np.float32))
        w = np.ascontiguousarray(w.astype(np.float32))
        return x, y, w


# --------------------------------------------------------------------------------------
# high-level loaders used by train.py / inference.py
# --------------------------------------------------------------------------------------
def _maybe_add_external_dem(X, meta, *, use_external_dem=False, external_dem_path=None):
    """Append explicitly supplied DEM derivatives, or leave the official stack unchanged.

    The competition's DEM URLs are optional external data, not a band that can be
    silently inferred from the 19-band GeoTIFF.  Requiring a local mosaic path when the
    flag is enabled prevents train/inference channel drift and makes provenance clear.
    """
    if not use_external_dem:
        return X
    if not external_dem_path:
        raise FileNotFoundError(
            "data.use_external_dem=true requires data.external_dem_path to point to a "
            "local, grid-covering DEM mosaic; download and mosaic the licensed USGS 3DEP "
            "tiles first, or set use_external_dem=false"
        )
    from .external_data import augment_from_dem_path
    return augment_from_dem_path(X, external_dem_path, meta, resolution=100)


def load_features_and_labels(feature_path=None, label_path=None, require_labels=True,
                             use_fixture=False, use_external_dem=False,
                             external_dem_path=None):
    """Back-compatible helper: returns (X(H,W,C), y(H,W), feat_meta, label_meta, tags).

    `use_fixture=True` loads data/fixture/ instead: a 512x512 window of the REAL
    competition rasters, stored int16-quantised so it fits in git (the dev sandbox can only
    receive bytes over github.com).  src.fixture verifies the manifest and inverts the
    quantisation to physical units before anything downstream sees the data.
    """
    if use_fixture:
        from .fixture import load_fixture

        X, y, meta, man = load_fixture()
        X = _maybe_add_external_dem(
            X, meta, use_external_dem=use_external_dem,
            external_dem_path=external_dem_path,
        )
        tags = meta.get("band_tags") or [{} for _ in range(X.shape[-1])]
        if len(tags) < X.shape[-1]:
            tags.extend({"description": n} for n in
                         ("dem_slope", "dem_curvature", "dem_tpi", "dem_tri", "dem_detrended"))
        return X, y, meta, meta, tags

    fp = resolve_path(feature_path, FEATURE_NAME_CANDIDATES)
    lp = None
    try:
        lp = resolve_path(label_path, LABEL_NAME_CANDIDATES)
    except FileNotFoundError:
        if require_labels:
            raise
    X, fmeta, tags = load_stack(fp)
    X = _maybe_add_external_dem(
        X, fmeta, use_external_dem=use_external_dem,
        external_dem_path=external_dem_path,
    )
    if len(tags) < X.shape[-1]:
        tags = list(tags) + [{"description": n} for n in
                             ("dem_slope", "dem_curvature", "dem_tpi", "dem_tri", "dem_detrended")]
    if lp is None:
        return X, None, fmeta, None, tags
    y, lmeta = load_labels(lp)
    if y.shape != X.shape[:2]:
        raise ValueError(f"label grid {y.shape} != feature grid {X.shape[:2]} - "
                         "the two rasters must share bounds/resolution (see problem page)")
    return X, y, fmeta, lmeta, tags
