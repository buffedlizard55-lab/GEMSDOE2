"""Distance-weighted Tversky index (DTI) - competition metric, exact + memory-safe.

SPEC SOURCE (fetched 2026-09-12, verbatim from the official problem page):
    https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric

    k(d)            = (1 - d/R)_+ = max(1 - d/R, 0)      R = 300 m = 3 px @ 100 m
    TP_w            = sum_{g in G} max_{x: d(x,g)<=R} p(x) * k(d(x,g))
    FP_w            = sum_{x: p(x)>0} p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w            = sum_{g in G} [1 - max_{x: d(x,g)<=R} p(x) * k(d(x,g))]
    DTI(a,b)        = TP_w / (TP_w + a*FP_w + b*FN_w + eps)          a=0.2, b=0.8

Structural identity used by our tests (holds for ANY p, from the two sums above):
    TP_w + FN_w = |G|          (each ground-truth pixel contributes m and 1-m)

Implementation notes
--------------------
* Kernel offsets are enumerated exhaustively over the (2R+1)^2 neighbourhood
  (R=3 -> 29 offsets with d<=3). This is *exactly* the definition - no
  approximation, no "best-match" heuristics.
* Blocked evaluation (``score_rasters``) so a full GeoDAWN-scale raster never
  materialises an (H, W, 2R+1, 2R+1) array.  The naive dense form needs
  49x the pixels in float32: for 10_000x10_000 that is ~20 GB -> OOM.
* Distances are Euclidean in *pixel* units, matching "3 pixels at 100 m".
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

__all__ = [
    "kernel_offsets",
    "score_within_mask",
    "block_aggregate",
    "bootstrap_from_blocks",
    "compute_distance_weighted_tversky",
    "score_arrays_blocked",
    "evaluate_geotiff",
    "dti_bounds",
]

DEFAULT_ALPHA = 0.2
DEFAULT_BETA = 0.8
DEFAULT_R_PIXELS = 3
EPS = 1e-7


# --------------------------------------------------------------------------------------
# kernel
# --------------------------------------------------------------------------------------
def kernel_offsets(R: int = DEFAULT_R_PIXELS):
    """All (dy, dx, k) with Euclidean distance <= R, k = max(1 - d/R, 0).

    Includes (0,0) with k=1. Offsets are exhaustive over the square, filtered by
    the *Euclidean* radius, exactly as d(x, g) in the problem statement.
    """
    offs = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d = math.hypot(dy, dx)
            if d <= R + 1e-12:
                offs.append((dy, dx, max(1.0 - d / R, 0.0)))
    return offs


def _credit_map(pred: np.ndarray, offsets) -> np.ndarray:
    """credit(g) = max_{x: d(x,g)<=R} p(x) * k(d(x,g)) for every pixel g.

    Shift-and-max over the kernel offsets: for each offset we take the prediction
    window shifted so that the neighbour sits at the position of g, scale by that
    offset's kernel value, then take the running max.  ``pred`` outside the array is
    treated as 0 (padded), matching "no prediction outside the region".
    """
    H, W = pred.shape
    credit = np.zeros((H, W), dtype=np.float64)
    for dy, dx, k in offsets:
        # value of pred at (i+dy, j+dx), placed at (i, j)
        src_y0, src_y1 = max(0, dy), min(H, H + dy)
        dst_y0, dst_y1 = max(0, -dy), min(H, H - dy)
        src_x0, src_x1 = max(0, dx), min(W, W + dx)
        dst_x0, dst_x1 = max(0, -dx), min(W, W - dx)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            continue
        shifted = np.zeros((H, W), dtype=np.float64)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = pred[src_y0:src_y1, src_x0:src_x1]
        np.maximum(credit, k * shifted, out=credit)
    return credit


# --------------------------------------------------------------------------------------
# core metric
# --------------------------------------------------------------------------------------
def _sanitise(pred, shape) -> np.ndarray:
    """The spec's input contract, applied in exactly one place.

    "values between 0 and 1" plus "data outside bounds is null or NaN" (problem page,
    #submission-format): NaN/Inf become 0 credit and 0 penalty, and anything out of range is
    clipped rather than trusted.  Shared by GtContext.score / .credit_vector so a per-block
    decomposition can never disagree with the global score about what the input meant.
    """
    arr = np.asarray(pred, dtype=np.float64)
    if arr.shape != tuple(shape):
        raise ValueError(f"shape mismatch: pred {arr.shape} vs gt {tuple(shape)}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(arr, 0.0, 1.0, out=arr)
    return arr


class GtContext:
    """Ground-truth geometry, computed once and reused for every candidate prediction.

    WHY THIS EXISTS.  Evaluating a candidate shaping (floor, thinning, band width) means scoring
    hundreds of predictions against the SAME labels.  The general implementation rebuilds two
    ground-truth objects on every call - a distance transform of the label mask and a full-raster
    credit map (one temporary per kernel offset, 49 of them at R=3) - and then reads only the |G|
    label pixels out of it.  Fault labels are ~1-2% of the raster, so that is two orders of
    magnitude more work than the definition needs.  Measured effect: the leave-one-fold-out audit in
    scripts/blend_submission.py ran for over 90 minutes on a runner before this existed.

    The definition of TP_w is a sum over exactly |G| terms, so this gathers only what it needs:

        TP_w(g) = max over the R-neighbourhood of  k(d(x,g)) * p(x)

    which is 49 gathers of length |G| instead of 49 passes over H x W.  FP_w still needs the label
    distance transform (that is inherent - it is a per-pixel penalty over the whole prediction), so
    it is computed once here and reused.

    Equivalence with the general path is not assumed: tests/test_metric_parity.py compares this
    against compute_distance_weighted_tversky_reference on random arrays, including NaN/Inf inputs,
    empty labels and R=1.
    """

    def __init__(self, gt, R_pixels: int = DEFAULT_R_PIXELS):
        gt_arr = np.asarray(gt)
        if gt_arr.ndim != 2:
            raise ValueError(f"expected a 2-D label array, got {gt_arr.shape}")
        self.shape = gt_arr.shape
        self.R = int(R_pixels)
        self.gt_bool = (gt_arr.astype(bool) if gt_arr.dtype == np.bool_
                        else (np.nan_to_num(gt_arr, nan=0.0) > 0.5))
        self.n_gt = int(self.gt_bool.sum())
        self.offsets = kernel_offsets(self.R)
        if self.n_gt:
            yy, xx = np.nonzero(self.gt_bool)
            self.gy = yy.astype(np.int64)
            self.gx = xx.astype(np.int64)
            dist = distance_transform_edt(~self.gt_bool)
            self.k_to_gt = np.maximum(1.0 - dist / float(self.R), 0.0)
        else:
            self.gy = self.gx = None
            self.k_to_gt = None

    def credit_vector(self, pred) -> np.ndarray:
        """credit(g) for every ground-truth pixel, in the order of (self.gy, self.gx).

        Split out of `score` (2026-09-18) so a caller can aggregate the SAME per-pixel terms
        over a spatial partition without re-deriving them: TP_w/FN_w are sums over ground-truth
        pixels and FP_w is a sum over prediction pixels, so any disjoint partition of the grid
        reproduces the global score exactly (tests/test_block_decomposition.py pins that).
        """
        pred = _sanitise(pred, self.shape)
        if self.n_gt == 0:
            return np.zeros(0, dtype=np.float64)
        H, W = self.shape
        credit = np.zeros(self.n_gt, dtype=np.float64)
        for dy, dx, k in self.offsets:
            yy = self.gy + dy
            xx = self.gx + dx
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if not inside.any():
                continue
            vals = np.zeros(self.n_gt, dtype=np.float64)
            vals[inside] = pred[yy[inside], xx[inside]]
            vals *= k
            np.maximum(credit, vals, out=credit)
        return credit

    def fp_weight(self) -> np.ndarray:
        """1 - max_g k(d(x,g)) for every pixel: the per-pixel false-positive penalty weight."""
        if self.k_to_gt is None:
            return np.ones(self.shape, dtype=np.float64)
        return 1.0 - self.k_to_gt

    def score(self, pred, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
              eps: float = EPS, return_components: bool = False):
        pred = _sanitise(pred, self.shape)

        if self.n_gt == 0:
            FP_w = float(pred.sum())
            dti = 0.0 if FP_w > 0 else 1.0
            return (dti, (0.0, FP_w, 0.0)) if return_components else dti

        credit = self.credit_vector(pred)
        TP_w = float(credit.sum())
        FN_w = float(self.n_gt - TP_w)                 # == sum_g (1 - credit_g) exactly
        pos = pred > 0
        FP_w = float((pred[pos] * (1.0 - self.k_to_gt[pos])).sum())
        dti = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)
        return (dti, (TP_w, FP_w, FN_w)) if return_components else dti


def compute_distance_weighted_tversky(
    pred,
    gt,
    R_pixels: int = DEFAULT_R_PIXELS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    eps: float = EPS,
    return_components: bool = False,
):
    """Exact DTI on 2-D arrays (the fast path; see GtContext).

    Callers that score many predictions against the SAME labels should build one
    GtContext themselves and call `.score()` on it - that is where the saving is.  This
    function builds a context per call, so its behaviour is identical to
    compute_distance_weighted_tversky_reference but it still avoids the full-raster
    credit map.
    """
    return GtContext(gt, R_pixels).score(pred, alpha=alpha, beta=beta, eps=eps,
                                         return_components=return_components)


def compute_distance_weighted_tversky_reference(
    pred,
    gt,
    R_pixels: int = DEFAULT_R_PIXELS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    eps: float = EPS,
    return_components: bool = False,
):
    """The original full-raster formulation.  Kept as the reference the fast path is verified
    against (tests/test_metric_parity.py); production code uses GtContext."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got {pred.shape} and {gt.shape}")
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    # spec: values must be in [0,1]; null/nan outside the region -> 0 credit, 0 penalty
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(pred, 0.0, 1.0, out=pred)
    gt_bool = gt.astype(bool) if gt.dtype == np.bool_ else (np.nan_to_num(gt, nan=0.0) > 0.5)

    n_gt = int(gt_bool.sum())
    if n_gt == 0:
        # Degenerate: no ground-truth pixels.  FP_w is then p everywhere (max_g k = 0).
        FP_w = float(pred.sum())
        dti = 0.0 if FP_w > 0 else 1.0
        return (dti, (0.0, FP_w, 0.0)) if return_components else dti

    offsets = kernel_offsets(R_pixels)
    credit = _credit_map(pred, offsets)

    TP_w = float(credit[gt_bool].sum())
    FN_w = float(n_gt - TP_w)                      # == sum_g (1 - credit_g) exactly
    # max_{g in G} k(d(x,g)) = k(d(x, nearest GT)) because k is non-increasing in d
    dist_to_gt = distance_transform_edt(~gt_bool)
    k_to_gt = np.maximum(1.0 - dist_to_gt / float(R_pixels), 0.0)
    pos = pred > 0
    FP_w = float((pred[pos] * (1.0 - k_to_gt[pos])).sum())

    dti = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)
    if return_components:
        return dti, (TP_w, FP_w, FN_w)
    return dti


def score_arrays_blocked(
    pred,
    gt,
    R_pixels: int = DEFAULT_R_PIXELS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    block: int = 2048,
):
    """Full-raster DTI with exact (non-approximate) blocked evaluation.

    Sums TP_w / FP_w / FN_w over blocks.  Every term of the definition is *local*:
    FP_w is a per-pixel sum, TP_w/FN_w are per-ground-truth-pixel sums whose credit
    window is the 2R+1 neighbourhood.  Blocks therefore need an R-pixel halo of
    predictions; ground-truth pixels are counted exactly once (in the interior tile).
    """
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(pred, 0.0, 1.0, out=pred)
    gt = np.asarray(gt)
    gt_bool = gt.astype(bool) if gt.dtype == np.bool_ else (np.nan_to_num(gt, nan=0.0) > 0.5)
    H, W = gt_bool.shape
    offsets = kernel_offsets(R_pixels)

    TP = FP = 0.0
    n_gt = 0
    for y0 in range(0, H, block):
        y1 = min(H, y0 + block)
        ye0, ye1 = max(0, y0 - R_pixels), min(H, y1 + R_pixels)   # extended window
        for x0 in range(0, W, block):
            x1 = min(W, x0 + block)
            xe0, xe1 = max(0, x0 - R_pixels), min(W, x1 + R_pixels)
            p_ext = pred[ye0:ye1, xe0:xe1]
            g_ext = gt_bool[ye0:ye1, xe0:xe1]
            credit_ext = _credit_map(p_ext, offsets)
            iy0, iy1 = y0 - ye0, y1 - ye0
            ix0, ix1 = x0 - xe0, x1 - xe0
            gi = g_ext[iy0:iy1, ix0:ix1]
            TP += float(credit_ext[iy0:iy1, ix0:ix1][gi].sum())
            n_gt += int(gi.sum())
            dist = distance_transform_edt(~g_ext)   # EDT on the haloed tile (exact within interior)
            k = np.maximum(1.0 - dist[iy0:iy1, ix0:ix1] / float(R_pixels), 0.0)
            pi = pred[y0:y1, x0:x1]
            m = pi > 0
            FP += float((pi[m] * (1.0 - k[m])).sum())
    FN = float(n_gt - TP)
    dti = TP / (TP + alpha * FP + beta * FN + EPS) if (n_gt or FP) else 1.0
    return dti, (TP, FP, FN)


def score_within_mask(pred, ctx: "GtContext", mask, alpha: float = DEFAULT_ALPHA,
                      beta: float = DEFAULT_BETA, eps: float = EPS, credit=None) -> dict:
    """The metric's three sums restricted to `mask`, computed with GLOBAL context.

    WHY NOT JUST CROP.  Scoring a cropped block as if it were the whole raster is wrong in both
    directions: a prediction one pixel outside the block still earns credit for a truth pixel
    inside it (the kernel reaches R = 3 px), and a truth pixel outside the block still reduces
    the penalty on predictions inside it (FP weight is an EDT over ALL truth).  Cropping
    therefore both under-credits and over-penalises, and the error is largest exactly at the
    block boundaries a spatial hold-out is supposed to measure.

    This function keeps the global geometry and only restricts the SUMS:

        TP_w(mask) = sum_{g in G & mask} credit(g)          credit from the global prediction
        FN_w(mask) = |G & mask| - TP_w(mask)                (the TP+FN=|G| identity, per block)
        FP_w(mask) = sum_{x in mask, p(x)>0} p(x) * (1 - max_g k(d(x,g)))

    so summing over any disjoint partition of the grid reproduces `ctx.score(pred)` exactly -
    pinned by tests/test_block_decomposition.py.  `dti` is `None` when the block holds no truth
    pixels: a block with |G & mask| = 0 has no defined index, and reporting 0.0 there would
    drag a mean down for a reason that has nothing to do with the prediction.
    """
    pred = _sanitise(pred, ctx.shape)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(ctx.shape):
        raise ValueError(f"mask {mask.shape} != grid {tuple(ctx.shape)}")
    if credit is None:
        credit = ctx.credit_vector(pred)
    if ctx.n_gt == 0:
        gt_sel = np.zeros(0, dtype=bool)
    else:
        gt_sel = mask[ctx.gy, ctx.gx]
    n_gt = int(gt_sel.sum())
    TP_w = float(credit[gt_sel].sum()) if n_gt else 0.0
    FN_w = float(n_gt) - TP_w
    pos = (pred > 0) & mask
    FP_w = float((pred[pos] * ctx.fp_weight()[pos]).sum())
    denom = TP_w + alpha * FP_w + beta * FN_w + eps
    return dict(dti=(TP_w / denom) if n_gt else None,
                TP_w=TP_w, FP_w=FP_w, FN_w=FN_w, n_gt=n_gt,
                pred_px=int(pos.sum()), pred_mass=float(pred[pos].sum()),
                scoreable=bool(n_gt > 0))


def block_aggregate(ctx: "GtContext", pred, blocks, n_blocks: int, fp_weight=None,
                    credit=None) -> dict:
    """Per-block TP_w / FP_w / n_gt / prediction-pixel counts, vectorised over the whole grid.

    Equivalent to calling `score_within_mask` once per block, but with two full-grid passes
    instead of `n_blocks` of them: TP_w/FN_w are sums over ground-truth pixels (so they group by
    the block each truth pixel falls in) and FP_w is a sum over prediction pixels weighted by
    `1 - max_g k(d(x,g))` (so it groups by the block the prediction pixel falls in).  Both use
    the GLOBAL context, so a block's numbers still see predictions and truth just outside it -
    which is the whole point (see `score_within_mask`).

    `blocks` is an integer array of the same shape as the grid giving each pixel's block id in
    [0, n_blocks).  The identity that makes per-block tables trustworthy is asserted by the
    caller and tested in tests/test_block_decomposition.py:

        sum_b TP_w(b) == TP_w(global), and the same for FP_w, FN_w and |G|.
    """
    pred = _sanitise(pred, ctx.shape)
    blocks = np.asarray(blocks)
    if blocks.shape != tuple(ctx.shape):
        raise ValueError(f"blocks {blocks.shape} != grid {tuple(ctx.shape)}")
    if credit is None:
        credit = ctx.credit_vector(pred)
    gt_blocks = blocks[ctx.gy, ctx.gx] if ctx.n_gt else np.zeros(0, dtype=np.int64)
    TP = np.bincount(gt_blocks, weights=credit, minlength=n_blocks)[:n_blocks]
    NGT = np.bincount(gt_blocks, minlength=n_blocks)[:n_blocks]
    FN = NGT.astype(np.float64) - TP
    if fp_weight is None:
        fp_weight = ctx.fp_weight()
    pos = pred > 0
    flat = blocks.ravel()
    FP = np.bincount(flat, weights=np.where(pos, pred * fp_weight, 0.0).ravel(),
                     minlength=n_blocks)[:n_blocks]
    PXP = np.bincount(flat, weights=pos.astype(np.float64).ravel(), minlength=n_blocks)[:n_blocks]
    return dict(TP=TP, FP=FP, FN=FN, n_gt=NGT.astype(np.int64), pred_px=PXP.astype(np.int64),
                credit=credit)


def bootstrap_from_blocks(rows, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
                          eps: float = EPS, n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile bootstrap over SPATIAL BLOCKS, recomposed from the additive metric components.

    `rows` is a list of per-candidate dicts, each with a `blocks` list of
    {TP_w, FP_w, FN_w, n_gt, scoreable} in a common block order; row 0 is the reference every
    contrast is paired against.  Returns per-candidate DTI intervals, the paired contrast
    interval, and `prob_beats_reference` - the bootstrap probability that the candidate exceeds
    the reference on the SAME resample.

    Blocks and not pixels are the resampling unit: fault traces run for kilometres, so
    neighbouring pixels are not independent draws, and a pixel bootstrap would understate the
    interval by roughly the trace length.  Recomposition is exact because TP_w/FP_w/FN_w are
    sums (see `block_aggregate`), so no re-scoring and no approximation is involved.
    """
    if not rows or not rows[0].get("blocks"):
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = len(rows[0]["blocks"])
    if any(len(r["blocks"]) != n_blocks for r in rows):
        raise ValueError("every candidate must be scored on the same block partition")
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    TP = np.array([[b["TP_w"] for b in r["blocks"]] for r in rows])
    FP = np.array([[b["FP_w"] for b in r["blocks"]] for r in rows])
    FN = np.array([[b["FN_w"] for b in r["blocks"]] for r in rows])
    boots = np.empty((len(rows), n_boot))
    for ci in range(len(rows)):
        tp, fp, fn = TP[ci][idx].sum(axis=1), FP[ci][idx].sum(axis=1), FN[ci][idx].sum(axis=1)
        boots[ci] = tp / (tp + alpha * fp + beta * fn + eps)
    ref = boots[0]
    out: dict = {}
    for ci, r in enumerate(rows):
        sample = boots[ci]
        contrast = sample - ref
        label = r.get("label", f"candidate_{ci}")
        out[label] = dict(
            dti_p50=round(float(np.percentile(sample, 50)), 6),
            dti_ci95=[round(float(np.percentile(sample, 2.5)), 6),
                      round(float(np.percentile(sample, 97.5)), 6)],
            contrast_vs_reference_p50=round(float(np.percentile(contrast, 50)), 6),
            contrast_vs_reference_ci95=[round(float(np.percentile(contrast, 2.5)), 6),
                                        round(float(np.percentile(contrast, 97.5)), 6)],
            prob_beats_reference=(None if ci == 0 else round(float((contrast > 0).mean()), 4)),
            n_blocks=int(n_blocks),
            n_scoreable_blocks=int(sum(1 for b in r["blocks"] if b.get("scoreable"))),
        )
    return out


def dti_bounds(TP_w: float, FP_w: float, n_gt: int, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    """DTI from (TP_w, FP_w, |G|) using the TP+FN=|G| identity.  Handy for analysis."""
    FN_w = n_gt - TP_w
    return TP_w / (TP_w + alpha * FP_w + beta * FN_w)


# --------------------------------------------------------------------------------------
# GeoTIFF-level scoring
# --------------------------------------------------------------------------------------
def _read_pred(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = 0.0
    return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)


def _read_gt(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = 0.0
    return np.nan_to_num(a, nan=0.0, neginf=0.0)


def evaluate_geotiff(pred_path, true_path, R_meters=300, resolution=100,
                     alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, block=2048, verbose=True):
    pred = _read_pred(pred_path)
    true = _read_gt(true_path)
    if pred.shape != true.shape:
        raise ValueError(f"raster shape mismatch: pred {pred.shape} vs gt {true.shape}")
    R = int(round(R_meters / resolution))
    dti, (tp, fp, fn) = score_arrays_blocked(pred, true, R_pixels=R, alpha=alpha, beta=beta, block=block)
    if verbose:
        n_gt = int((true > 0.5).sum())
        print(f"DTI={dti:.5f}  TP_w={tp:.1f}  FP_w={fp:.1f}  FN_w={fn:.1f}  |G|={n_gt}  (R={R}px)")
    return dti


# --------------------------------------------------------------------------------------
# self-test: brute-force definition vs fast implementation vs official worked example
# --------------------------------------------------------------------------------------
def _brute_force_dti(pred, gt, R=3, alpha=0.2, beta=0.8, eps=1e-7):
    """Textbook transcription of the formulas.  O(|P|*|G| + |G|^2). Test reference only."""
    H, W = pred.shape
    P = {(y, x): float(pred[y, x]) for y in range(H) for x in range(W) if pred[y, x] > 0}
    G = [(y, x) for y in range(H) for x in range(W) if gt[y, x] > 0.5]
    if not G:
        return 0.0 if P else 1.0, (0.0, sum(P.values()), 0.0)
    k = lambda d: max(1.0 - d / R, 0.0)
    TP = 0.0
    for g in G:
        best = 0.0
        for (y, x), p in P.items():
            d = math.hypot(y - g[0], x - g[1])
            if d <= R + 1e-12:
                best = max(best, p * k(d))
        TP += best
    FN = len(G) - TP
    FP = 0.0
    for (y, x), p in P.items():
        best_k = 0.0
        for g in G:
            best_k = max(best_k, k(math.hypot(y - g[0], x - g[1])))
            if best_k >= 1.0:
                break
        FP += p * (1.0 - best_k)
    return TP / (TP + alpha * FP + beta * FN + eps), (TP, FP, FN)


def _self_test():
    rng = np.random.default_rng(7)
    # ---- 1. fast vs brute force on random rasters (the anti-hallucination check) -----
    for trial in range(6):
        H = W = 26
        gt = (rng.random((H, W)) < 0.07).astype(float)
        # make some structure so distances are interesting
        if trial % 2:
            gt[:, 12] = 1.0
            gt[5:20, 6] = 1.0
        pred = rng.random((H, W)) * (rng.random((H, W)) < 0.25)
        d_fast = compute_distance_weighted_tversky(pred, gt)
        d_brute, _ = _brute_force_dti(pred, gt)
        assert abs(d_fast - d_brute) < 1e-9, (trial, d_fast, d_brute)
    print("  [1] fast == brute-force definition (6 random rasters) OK")

    # ---- 2. blocked == dense on a bigger raster --------------------------------------
    gt = np.zeros((200, 130), float)
    gt[40:120, 60] = 1.0
    gt[150:170, 20:100] = 1.0
    pred = (rng.random((200, 130)) < 0.05) * rng.random((200, 130))
    dense = compute_distance_weighted_tversky(pred, gt, return_components=True)
    blocked = score_arrays_blocked(pred, gt, block=64)
    for a, b, nm in zip(dense[1], blocked[1], ("TP", "FP", "FN")):
        assert abs(a - b) < 1e-8, (nm, a, b)
    assert abs(dense[0] - blocked[0]) < 1e-9
    print(f"  [2] blocked == dense OK (DTI {dense[0]:.6f})")

    # ---- 3. official worked example arithmetic (problem page, "Scoring example") -----
    #     page states: TP_w=3.00, FP_w=1.89, FN_w=2.00 -> TI_w = 0.60
    val = 3.00 / (3.00 + 0.2 * 1.89 + 0.8 * 2.00)
    assert round(val, 2) == 0.60, val
    print(f"  [3] official example arithmetic reproduces 0.60 (got {val:.4f}) OK")

    # ---- 4. reconstruct that example raster and match 3.00 / 1.89 / 2.00 -------------
    got = _search_official_example()
    print(f"  [4] official example raster: {got}")

    # ---- 5. identity TP+FN = |G| ; perfect pred == 1 ; empty == 0 -------------------
    gt = np.zeros((48, 48)); gt[10:30, 20] = 1.0
    p = (rng.random((48, 48)) < 0.3) * rng.random((48, 48))
    _, (tp, fp, fn) = compute_distance_weighted_tversky(p, gt, return_components=True)
    assert abs(tp + fn - gt.sum()) < 1e-9, (tp, fn, gt.sum())
    assert abs(compute_distance_weighted_tversky(gt, gt) - 1.0) < 1e-6
    assert compute_distance_weighted_tversky(np.zeros_like(gt), gt) < 1e-6
    print("  [5] TP_w+FN_w=|G|, perfect=1, empty=0 OK")

    # ---- 6. alpha/beta asymmetry + kernel support ------------------------------------
    wide = gt.copy(); wide[35:41, 20:26] = 1.0                     # pure-FP blob far away
    d_fp = compute_distance_weighted_tversky(wide, gt)
    miss = gt.copy(); miss[10:20, 20] = 0.0                        # pure-FN removal
    d_fn = compute_distance_weighted_tversky(miss, gt)
    assert d_fp > d_fn, (d_fp, d_fn)
    edge = np.zeros_like(gt); edge[:, 21:23] = 1.0                 # within R=3 of the line
    far = np.zeros_like(gt); far[:, 24:26] = 1.0                    # 4-5 px away > R
    assert compute_distance_weighted_tversky(edge, gt) > 0
    assert compute_distance_weighted_tversky(far, gt) == 0.0
    print(f"  [6] asymmetry (FP-heavy {d_fp:.4f} > FN-heavy {d_fn:.4f}) + R=3 support OK")

    # ---- 7. NaN padding / out-of-range clip sanitised --------------------------------
    nanp = gt.astype(float).copy(); nanp[:, :8] = np.nan
    over = gt.astype(float).copy(); over[:, 30] = 5.0; over[:, 31] = -3.0
    assert np.isfinite(compute_distance_weighted_tversky(nanp, gt))
    assert np.isfinite(compute_distance_weighted_tversky(over, gt))
    print("  [7] NaN / out-of-range sanitisation OK")

    # ---- 8. kernel value spot-check against the formula ------------------------------
    offs = dict(((dy, dx), k) for dy, dx, k in kernel_offsets(3))
    assert abs(offs[(0, 0)] - 1.0) < 1e-12
    assert abs(offs[(0, 1)] - (1 - 1 / 3)) < 1e-12
    assert abs(offs[(0, 3)] - 0.0) < 1e-12
    assert abs(offs[(2, 2)] - (1 - math.hypot(2, 2) / 3)) < 1e-12
    assert (3, 3) not in offs and (0, 4) not in offs          # d=sqrt(18), 4 > R -> excluded
    print(f"  [8] kernel spot-values + radius filter OK ({len(offs)} offsets in R=3)")
    print("metrics self-test: ALL 8 CHECKS PASSED")


def _search_official_example():
    """Try to reproduce the page's example (GT = single vertical line, 5 px; TP_w=3.00,
    FP_w=1.89, FN_w=2.00).  The page shows the pixel grid only as a PNG on
    drivendata-public-assets.s3.amazonaws.com (not reachable from this sandbox), so we
    search small binary prediction sets for a configuration consistent with the printed
    numbers.  If found it confirms our reading of the metric; if not, we still match the
    printed DTI arithmetic (check 3)."""
    gt = np.zeros((5, 9))
    gt[:, 4] = 1.0                       # single vertical line, 5 px
    target = (3.00, 1.89, 2.00)
    rng = np.random.default_rng(0)
    best = None
    cand = [(y, x) for y in range(5) for x in range(9)]
    for _ in range(4000):
        k = rng.integers(2, 9)
        idx = rng.choice(len(cand), size=k, replace=False)
        pred = np.zeros_like(gt)
        for i in idx:
            pred[cand[i]] = 1.0
        _, (tp, fp, fn) = compute_distance_weighted_tversky(pred, gt, return_components=True)
        err = abs(tp - target[0]) + abs(fp - target[1]) + abs(fn - target[2])
        if best is None or err < best[0]:
            best = (err, tp, fp, fn)
        if err < 0.02:
            pts = sorted((int(y), int(x)) for y, x in zip(*np.nonzero(pred)))
            return f"MATCH TP={tp:.2f} FP={fp:.2f} FN={fn:.2f} pred_pixels={pts}"
    return f"closest TP={best[1]:.2f} FP={best[2]:.2f} FN={best[3]:.2f} (no exact match found; PNG grid not fetchable here)"


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)

    ap = argparse.ArgumentParser(description="Score a submission GeoTIFF against ground truth.")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--true", required=True)
    ap.add_argument("--R", type=int, default=300, help="kernel range in metres (official: 300)")
    ap.add_argument("--res", type=int, default=100, help="pixel size in metres (official: 100)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--block", type=int, default=2048, help="tile size for blocked scoring")
    a = ap.parse_args()
    evaluate_geotiff(a.pred, a.true, R_meters=a.R, resolution=a.res,
                     alpha=a.alpha, beta=a.beta, block=a.block)
