"""Losses for the GEMS Prize.

Why not just TverskyLoss (what the reference solution uses)?
    Reference: https://github.com/drivendataorg/gems-prize-reference-solution, cell 16 ->
        criterion = TverskyLoss(alpha=0.2, beta=0.8, mode="binary")
    The *scoring* metric, however, credits a prediction for a ground-truth (GT) pixel if a
    prediction lies within R=300 m (3 px) of it (triangular kernel).  A plain pixel-wise
    Tversky punishes exactly the off-by-one rasterisation error the committee removed from
    the metric ("Rasterization is lossy and can induce off-by-one errors near pixel
    boundaries" - problem page, Performance metric).

`DistanceWeightedTverskyLoss` is the *differentiable surrogate of the actual competition
metric*: it reproduces TP_w / FP_w / FN_w of src/metrics.py exactly (verified in
tests/test_metric.py::test_dw_loss_matches_metric).  It is the objective we train on.

Sources (verified):
- metric definition: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
- alpha=0.2 / beta=0.8: same page ("we set alpha = 0.2 and beta = 0.8")
- Tversky index background: https://en.wikipedia.org/wiki/Tversky_index
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# kernel helpers (shared by loss and the metric-based early stopping)
# --------------------------------------------------------------------------------------
def kernel_offsets(R: int = 3, device=None):
    """(dy, dx, k) with sqrt(dy^2+dx^2) <= R, k = max(1 - d/R, 0) - same as src.metrics."""
    offs = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d = math.hypot(dy, dx)
            if d <= R + 1e-12:
                offs.append((dy, dx, max(1.0 - d / R, 0.0)))
    return offs


def _kernel_tensor(R: int, device, dtype):
    offs = kernel_offsets(R)
    size = 2 * R + 1
    k = torch.zeros(size, size, device=device, dtype=dtype)
    for dy, dx, val in offs:
        k[dy + R, dx + R] = val
    return k, offs


def soft_credit(p: torch.Tensor, R: int = 3) -> torch.Tensor:
    """credit(y,x) = max_{(dy,dx)} p(y+dy, x+dx) * k(dy,dx), zero outside the patch.

    p: (B,1,H,W) probabilities.  Returns same shape.  Exact, differentiable
    (subgradient flows through the winning offset only).
    """
    B, _, H, W = p.shape
    pad = F.pad(p, (R, R, R, R), mode="constant", value=0.0)
    best = None
    for dy, dx, kval in kernel_offsets(R):
        # view of p(y+dy, x+dx) anchored at (y, x)
        shifted = pad[:, :, R + dy: R + dy + H, R + dx: R + dx + W]
        if kval == 0.0:
            continue
        w = kval * shifted
        best = w if best is None else torch.maximum(best, w)
    if best is None:
        best = torch.zeros_like(p)
    return best


def distance_weighted_counts(probs, target, R=3, fp_weight=None):
    """(TP_w, FP_w, FN_w) of the competition metric, batchwise, torch.

    fp_weight: (B,1,H,W) or None.  fp_weight = 1 - max_g k(d(x,g)) per pixel (how much a
        prediction at x counts as a false positive).  Pass the PRECOMPUTED map cropped
        from the *global* EDT (FaultDataset stores it as `fpw`): computing an EDT per patch
        both stalls training and truncates the distance to the nearest GT at the patch
        border, which would over-penalise predictions next to a patch edge.
    """
    if fp_weight is None:
        g = target[:, 0] if target.dim() == 4 else target
        fp_weight = torch.as_tensor(fp_weights_np(g), device=probs.device, dtype=probs.dtype)[:, None]
    credit = soft_credit(probs, R=R)
    t = (target > 0.5).to(probs.dtype)
    n_gt = t.sum(dim=(1, 2, 3))
    TP = (credit * t).sum(dim=(1, 2, 3))
    FN = n_gt - TP
    pos = (probs > 0).to(probs.dtype)
    FP = (probs * fp_weight * pos).sum(dim=(1, 2, 3))
    return TP, FP, FN, n_gt


def fp_weights_np(gt_binary_2d_or_BHW, R: int = 3):
    """1 - max_g k(d(x,g)) per pixel; needs scipy (CPU). Accepts (H,W) or (B,H,W)."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    a = np.asarray(gt_binary_2d_or_BHW)
    squeeze = False
    if a.ndim == 2:
        a = a[None]
        squeeze = True
    out = np.empty(a.shape, dtype=np.float32)
    for i in range(a.shape[0]):
        g = a[i] > 0.5
        if not g.any():
            out[i] = 1.0
            continue
        d = distance_transform_edt(~g)
        out[i] = 1.0 - np.maximum(1.0 - d / float(R), 0.0)
    return out[0] if squeeze else out


# --------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------
class TverskyLoss(nn.Module):
    """Pixel-wise soft Tversky (reference solution's criterion, for comparison/A-B runs)."""

    def __init__(self, alpha=0.2, beta=0.8, smooth=1e-7):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets, fp_weight=None):
        # fp_weight accepted (and ignored) so every loss shares one call signature:
        # train.py always passes it.  A crash here is what the 2026-09-12 A/B run found.
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        p = torch.sigmoid(logits).flatten(1)
        t = targets.flatten(1).float()
        TP = (p * t).sum(1)
        FP = (p * (1 - t)).sum(1)
        FN = ((1 - p) * t).sum(1)
        return 1 - ((TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)).mean()


class DistanceWeightedTverskyLoss(nn.Module):
    """1 - DTI of the *official metric*, differentiable.  The objective we train on.

    logits : (B,1,H,W) or (B,H,W) raw scores
    targets: (B,H,W) or (B,1,H,W) binary
    fp_weight (optional, precomputed): (B,1,H,W) float, 1 - max_g k(d(x,g)).
        Precompute once per patch in FaultDataset (`fpw`); computing a scipy EDT per
        training step is the difference between a usable epoch and a stalled one.
    """

    def __init__(self, alpha=0.2, beta=0.8, R=3, eps=1e-7, ignore_bg_penalty=0.0):
        super().__init__()
        self.alpha, self.beta, self.R, self.eps = alpha, beta, R, eps
        self.ignore_bg_penalty = ignore_bg_penalty  # 0 = faithful to metric

    def forward(self, logits, targets, fp_weight=None):
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        p = torch.sigmoid(logits)[:, None]
        t = targets[:, None]
        TP, FP, FN, n_gt = distance_weighted_counts(
            p, t, R=self.R,
            fp_weight=(fp_weight[:, None] if fp_weight is not None and fp_weight.dim() == 3 else fp_weight),
        )
        dti = TP / (TP + self.alpha * FP + self.beta * FN + self.eps)
        # patches with no GT pixels: DTI undefined (|G|=0); penalise only via FP mass
        empty = (n_gt == 0)
        if empty.any():
            dti = torch.where(empty, 1.0 - self.ignore_bg_penalty * (FP / p[0].numel()).clamp(0, 1), dti)
        return (1.0 - dti).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=0.75, smooth=1e-7):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets, fp_weight=None):
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        p = torch.sigmoid(logits).flatten(1)
        t = targets.flatten(1).float()
        TP = (p * t).sum(1)
        FP = (p * (1 - t)).sum(1)
        FN = ((1 - p) * t).sum(1)
        tv = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return ((1 - tv) ** self.gamma).mean()


class CombinedLoss(nn.Module):
    """BCE (dense gradient, stable early training) + DW-Tversky (the metric) + focal Tversky.

    Defaults tuned for this metric: the DW term carries the majority of the weight because
    it is the only term aligned with how submissions are scored.
    """

    def __init__(self, alpha=0.2, beta=0.8, bce_weight=0.3, dw_weight=0.5, focal_weight=0.2,
                 focal_gamma=0.75, R=3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dw = DistanceWeightedTverskyLoss(alpha=alpha, beta=beta, R=R)
        self.focal = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=focal_gamma)
        self.w = (bce_weight, dw_weight, focal_weight)

    def forward(self, logits, targets, fp_weight=None):
        t = targets.unsqueeze(1) if targets.dim() == 3 else targets
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        t = t.float()
        b = self.bce(logits, t)
        d = self.dw(logits, targets, fp_weight=fp_weight)
        f = self.focal(logits, targets)
        return self.w[0] * b + self.w[1] * d + self.w[2] * f
