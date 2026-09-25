#!/usr/bin/env python
"""Stress-test the shaping morphology against mislocalisation.

WHY.  `dominant_thin` is optimal when the predicted trace lies on the true trace: TP_w takes
a max within R pixels, so extra thickness is pure FP mass.  But the competition is scored
against faults that are NEW to the expert-reviewed dataset -- faults the model never saw in
training and therefore cannot be expected to localise as precisely as the catalogued ones it
did train on.  If the model's line is off by more than R, a thin prediction scores ZERO on
that fault while still paying FP mass elsewhere.

This script quantifies the exchange rate on data we actually have: it takes a submission (or
any probability map) plus a label raster, and recomputes DTI while translating the labels by
a set of offsets (simulating "the true fault is a few cells away from where we drew it") and
while growing the prediction by k pixels.  The output is the table that says which morphology
degrades least under mislocalisation -- i.e. what to use for the scored (new-fault) universe.

HONEST LIMITS, stated up front: the label raster here is the PUBLIC catalogue, which is NOT
the scored set, and translating the catalogue is a proxy for localisation error -- it is not a
measurement of the new faults.  It is a stress test of the operator, and it is labelled as
such wherever the number is reported.

Usage
    python scripts/measure_shift_robustness.py --pred submission.tif --labels labels.tif \
        --window 2048,1280,512 --dilate 0,1,2,3 --shifts 0,2,3 --out data/evidence/shift_robustness.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import compute_distance_weighted_tversky        # noqa: E402
from src.submission_optim import dilate_mask                      # noqa: E402


def offsets_at(radius: int):
    """Identity + the 8 compass neighbours at `radius` (a fixed, reviewable set)."""
    if radius <= 0:
        return [(0, 0)]
    r = int(radius)
    return [(0, 0), (0, r), (0, -r), (r, 0), (-r, 0), (r, r), (r, -r), (-r, r), (-r, -r)]


def shifted(arr: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(arr, fill)
    h, w = arr.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = arr[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--window", default=None, help="row,col,size — restrict to one window")
    ap.add_argument("--dilate", default="0,1,2,3")
    ap.add_argument("--shifts", default="0,2,3", help="radii of the translation set")
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with rasterio.open(a.pred) as src:
        q = src.read(1).astype(np.float32)
    with rasterio.open(a.labels) as src:
        lab = src.read(1)
        if src.nodata is not None:
            lab = np.where(lab == src.nodata, 0, lab)
    gt = (np.asarray(lab, np.float32) > 0.5).astype(np.float32)

    if a.window:
        row, col, size = (int(v) for v in a.window.split(","))
        # a raster that is ALREADY the window (an extracted fixture slice) is left alone;
        # a full-grid raster is cropped to the window
        if q.shape[0] > size or q.shape[1] > size:
            q = q[row:row + size, col:col + size]
        if gt.shape[0] > size or gt.shape[1] > size:
            gt = gt[row:row + size, col:col + size]
    if gt.shape != q.shape:
        gt = gt[: q.shape[0], : q.shape[1]]

    dil = [int(v) for v in a.dilate.split(",")]
    radii = [int(v) for v in a.shifts.split(",")]
    base = np.nan_to_num(q) > 0.5
    rows = []
    for k in dil:
        pred = dilate_mask(base, radius=k) if k else base.astype(np.float32)
        row = dict(dilate=k, kept_px=int(pred.sum()), per_shift={})
        for r in radii:
            vals = []
            for dy, dx in offsets_at(r):
                vals.append(float(compute_distance_weighted_tversky(pred, shifted(gt, dy, dx),
                                                                   R_pixels=a.R, alpha=a.alpha,
                                                                   beta=a.beta)))
            row["per_shift"][str(r)] = dict(mean=float(np.mean(vals)), worst=float(np.min(vals)),
                                            n=len(vals))
        rows.append(row)

    print(f"prediction {a.pred}: {int(base.sum())} px kept; labels {int(gt.sum())} px "
          f"(R={a.R} px = {a.R * 100} m)")
    hdr = "  dilate  kept_px  " + "".join(f"  shift=±{r}px mean (worst)  " for r in radii)
    print(hdr)
    for row in rows:
        cells = "".join(f"{row['per_shift'][str(r)]['mean']:.4f} ({row['per_shift'][str(r)]['worst']:.4f})"
                        .rjust(26) for r in radii)
        print(f"{row['dilate']:>8d}{row['kept_px']:>9d}  {cells}")

    # which operator loses least between "no offset" and the largest offset?
    r_big = max(radii)
    base_row = next(r for r in rows if r["dilate"] == min(dil))
    print("\nrobustness ranking (largest shift radius):")
    for row in sorted(rows, key=lambda r: -r["per_shift"][str(r_big)]["mean"]):
        drop = row["per_shift"][str(r_big)]["mean"] - row["per_shift"]["0"]["mean"]
        print(f"  dilate={row['dilate']}: mean {row['per_shift'][str(r_big)]['mean']:.4f}, "
              f"worst {row['per_shift'][str(r_big)]['worst']:.4f}, "
              f"vs no-offset change {drop:+.4f}")

    if a.out:
        Path(a.out).write_text(json.dumps(dict(
            generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            script="scripts/measure_shift_robustness.py",
            prediction=a.pred, labels=a.labels, window=a.window,
            R_pixels=a.R, alpha=a.alpha, beta=a.beta,
            caveat=("The label raster is the PUBLIC catalogue, not the scored (new-fault) set; "
                    "translating it is a stress test of the shaping operator against "
                    "mislocalisation, not a measurement of the competition metric."),
            rows=rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
