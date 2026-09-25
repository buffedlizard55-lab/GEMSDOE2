#!/usr/bin/env python3
"""Decide how to move 419 MB of competition features into a github.com-only sandbox.

Context: the dev sandbox reaches github.com and pypi.org and nothing else. Actions
ARTIFACTS are served from an Azure blob host that is blocked, as are Releases
(objects.githubusercontent.com) and Git LFS (github-cloud.s3). The one channel that
works is ordinary git objects over github.com. So a *compact* derivative of the
features has to be small enough to commit without bloating the repo.

This script measures, rather than guesses:
  * the bounding box of actually-valid (non-nodata) pixels, per band and overall
  * whether all bands share one validity mask
  * byte sizes for several candidate encodings of the cropped cube
  * whether example_submission.tif is bit-identical in content to existing_faults.tif
    (the problem page says the sample "predicts total fault absence"; the measured
    stats say otherwise, so this is checked explicitly and reported)

Writes data/evidence/transfer_analysis.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio


def valid_mask_and_bbox(path: Path):
    with rasterio.open(path) as src:
        nod = src.nodata
        masks, per_band = [], []
        for i in range(1, src.count + 1):
            a = src.read(i)
            m = np.isfinite(a)
            if nod is not None:
                m &= a != nod
            m &= np.abs(a) < 1e30
            masks.append(m)
            rows, cols = np.where(m)
            per_band.append(
                {
                    "index": i,
                    "valid_px": int(m.sum()),
                    "bbox_rowcol": [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())]
                    if m.any()
                    else None,
                }
            )
        allm = np.logical_or.reduce(masks)
        anym = np.logical_and.reduce(masks)
        rows, cols = np.where(allm)
        bbox = [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())]
        return {
            "shape": [src.height, src.width],
            "union_valid_px": int(allm.sum()),
            "intersection_valid_px": int(anym.sum()),
            "all_bands_share_mask": bool((allm == anym).all()),
            "union_bbox_rowcol": bbox,
            "cropped_shape": [bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1],
            "per_band": per_band,
        }


def encoding_trials(path: Path, bbox, out: dict) -> None:
    """Write the cropped cube under several encodings and record file sizes."""
    r0, r1, c0, c1 = bbox
    win = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    with rasterio.open(path) as src:
        data = src.read(window=win)  # (C,h,w) float32
        prof = src.profile.copy()
        transform = src.window_transform(win)
    C, h, w = data.shape
    nod = prof.get("nodata")
    finite = np.isfinite(data) & (np.abs(data) < 1e30)
    if nod is not None:
        finite &= data != nod

    trials = {}
    base = dict(driver="GTiff", height=h, width=w, count=C, crs=prof["crs"], transform=transform)

    def write(name, **kw):
        p = Path(tempfile.gettempdir()) / f"trial_{name}.tif"
        with rasterio.open(p, "w", **kw["profile"]) as dst:
            dst.write(kw["data"])
        trials[name] = int(p.stat().st_size)
        p.unlink()

    d32 = np.where(finite, data, np.nan).astype("float32")
    write("float32_zstd22_pred3",
          profile={**base, "dtype": "float32", "nodata": float("nan"), "compress": "zstd",
                   "zstd_level": 22, "predictor": 3, "tiled": True, "blockxsize": 256,
                   "blockysize": 256},
          data=d32)
    write("float32_deflate9_pred3",
          profile={**base, "dtype": "float32", "nodata": float("nan"), "compress": "deflate",
                   "zlevel": 9, "predictor": 3, "tiled": True, "blockxsize": 256,
                   "blockysize": 256},
          data=d32)

    # int16 quantisation: per band, 65535 levels across [p0.1, p99.9], clipped.
    q = np.zeros((C, h, w), dtype="int16")
    qmeta = []
    for i in range(C):
        v = data[i][finite[i]]
        lo, hi = (float(np.percentile(v, 0.1)), float(np.percentile(v, 99.9))) if v.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        scale = (hi - lo) / 65000.0
        z = np.clip((data[i] - lo) / scale, 0, 65000).astype("int32") - 32000
        z = np.where(finite[i], z, -32768)
        q[i] = z.astype("int16")
        qmeta.append({"band": i + 1, "lo": lo, "hi": hi, "scale": scale, "offset": -32000})
        # measured round-trip error on valid pixels
        recon = (q[i].astype("float64") + 32000) * scale + lo
        err = np.abs(recon[finite[i]] - data[i][finite[i]])
        rng = max(hi - lo, 1e-12)
        qmeta[-1]["max_abs_err_relative_to_range"] = float(err.max() / rng) if err.size else 0.0
    write("int16_zstd22_pred2",
          profile={**base, "dtype": "int16", "nodata": -32768, "compress": "zstd",
                   "zstd_level": 22, "predictor": 2, "tiled": True, "blockxsize": 256,
                   "blockysize": 256},
          data=q)
    out["quantisation"] = qmeta
    out["encoding_bytes"] = trials
    out["cropped_cube_raw_bytes"] = int(C * h * w * 4)


def compare_sample_to_labels(sample: Path, labels: Path) -> dict:
    with rasterio.open(sample) as s, rasterio.open(labels) as l:  # noqa: E741
        a = s.read(1).astype("float64")
        b = l.read(1).astype("float64")
        an, bn = s.nodata, l.nodata
    amask = np.isfinite(a) & ((a != an) if an is not None and np.isfinite(an) else True)
    bmask = np.isfinite(b) & ((b != bn) if bn is not None else True)
    both = amask & bmask
    eq = bool(np.array_equal(a[both] > 0.5, b[both] > 0.5))
    return {
        "sample": sample.name,
        "labels": labels.name,
        "same_shape": a.shape == b.shape,
        "sample_positive_px": int((a[amask] > 0.5).sum()),
        "labels_positive_px": int((b[bmask] > 0.5).sum()),
        "binary_content_identical_on_common_valid": eq,
        "n_pixels_compared": int(both.sum()),
        "problem_page_claim": (
            "A sample submission that predicts total fault absence is provided for your "
            "reference on the data download page. "
            "(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/"
            "#submission-format)"
        ),
        "claim_holds": bool(int((a[amask] > 0.5).sum()) == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/evidence/transfer_analysis.json")
    a = ap.parse_args()
    d = Path(a.data_dir)
    feats = d / "gems-geodawn-numerical-features.tif"
    res: dict = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/analyze_for_transfer.py on a GitHub-hosted runner",
    }
    if feats.exists():
        res["features"] = valid_mask_and_bbox(feats)
        encoding_trials(feats, res["features"]["union_bbox_rowcol"], res)
    smp, lab = d / "example_submission.tif", d / "existing_faults.tif"
    if smp.exists() and lab.exists():
        res["sample_vs_labels"] = compare_sample_to_labels(smp, lab)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "quantisation"}, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
