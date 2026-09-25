#!/usr/bin/env python3
"""Cut a small, REAL-data development fixture out of the competition rasters.

Why: the full feature stack is 419 MB and the dev sandbox can only receive bytes
through ordinary git objects over github.com (Actions artifacts, Releases and LFS
all live on hosts the sandbox cannot reach).  Committing 419 MB is not acceptable,
but a few MB is — and a few MB of *real* competition data is worth far more for
development than any synthetic stand-in, because it carries the true band
semantics, nodata pattern, value ranges, CRS and label sparsity.

What it does:
  * scores every candidate window by fault-pixel count and data validity
  * picks the window with the most faults subject to >= `min_valid` valid fraction
  * writes features (int16-quantised + zstd) and labels (uint8) for that window
  * records the exact de-quantisation constants and a round-trip error measurement,
    so `src/fixture.py` can reconstruct physical units and prove the error is bounded

The fixture is explicitly NOT for leaderboard training - it is a correctness
harness.  Full training runs on the runner against the full-size rasters.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def pick_window(labels_path: Path, size: int, feats_path: Path, min_valid: float = 0.95):
    with rasterio.open(labels_path) as src:
        y = src.read(1).astype("float32")
        nod = src.nodata
    valid = np.isfinite(y)
    if nod is not None:
        valid &= y != nod
    pos = valid & (y > 0.5)
    H, W = y.shape

    # integral images -> O(1) window sums
    ip = np.pad(pos.astype("int64").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    iv = np.pad(valid.astype("int64").cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def box(I, r, c):
        return int(I[r + size, c + size] - I[r, c + size] - I[r + size, c] + I[r, c])

    best = None
    step = max(32, size // 4)
    for r in range(0, H - size + 1, step):
        for c in range(0, W - size + 1, step):
            nv = box(iv, r, c)
            if nv < min_valid * size * size:
                continue
            npos = box(ip, r, c)
            if best is None or npos > best[0]:
                best = (npos, nv, r, c)
    if best is None:
        raise SystemExit(f"no window of {size}px reaches {min_valid:.0%} valid coverage")
    npos, nv, r, c = best
    return {
        "row": r,
        "col": c,
        "size": size,
        "fault_px": npos,
        "valid_px": nv,
        "valid_fraction": nv / (size * size),
        "fault_fraction": npos / (size * size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data/fixture")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = d / "gems-geodawn-numerical-features.tif"
    labs = d / "existing_faults.tif"

    win = pick_window(labs, a.size, feats)
    print("window:", json.dumps(win))
    W = Window(win["col"], win["row"], a.size, a.size)

    with rasterio.open(feats) as src:
        cube = src.read(window=W).astype("float64")  # (C,h,w)
        nod = src.nodata
        transform = src.window_transform(W)
        crs = src.crs
        descs = [src.descriptions[i] for i in range(src.count)]
        btags = [dict(src.tags(i + 1) or {}) for i in range(src.count)]
    finite = np.isfinite(cube) & (np.abs(cube) < 1e30)
    if nod is not None:
        finite &= cube != nod

    C = cube.shape[0]
    q = np.zeros_like(cube, dtype="int16")
    meta = []
    NA = -32768
    for i in range(C):
        v = cube[i][finite[i]]
        lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        # full-range (no clipping) so round-trip error is pure quantisation:
        # 32000 levels over [lo,hi] -> max error = (hi-lo)/64000
        scale = (hi - lo) / 32000.0
        z = np.rint((cube[i] - lo) / scale) - 16000
        z = np.where(finite[i], np.clip(z, -16000, 16000), NA)
        q[i] = z.astype("int16")
        recon = (q[i].astype("float64") + 16000) * scale + lo
        err = np.abs(recon[finite[i]] - cube[i][finite[i]]) if finite[i].any() else np.array([0.0])
        meta.append(
            {
                "band": i + 1,
                "band_name": btags[i].get("band_name"),
                "description": descs[i],
                "data_category": btags[i].get("data_category"),
                "lo": lo,
                "hi": hi,
                "scale": scale,
                "offset": -16000,
                "nodata_int16": NA,
                "valid_fraction": float(finite[i].mean()),
                "max_abs_roundtrip_err": float(err.max()),
                "max_rel_roundtrip_err": float(err.max() / max(hi - lo, 1e-12)),
            }
        )

    prof = dict(
        driver="GTiff", height=a.size, width=a.size, count=C, dtype="int16",
        crs=crs, transform=transform, nodata=NA, compress="zstd", zstd_level=22,
        predictor=2, tiled=True, blockxsize=256, blockysize=256,
    )
    fpath = out / "fixture_features_int16.tif"
    with rasterio.open(fpath, "w", **prof) as dst:
        dst.write(q)
        for i in range(C):
            dst.set_band_description(i + 1, descs[i] or f"band_{i + 1}")
            dst.update_tags(i + 1, **btags[i])

    with rasterio.open(labs) as src:
        yl = src.read(1, window=W)
        lnod = src.nodata
    ybin = (yl > 0.5).astype("uint8")
    lpath = out / "fixture_labels.tif"
    with rasterio.open(
        lpath, "w", driver="GTiff", height=a.size, width=a.size, count=1, dtype="uint8",
        crs=crs, transform=transform, nodata=255, compress="zstd", zstd_level=22,
    ) as dst:
        dst.write(ybin, 1)

    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/make_dev_fixture.py on a GitHub-hosted runner",
        "purpose": (
            "Small REAL-data slice of the official competition rasters so the sandbox "
            "(github.com-only egress) can develop and test against true band semantics. "
            "NOT a training set - full training runs on the runner against the full rasters."
        ),
        "source_files": {
            "features": feats.name,
            "labels": labs.name,
        },
        "window": win,
        "crs": str(crs),
        "transform": list(transform)[:6],
        "n_bands": C,
        "bands": meta,
        "labels": {
            "path": lpath.name,
            "positive_px": int(ybin.sum()),
            "nodata_in_source": (None if lnod is None else float(lnod)),
        },
        "dequantise": "physical = (int16 + 16000) * scale + lo   (int16 == -32768 means nodata)",
        "file_bytes": {
            fpath.name: fpath.stat().st_size,
            lpath.name: lpath.stat().st_size,
        },
    }
    (out / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("window", "n_bands", "labels", "file_bytes")}, indent=2))
    print("max rel round-trip error:", max(m["max_rel_roundtrip_err"] for m in meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
