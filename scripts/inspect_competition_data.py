#!/usr/bin/env python3
"""Machine-generate a verifiable inventory of whatever sits in data/.

Everything this writes is *measured* from the bytes on disk (hashlib + rasterio),
never transcribed from documentation.  That is the whole point: the sandbox cannot
reach Dropbox, so the only trustworthy description of the competition rasters is
one computed by a machine that could download them, committed back verbatim.

Outputs (JSON, one file each, plus a combined index):
  <out>/inventory.json      sizes + sha256 + `file(1)`-style magic sniff
  <out>/rasters.json        per-raster: CRS, transform, bounds, res, dtype, nodata,
                            per-band tags (description/data_category), per-band stats
  <out>/labels_summary.json label raster: positive count/fraction, value histogram

Usage:  python scripts/inspect_competition_data.py --data-dir data --out data/evidence
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

# Anything a GeoTIFF/PDF should start with, so a Dropbox HTML error page is caught
# instead of being silently treated as data.
MAGIC = {
    b"II*\x00": "TIFF (little-endian)",
    b"MM\x00*": "TIFF (big-endian)",
    b"II+\x00": "BigTIFF (little-endian)",
    b"MM\x00+": "BigTIFF (big-endian)",
    b"%PDF": "PDF",
    b"<!DO": "HTML (!! not data - download failed)",
    b"<htm": "HTML (!! not data - download failed)",
}


def sniff(path: Path) -> str:
    head = path.open("rb").read(8)
    for sig, name in MAGIC.items():
        if head.startswith(sig):
            return name
    return f"unknown (first bytes: {head!r})"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_raster(path: Path) -> dict:
    import rasterio

    out: dict = {"path": path.name}
    with rasterio.open(path) as src:
        out.update(
            driver=src.driver,
            width=src.width,
            height=src.height,
            count=src.count,
            crs=str(src.crs),
            epsg=(src.crs.to_epsg() if src.crs else None),
            transform=list(src.transform)[:6],
            res=[float(src.res[0]), float(src.res[1])],
            bounds=[float(b) for b in src.bounds],
            dtypes=list(src.dtypes),
            nodata=(None if src.nodata is None else float(src.nodata)),
            dataset_tags=dict(src.tags() or {}),
            block_shapes=[list(b) for b in src.block_shapes],
            compression=(src.compression.name if src.compression else None),
        )
        bands = []
        for i in range(1, src.count + 1):
            a = src.read(i).astype("float64")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            # GDAL commonly encodes nodata as a huge sentinel; treat it as missing
            a[np.abs(a) > 1e30] = np.nan
            finite = np.isfinite(a)
            b: dict = {
                "index": i,
                "description": src.descriptions[i - 1],
                "tags": dict(src.tags(i) or {}),
                "valid_fraction": float(finite.mean()),
            }
            if finite.any():
                v = a[finite]
                b.update(
                    min=float(v.min()),
                    max=float(v.max()),
                    mean=float(v.mean()),
                    std=float(v.std()),
                    p01=float(np.percentile(v, 1)),
                    p50=float(np.percentile(v, 50)),
                    p99=float(np.percentile(v, 99)),
                    n_unique_sample=int(np.unique(v[:: max(1, v.size // 100000)]).size),
                )
            bands.append(b)
        out["bands"] = bands
    return out


def summarize_labels(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        nod = src.nodata
    finite = np.isfinite(a)
    if nod is not None:
        finite &= a != nod
    v = a[finite]
    vals, counts = np.unique(v, return_counts=True)
    top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda t: -t[1])[:20]
    return {
        "path": path.name,
        "shape": list(a.shape),
        "n_pixels": int(a.size),
        "n_valid": int(finite.sum()),
        "n_positive_gt_0p5": int((v > 0.5).sum()),
        "positive_fraction_of_valid": float((v > 0.5).mean()) if v.size else None,
        "value_histogram_top20": [{"value": val, "count": cnt} for val, cnt in top],
        "min": float(v.min()) if v.size else None,
        "max": float(v.max()) if v.size else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/evidence")
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    files = sorted(p for p in data.iterdir() if p.is_file())
    inventory = {
        "generated_utc": stamp,
        "generated_by": "scripts/inspect_competition_data.py on a GitHub-hosted runner",
        "note": "All values measured from the downloaded bytes. Not transcribed from docs.",
        "files": [
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
                "magic": sniff(p),
            }
            for p in files
        ],
    }
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2))

    rasters, errors = [], []
    for p in files:
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        try:
            rasters.append(inspect_raster(p))
        except Exception as exc:  # noqa: BLE001 - record, never crash the bridge
            errors.append({"file": p.name, "error": repr(exc)})
    (out / "rasters.json").write_text(
        json.dumps({"generated_utc": stamp, "rasters": rasters, "errors": errors}, indent=2) + "\n"
    )

    for p in files:
        if "fault" in p.name.lower() or "label" in p.name.lower():
            try:
                (out / "labels_summary.json").write_text(
                    json.dumps({"generated_utc": stamp, **summarize_labels(p)}, indent=2) + "\n"
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"file": p.name, "error": repr(exc)})
            break

    for e in errors:
        print("ERROR:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
