#!/usr/bin/env python3
"""
Build a RECONSTRUCTED training dataset from OFFICIAL PUBLIC sources.

THIS IS NOT THE OFFICIAL COMPETITION TRAINING DATA.
The official files (training_features.tif / existing_faults.tif) are
login-gated on DrivenData and cannot be fetched from this sandbox (egress
allowlist - see LIMITATIONS.md section 1). This script instead assembles a
real (non-synthetic) dataset from the same official public sources the
competition derives its data from, so the full pipeline can run end-to-end.

Sources (all fetched from github.com allowlisted egress; underlying data is
USGS public domain / INGENIOUS, DOI 10.15121/1881483):
  - GeoDAWN airborne survey 22103 area 1 rasters (mag + radiometrics + DEM),
    already EPSG:32611, 50 m  [USGS, DOI 10.5066/P93LGLVQ]
  - INGENIOUS regional quaternary faults shapefile (label SOURCE)  [EPSG:4269]
  - INGENIOUS dependent earthquake rate-density GeoTIFF ("density of
    earthquakes" feature SOURCE)  [Albers-117, 500 m]

Output: data/reconstructed/
  recon_training_features.tif  16 bands float32, EPSG:32611, 100 m
  recon_labels.tif             binary fault mask (rasterized QFaults lines)
  recon_sample_submission.tif  zeros (total fault absence), same grid
  provenance.json              SHA256 of every input, band->official-source map
"""
import hashlib, json, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/external/jklinck"
OUT = ROOT / "data/reconstructed"
GEODAWN = SRC / "GeoDAWN_tiffs/22103_area1_tiffs"
EQ = SRC / "seismicity_INGENIOUS_regional_data/dependent_eq/geotiffs/deq_n100a05.tif"
FAULTS = SRC / "faults_quaternary_INGENIOUS_regional_data/faults_quaternary_regional.shp"

# Common grid: 100 m, EPSG:32611, inside the intersection of all area1 extents
GRID = dict(crs="EPSG:32611", x0=406700.0, y0=4202700.0, width=667, height=489)  # y0 = top

# Band name -> (source tif stem or None, official-band approximation note)
BANDS = [
    ("22103_dem_a1",           "dem",        "(analog) elevation input for detrended elevation + slope bands"),
    ("22103_rtp_a1",           "rtp",        "(analog) reduced-to-pole magnetic anomaly"),
    ("22103_tmi_a1",           "tmi",        "(analog) total magnetic intensity"),
    ("22103_tmi_vg_a1",        "tmi_vg",     "(analog) vertical gradient ~ vertical slope of TMI"),
    ("22103_tmi_hg_a1",        "tmi_hg",     "(analog) horizontal gradient ~ horizontal slope of TMI"),
    ("22103_upcont_tmi150_a1", "upcont_tmi150", "(extra) upward-continued TMI, regional depth signal"),
    ("22103_tc_a1",            "tc",         "(analog) total radiometric counts"),
    ("22103_k_a1",             "k",          "(analog) potassium concentration"),
    ("22103_th_a1",            "th",         "(analog) thorium concentration"),
    ("22103_u_a1",             "u",          "(analog) uranium concentration"),
    ("22103_thk_a1",           "thk",        "(extra) Th/K ratio"),
    ("22103_uk_a1",            "uk",         "(extra) U/K ratio"),
    ("22103_uth_a1",           "uth",        "(extra) U/Th ratio"),
]
NODATA_SRC = -999999.0


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def resample_to_grid(src_path, dst_transform, dst_crs, width, height, method=Resampling.bilinear):
    with rasterio.open(src_path) as s:
        src = s.read(1).astype(np.float32)
        src_nodata = s.nodata if s.nodata is not None else NODATA_SRC
        src_crs = s.crs
    src = np.where((src == src_nodata) | (src < -1e30) | ~np.isfinite(src), np.nan, src)
    # fill NaN with mean for resampling stability, restore mask after
    valid = np.isfinite(src)
    fill = np.nanmean(src) if valid.any() else 0.0
    src_filled = np.where(valid, src, fill).astype(np.float32)
    dst = np.zeros((height, width), np.float32)
    reproject(src_filled, dst,
              src_transform=rasterio.open(src_path).transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resample=method)
    # recompute destination-side validity from source footprint: any output
    # equal to the far-fill of an all-nan neighborhood is masked via src mask warp
    mask = np.zeros((height, width), np.uint8)
    reproject((~valid).astype(np.uint8), mask,
              src_transform=rasterio.open(src_path).transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resample=Resampling.nearest)
    dst[mask > 0] = np.nan
    return dst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    transform = from_origin(GRID["x0"], GRID["y0"], 100.0, 100.0)
    H, W = GRID["height"], GRID["width"]

    prov = {"generated": time.strftime("%F %T"), "note": "RECONSTRUCTED dataset - NOT official competition files",
            "grid": {**GRID, "resolution_m": 100, "bounds_32611": [GRID["x0"], GRID["y0"] - H * 100, GRID["x0"] + W * 100, GRID["y0"]]},
            "source_repo": "https://github.com/jklinck/geothermal_research",
            # pinned upstream commit (verified 2026-09-12 via api.github.com and recorded in
            # data/external/jklinck_pinned_commit.txt; the tree was delivered as a
            # codeload tarball because this sandbox blocks raw.githubusercontent.com)
            "source_commit": (SRC.parent / "jklinck_pinned_commit.txt").read_text().strip()
                             if (SRC.parent / "jklinck_pinned_commit.txt").exists() else "unknown",
            "inputs": {}, "bands": []}

    # ---- GeoDAWN bands ----
    stack = []
    for stem, name, note in BANDS:
        p = GEODAWN / f"{stem}.tif"
        arr = resample_to_grid(p, transform, GRID["crs"], W, H)
        stack.append(arr)
        prov["inputs"][p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        prov["bands"].append({"index": len(stack), "name": name, "source": str(p.relative_to(SRC)), "maps_to": note})
        print(f"  band {len(stack):2d} {name:14s} valid={np.isfinite(arr).mean()*100:5.1f}%")

    # ---- DEM derivatives (official band analogs: detrended elevation + its slope) ----
    dem = stack[0]
    valid = np.isfinite(dem)
    med = np.nanmedian(dem)
    dem_f = np.where(valid, dem, med)
    from scipy.ndimage import gaussian_filter, sobel
    smooth = gaussian_filter(dem_f, sigma=30, mode="nearest")  # ~3 km low-pass
    detrended = dem_f - smooth
    gy, gx = np.gradient(dem_f, 100.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    detrended[~valid] = 0.0
    slope[~valid] = 0.0
    stack.append(detrended.astype(np.float32))
    prov["bands"].append({"index": 14, "name": "dem_detrended", "source": "derived from 22103_dem_a1 (gaussian sigma=30px)", "maps_to": "(analog) detrended elevation"})
    stack.append(slope.astype(np.float32))
    prov["bands"].append({"index": 15, "name": "dem_slope", "source": "derived from 22103_dem_a1 (gradient, degrees)", "maps_to": "(analog) slope of detrended elevation"})

    # ---- Earthquake density (official band analog) ----
    eq = resample_to_grid(EQ, transform, GRID["crs"], W, H)
    eq = np.where(np.isfinite(eq), eq, 0.0).astype(np.float32)
    stack.append(eq)
    prov["inputs"][EQ.name] = {"sha256": sha256(EQ), "bytes": EQ.stat().st_size}
    prov["bands"].append({"index": 16, "name": "eq_density", "source": str(EQ.relative_to(SRC)), "maps_to": "(analog) density of earthquakes (INGENIOUS dependent_eq rate-density)"})

    feats = np.stack(stack).astype(np.float32)
    assert feats.shape == (16, H, W), feats.shape

    # ---- Labels: rasterize quaternary faults ----
    import geopandas as gpd
    gdf = gpd.read_file(FAULTS)
    prov["inputs"][FAULTS.name] = {"sha256": sha256(FAULTS), "bytes": FAULTS.stat().st_size, "features": int(len(gdf)), "crs_source": str(gdf.crs)}
    gdf = gdf.to_crs(GRID["crs"])
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None)
    labels = rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="uint8")

    # ---- Write outputs ----
    prof = dict(driver="GTiff", height=H, width=W, count=16, dtype="float32",
                crs=GRID["crs"], transform=transform, nodata=None, compress="lzw")
    fpath = OUT / "recon_training_features.tif"
    with rasterio.open(fpath, "w", **prof) as dst:
        dst.write(feats)
        for i, b in enumerate(prov["bands"], start=1):
            dst.set_band_description(i, b["name"])
            dst.update_tags(i, name=b["name"], maps_to=b["maps_to"], source=b["source"])
    lpath = OUT / "recon_labels.tif"
    with rasterio.open(lpath, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="uint8", crs=GRID["crs"], transform=transform, nodata=None, compress="lzw") as dst:
        dst.write(labels, 1)
        dst.set_band_description(1, "rasterized quaternary faults (all_touched)")
    spath = OUT / "recon_sample_submission.tif"
    with rasterio.open(spath, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs=GRID["crs"], transform=transform, nodata=None, compress="lzw") as dst:
        dst.write(np.zeros((H, W), np.float32), 1)
        dst.set_band_description(1, "total fault absence (format template)")

    prov["outputs"] = {f.name: {"sha256": sha256(f), "bytes": f.stat().st_size} for f in [fpath, lpath, spath]}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))

    print(f"\nfeatures: {feats.shape} {feats.dtype} | fault pixels: {labels.mean()*100:.2f}% ({int(labels.sum())} px)")
    print(f"finite fraction: {np.isfinite(feats).mean()*100:.1f}% | range: [{np.nanmin(feats):.3g}, {np.nanmax(feats):.3g}]")
    print(f"labels unique: {np.unique(labels)}")
    print(f"wrote {OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
