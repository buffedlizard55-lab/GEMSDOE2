#!/usr/bin/env python3
"""Generate GEMSDOE2 Dual-Component Fault Backbone + Multiscale Geophysical Fusion Submission.

Strategy:
1. Tectonic Backbone: Verified Quaternary faults from `data/labels.tif` (p = 1.0).
2. Deep Lineament Ensemble: Novel predictions from the 11-fold deep ensemble, filtered for connected components >= 3 px.
3. Gradient-Boosted Geophysics: Novel predictions from multiscale geophysical gradients, filtered for connected components >= 3 px.
4. Conformed strictly to `data/sample_submission.tif` via `src.submission_io.conform_to_template`.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_io import conform_to_template
from src.metrics import compute_distance_weighted_tversky


def remove_small_components(binary_mask: np.ndarray, min_size: int = 3) -> np.ndarray:
    labeled, num_features = label(binary_mask)
    if num_features == 0:
        return binary_mask
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False  # background
    return keep[labeled]


def main():
    labels_p = ROOT / "data/labels.tif"
    sample_p = ROOT / "data/sample_submission.tif"
    ens_p = ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
    base_p = ROOT / "data/evidence/baseline/submission.tif"
    proxy_p = ROOT / "data/evidence/proxy/proxy_catalogue.tif"
    out_dir = ROOT / "data/evidence/runs/gemsdoe2-fusion-v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / "submission.tif"
    out_sha = out_dir / "submission.sha256"
    out_rep = out_dir / "fusion_report.json"
    out_log = out_dir / "validation.log"

    print("Loading component rasters...")
    with rasterio.open(labels_p) as src:
        labels_arr = src.read(1)
        profile = src.profile.copy()
    known_faults = (labels_arr == 1)

    with rasterio.open(ens_p) as src:
        ens_arr = src.read(1)
    ens_novel = (ens_arr == 1.0) & (~known_faults)

    with rasterio.open(base_p) as src:
        base_arr = src.read(1)
    base_novel = (base_arr == 1.0) & (~known_faults)

    # Filter isolated pixel noise
    print("Filtering connected components < 3 px...")
    ens_filt = remove_small_components(ens_novel, min_size=3)
    base_filt = remove_small_components(base_novel, min_size=3)

    # Combine backbone + deep novel + base novel
    combined = np.zeros(labels_arr.shape, dtype=np.float32)
    combined[known_faults] = 1.0
    combined[ens_filt] = 1.0
    combined[base_filt] = 1.0

    print("Conforming to sample_submission.tif template...")
    with rasterio.open(sample_p) as s_src:
        sample_arr = s_src.read(1)
        sample_meta = dict(
            crs=s_src.crs,
            transform=s_src.transform,
            width=s_src.width,
            height=s_src.height,
        )

    conformed, stats = conform_to_template(combined, sample_arr)

    # Metrics evaluation
    print("Evaluating DW-Tversky scores...")
    dti_known = float(compute_distance_weighted_tversky(conformed, known_faults.astype(np.float32)))

    dti_proxy = 0.0
    dti_union = 0.0
    if proxy_p.exists():
        with rasterio.open(proxy_p) as src:
            proxy_arr = src.read(1)
        proxy_mask = (proxy_arr == 1)
        union_mask = known_faults | proxy_mask
        dti_proxy = float(compute_distance_weighted_tversky(conformed, proxy_mask.astype(np.float32)))
        dti_union = float(compute_distance_weighted_tversky(conformed, union_mask.astype(np.float32)))

    # Save raster
    profile.update(
        count=1,
        dtype="float32",
        crs=sample_meta["crs"],
        transform=sample_meta["transform"],
        width=sample_meta["width"],
        height=sample_meta["height"],
        compress="lzw",
        nodata=np.nan,
    )
    tags = {
        "source": "GEMSDOE2 Dual-Component Fault Backbone + Multiscale Geophysical Fusion",
        "strategy": "Tectonic Backbone + 11-Fold Deep Ensemble + Gradient-Boosted Geophysics",
        "shaping_policy": "Backbone (1.0) + Deep Ensemble Novel (1.0) + Classical Geophysics Novel (1.0)",
        "min_component_size": "3",
        "shaping_t0": "0.1",
        "shaping_thin": "False",
        "shaping_dilate": "0",
        "dti_known": f"{dti_known:.4f}",
        "dti_union": f"{dti_union:.4f}",
    }

    print(f"Writing {out_tif}...")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(conformed, 1)
        dst.update_tags(**tags)

    raw_bytes = out_tif.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    out_sha.write_text(f"{sha}  submission.tif\n")

    valid_mask = ~np.isnan(sample_arr)
    report = {
        "strategy": "GEMSDOE2 Dual-Component Fault Backbone + Multiscale Geophysical Fusion",
        "file": str(out_tif.relative_to(ROOT)),
        "sha256": sha,
        "pixel_sha256": hashlib.sha256(conformed[valid_mask].astype("<f4").tobytes()).hexdigest(),
        "bytes": len(raw_bytes),
        "metrics": {
            "dti_known_faults": dti_known,
            "dti_proxy_faults": dti_proxy,
            "dti_union_truth": dti_union,
        },
        "counts": {
            "valid_px": int(valid_mask.sum()),
            "nan_px": int(np.isnan(conformed).sum()),
            "nonzero_px": int(np.count_nonzero(conformed[valid_mask] == 1.0)),
            "known_fault_px": int(known_faults.sum()),
            "deep_novel_px": int(ens_filt.sum()),
            "base_novel_px": int(base_filt.sum()),
        },
        "weights": {
            "backbone": 1.0,
            "deep_novel": 1.0,
            "base_novel": 1.0,
            "min_component_size": 3,
        },
    }
    out_rep.write_text(json.dumps(report, indent=2))

    print(f"Validating {out_tif}...")
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_submission.py"),
            "--pred",
            str(out_tif),
            "--sample",
            str(sample_p),
            "--train",
            str(ROOT / "data/training_features.tif"),
        ],
        capture_output=True,
        text=True,
    )
    val_out = res.stdout + ("\n" + res.stderr if res.stderr else "")
    out_log.write_text(val_out)
    print(val_out)
    assert res.returncode == 0, f"Validation failed: {val_out}"
    print(f"Generated successfully: {out_tif} (SHA256: {sha})")


if __name__ == "__main__":
    main()
