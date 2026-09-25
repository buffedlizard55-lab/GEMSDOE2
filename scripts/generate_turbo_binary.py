#!/usr/bin/env python3
"""
Generate a BINARY TURBO submission for in-browser generation.
This creates a submission with only values 0.0, 1.0, and NaN
which can be encoded for browser-based generation.
"""

import hashlib
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
from scipy import ndimage
from skimage.filters import gaussian, sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_io import conform_to_template


def load_raster(path):
    """Load raster and return array + profile"""
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def process_in_chunks(arr, func, chunk_size=500):
    """Process array in chunks to save memory."""
    result = np.zeros_like(arr)
    h, w = arr.shape
    
    for i in range(0, h, chunk_size):
        for j in range(0, w, chunk_size):
            chunk = arr[i:i+chunk_size, j:j+chunk_size]
            result[i:i+chunk_size, j:j+chunk_size] = func(chunk)
    
    return result


def safe_normalize(arr):
    """Safely normalize array to [0, 1]."""
    arr = arr.copy()
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0
    
    if np.max(arr) > np.min(arr):
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    return np.clip(arr, 0, 1)


def detect_edges_simple(band, sigma=2):
    """Simplified edge detection."""
    try:
        # Smooth
        smoothed = gaussian(band, sigma=sigma, preserve_range=True)
        
        # Normalize
        smoothed = safe_normalize(smoothed)
        
        # Sobel gradient
        grad = sobel(smoothed)
        grad = safe_normalize(grad)
        
        return grad
    except:
        return np.zeros_like(band)


def create_binary_turbo(labels_path, features_path, sample_path):
    """Create binary TURBO submission (values are only 0.0, 1.0, NaN)."""
    print("Loading data...")
    
    labels_arr, _ = load_raster(labels_path)
    sample_arr, sample_profile = load_raster(sample_path)
    
    # Load first few bands only to save memory
    with rasterio.open(features_path) as src:
        # Read only bands 0-4 (magnetic and gravity)
        features_stack = src.read([1, 2, 3, 4, 5])
    
    print(f"  Labels: {labels_arr.shape}")
    print(f"  Features: {features_stack.shape}")
    print(f"  Sample: {sample_arr.shape}")
    
    # Known faults
    known_faults = (labels_arr == 1).astype(np.float32)
    
    # Process each band with edge detection
    edge_maps = []
    print("Processing bands...")
    
    for band_idx in range(features_stack.shape[0]):
        band = features_stack[band_idx]
        edge = process_in_chunks(band, lambda x: detect_edges_simple(x, sigma=2), chunk_size=500)
        edge_maps.append(edge)
        print(f"  Band {band_idx}: done")
    
    # Average edge responses
    avg_edges = np.mean(edge_maps, axis=0)
    print(f"Average edges computed: {avg_edges.shape}")
    
    # Create final prediction - BINARY ONLY
    prediction = np.zeros_like(sample_arr, dtype=np.float32)
    
    # Known faults at 1.0
    prediction[known_faults == 1] = 1.0
    
    # New detections based on edge strength
    # Boost near known faults
    distance_from_known = ndimage.distance_transform_edt(known_faults == 0)
    proximity_boost = np.exp(-distance_from_known / 500)
    
    # Combine
    new_detections = avg_edges * proximity_boost
    
    # Threshold to create binary
    threshold = 0.35
    binary_new = (new_detections > threshold).astype(np.float32)
    
    # Combine with known faults
    prediction = np.maximum(prediction, binary_new)
    
    # Connected component filtering
    labeled, num_features = ndimage.label(prediction > 0.5)
    if num_features > 0:
        sizes = np.bincount(labeled.ravel())[1:]
        keep = sizes >= 3
        binary_pred = np.isin(labeled, np.where(keep)[0] + 1).astype(np.float32)
        prediction = binary_pred
    
    print(f"Final prediction: {prediction.shape}")
    print(f"Non-zero pixels: {np.count_nonzero(prediction > 0.5)}")
    print(f"Unique values: {np.unique(prediction[np.isfinite(prediction)])}")
    
    # Conform to template
    print("Conforming to template...")
    conformed, stats = conform_to_template(prediction, sample_arr)
    
    return conformed, sample_profile, sample_arr


def generate_binary_turbo():
    """Generate a binary TURBO submission."""
    
    labels_path = ROOT / "data/labels.tif"
    features_path = ROOT / "data/training_features.tif"
    sample_path = ROOT / "data/sample_submission.tif"
    
    if not all(p.exists() for p in [labels_path, features_path, sample_path]):
        print("ERROR: Required data files not found.")
        return False
    
    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "data/evidence/runs" / f"turbo-binary-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    try:
        conformed, profile, sample_arr = create_binary_turbo(labels_path, features_path, sample_path)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save submission
    out_tif = output_dir / "submission.tif"
    out_sha = output_dir / "submission.sha256"
    out_rep = output_dir / "turbo_report.json"
    out_log = output_dir / "validation.log"
    
    print("Saving submission...")
    
    # Update profile
    profile.update(
        count=1,
        dtype="float32",
        compress="lzw",
        nodata=np.nan,
    )
    
    # Write GeoTIFF
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(conformed, 1)
        dst.update_tags(source="GEMSDOE2-TURBO-BINARY")
        dst.update_tags(strategy="Binary multi-scale edge detection")
        dst.update_tags(timestamp=timestamp)
    
    # Calculate SHA256
    raw_bytes = out_tif.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    out_sha.write_text(f"{sha256}  submission.tif\n")
    
    # Generate report
    valid_mask = ~np.isnan(sample_arr)
    n_valid = int(valid_mask.sum())
    n_nonzero = int(np.count_nonzero(conformed[valid_mask] > 0.5))
    
    report = {
        "strategy": "GEMSDOE2-TURBO-BINARY",
        "timestamp": timestamp,
        "file": str(out_tif.relative_to(ROOT)),
        "sha256": sha256,
        "short_sha": sha256[:12],
        "bytes": len(raw_bytes),
        "binary": True,
        "statistics": {
            "valid_pixels": n_valid,
            "nonzero_pixels": n_nonzero,
            "nan_pixels": int(np.isnan(conformed).sum()),
        },
        "unique_name": f"turbo-binary-{timestamp}-{sha256[:8]}.tif",
        "note": f"GEMSDOE2-TURBO-BINARY: Binary Multi-Scale Edge · build {sha256[:8]} · {timestamp}",
    }
    
    out_rep.write_text(json.dumps(report, indent=2))
    
    # Validate
    print("Validating...")
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_submission.py"),
            "--pred",
            str(out_tif),
            "--sample",
            str(sample_path),
            "--train",
            str(ROOT / "data/training_features.tif"),
        ],
        capture_output=True,
        text=True,
    )
    
    val_output = res.stdout + ("\n" + res.stderr if res.stderr else "")
    out_log.write_text(val_output)
    
    if res.returncode != 0:
        print("VALIDATION FAILED!")
        print(val_output)
        return False
    
    print(val_output)
    print(f"\n✅ SUCCESS: TURBO-BINARY submission generated!")
    print(f"File: {out_tif}")
    print(f"SHA256: {sha256}")
    print(f"Unique name: {report['unique_name']}")
    print(f"Note: {report['note']}")
    
    return True


if __name__ == "__main__":
    success = generate_binary_turbo()
    sys.exit(0 if success else 1)
