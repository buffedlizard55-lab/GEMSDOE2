#!/usr/bin/env python3
"""
Generate multiple TURBO submission variants with different parameters
for testing and optimization.
"""

import hashlib
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import itertools

import numpy as np
import rasterio
from scipy import ndimage
from skimage.feature import canny
from skimage.filters import gaussian

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_io import conform_to_template


def load_raster(path):
    """Load raster and return array + profile"""
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def detect_edges_multiscale(data, sigmas=[1, 2, 4]):
    """Multi-scale edge detection."""
    edge_probs = []
    for sigma in sigmas:
        smoothed = gaussian(data, sigma=sigma, preserve_range=True)
        if np.nanstd(smoothed) > 0:
            smoothed_norm = (smoothed - np.nanmean(smoothed)) / np.nanstd(smoothed)
        else:
            smoothed_norm = smoothed
        edges = canny(smoothed_norm, sigma=sigma)
        edge_probs.append(edges.astype(np.float32))
        grad = np.hypot(*np.gradient(smoothed_norm))
        edge_probs.append(np.clip(grad, 0, 1))
    return np.mean(edge_probs, axis=0)


def create_variant_fusion(labels_path, features_path, sample_path, 
                         edge_weight, lineament_weight, grad_weight,
                         threshold, min_component_size, proximity_decay):
    """Create a single variant with given parameters."""
    labels_arr, _ = load_raster(labels_path)
    sample_arr, sample_profile = load_raster(sample_path)
    
    with rasterio.open(features_path) as src:
        features_stack = src.read()
    
    known_faults = (labels_arr == 1).astype(np.float32)
    
    # Edge detection on first 10 bands
    edge_maps = []
    for band_idx in range(min(10, features_stack.shape[0])):
        band_data = features_stack[band_idx]
        edges = detect_edges_multiscale(band_data, sigmas=[1, 2, 3])
        edge_maps.append(edges)
    avg_edges = np.mean(edge_maps, axis=0)
    
    # Lineament detection (simplified)
    lineament_maps = []
    for band_idx in range(min(5, features_stack.shape[0])):
        band_data = features_stack[band_idx]
        grad_x, grad_y = np.gradient(band_data)
        grad_mag = np.hypot(grad_x, grad_y)
        if np.nanstd(grad_mag) > 0:
            grad_mag_norm = (grad_mag - np.nanmean(grad_mag)) / np.nanstd(grad_mag)
            grad_mag_norm = np.clip(grad_mag_norm, 0, 1)
        else:
            grad_mag_norm = grad_mag
        lineament_maps.append(grad_mag_norm)
    avg_lineaments = np.mean(lineament_maps, axis=0)
    
    # Gradient analysis
    gradient_maps = []
    for band_idx in range(features_stack.shape[0]):
        band_data = features_stack[band_idx]
        grad_x, grad_y = np.gradient(band_data)
        grad_mag = np.hypot(grad_x, grad_y)
        if np.nanstd(grad_mag) > 0:
            grad_mag_norm = (grad_mag - np.nanmean(grad_mag)) / np.nanstd(grad_mag)
            grad_mag_norm = np.clip(grad_mag_norm, 0, 1)
        else:
            grad_mag_norm = grad_mag
        gradient_maps.append(grad_mag_norm)
    avg_gradients = np.mean(gradient_maps, axis=0)
    
    # Clean inputs
    for arr in [avg_edges, avg_lineaments, avg_gradients]:
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0
        if np.max(arr) > 0:
            arr = arr / np.max(arr)
    
    # Weighted fusion
    fused = (edge_weight * avg_edges + 
             lineament_weight * avg_lineaments + 
             grad_weight * avg_gradients)
    
    # Sigmoid
    fused_prob = 1 / (1 + np.exp(-10 * (fused - 0.5)))
    
    # Proximity boost
    distance_from_known = ndimage.distance_transform_edt(known_faults == 0)
    proximity_boost = np.exp(-distance_from_known / proximity_decay)
    
    # Final prediction
    final_prediction = np.zeros_like(sample_arr, dtype=np.float32)
    final_prediction[known_faults == 1] = 0.95
    new_detections = fused_prob * proximity_boost
    final_prediction = np.maximum(final_prediction, new_detections)
    
    # Threshold and clean
    binary_pred = final_prediction > threshold
    labeled, num_features = ndimage.label(binary_pred)
    if num_features > 0:
        sizes = np.bincount(labeled.ravel())[1:]
        keep = sizes >= min_component_size
        binary_pred = np.isin(labeled, np.where(keep)[0] + 1)
    
    # Distance transform for smooth probabilities
    distance_to_pred = ndimage.distance_transform_edt(binary_pred == 0)
    final_prob = np.clip(1 - (distance_to_pred / 1000), 0, 1).astype(np.float32)
    final_prob[known_faults == 1] = np.maximum(final_prob[known_faults == 1], 0.9)
    
    # Conform to template
    conformed, _ = conform_to_template(final_prob, sample_arr)
    
    return conformed


def generate_variants():
    """Generate multiple parameter variants for testing."""
    
    labels_path = ROOT / "data/labels.tif"
    features_path = ROOT / "data/training_features.tif"
    sample_path = ROOT / "data/sample_submission.tif"
    
    if not all(p.exists() for p in [labels_path, features_path, sample_path]):
        print("ERROR: Required data files not found.")
        return False
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "data/evidence/runs" / f"turbo-variants-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameter grid
    edge_weights = [0.3, 0.4, 0.5]
    lineament_weights = [0.2, 0.3, 0.4]
    grad_weights = [0.2, 0.3, 0.4]
    thresholds = [0.3, 0.35, 0.4]
    min_sizes = [3, 5]
    decays = [300, 500, 700]
    
    # Limit combinations for practicality
    print("Generating parameter variants...")
    print(f"Total combinations: {len(edge_weights) * len(lineament_weights) * len(grad_weights) * len(thresholds) * len(min_sizes) * len(decays)}")
    
    variants = []
    count = 0
    max_variants = 20  # Limit to 20 variants
    
    for ew, lw, gw, th, ms, dec in itertools.product(
        edge_weights, lineament_weights, grad_weights, 
        thresholds, min_sizes, decays
    ):
        if count >= max_variants:
            break
        
        # Normalize weights
        total = ew + lw + gw
        if total > 0:
            ew_norm = ew / total
            lw_norm = lw / total
            gw_norm = gw / total
        else:
            ew_norm, lw_norm, gw_norm = 1/3, 1/3, 1/3
        
        try:
            conformed = create_variant_fusion(
                labels_path, features_path, sample_path,
                ew_norm, lw_norm, gw_norm, th, ms, dec
            )
            
            # Save variant
            variant_name = f"v{count:02d}-ew{ew_norm:.1f}-lw{lw_norm:.1f}-gw{gw_norm:.1f}-th{th}-ms{ms}-dec{dec}"
            variant_tif = output_dir / f"{variant_name}.tif"
            
            profile = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'nodata': np.nan,
                'width': conformed.shape[1],
                'height': conformed.shape[0],
                'count': 1,
                'crs': sample_path,  # Will be updated
                'transform': None,  # Will be updated
                'compress': 'lzw',
            }
            
            # Get proper profile from sample
            with rasterio.open(sample_path) as src:
                profile.update({
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                })
            
            with rasterio.open(variant_tif, "w", **profile) as dst:
                dst.write(conformed, 1)
                dst.update_tags({
                    'variant': variant_name,
                    'edge_weight': f"{ew_norm:.2f}",
                    'lineament_weight': f"{lw_norm:.2f}",
                    'grad_weight': f"{gw_norm:.2f}",
                    'threshold': f"{th}",
                    'min_component_size': f"{ms}",
                    'proximity_decay': f"{dec}",
                })
            
            # Validate
            res = subprocess.run(
                [sys.executable, str(ROOT / "scripts/validate_submission.py"),
                 "--pred", str(variant_tif), "--sample", str(sample_path)],
                capture_output=True, text=True
            )
            
            if res.returncode == 0:
                sha256 = hashlib.sha256(variant_tif.read_bytes()).hexdigest()
                n_nonzero = int(np.count_nonzero(conformed[~np.isnan(conformed)] > 0.1))
                
                variants.append({
                    'name': variant_name,
                    'file': str(variant_tif.relative_to(ROOT)),
                    'sha256': sha256,
                    'parameters': {
                        'edge_weight': ew_norm,
                        'lineament_weight': lw_norm,
                        'grad_weight': gw_norm,
                        'threshold': th,
                        'min_component_size': ms,
                        'proximity_decay': dec,
                    },
                    'nonzero_pixels': n_nonzero,
                    'status': 'VALID',
                })
                print(f"  ✓ {variant_name} - {n_nonzero} nonzero px - SHA: {sha256[:12]}")
            else:
                variants.append({
                    'name': variant_name,
                    'status': 'INVALID',
                    'error': res.stderr[:200],
                })
                print(f"  ✗ {variant_name} - INVALID")
            
            count += 1
            
        except Exception as e:
            print(f"  ✗ Variant {count} failed: {e}")
            count += 1
    
    # Save variant catalog
    catalog_path = output_dir / "variant_catalog.json"
    catalog = {
        'timestamp': timestamp,
        'total_variants': len(variants),
        'valid_variants': len([v for v in variants if v['status'] == 'VALID']),
        'variants': variants,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2))
    
    print(f"\n{'='*80}")
    print(f"Generated {len(variants)} variants")
    print(f"Valid: {catalog['valid_variants']}")
    print(f"Catalog: {catalog_path}")
    print(f"{'='*80}")
    
    return True


if __name__ == "__main__":
    success = generate_variants()
    sys.exit(0 if success else 1)
