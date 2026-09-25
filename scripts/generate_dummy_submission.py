"""
Generate a dummy submission that matches required format, for testing without training data.
Creates a GeoTIFF with same CRS, resolution, bounds as sample or training_features if available, otherwise creates a synthetic one based on competition spec.

Verified format from: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
- EPSG:32611, 100m, single band float32 [0,1], same bounds
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import argparse

def generate_dummy(out_path, sample_path=None, train_path=None, width=1000, height=1000,
                   seed=42):
    # A dummy submission is still a pipeline artefact: seed the RNG so two runs produce the
    # same file (rules §3.2/§3.5 reproducibility).  Pass --seed -1 for a non-reproducible draw.
    # (Seed added 2026-09-16, session 6; previously every invocation differed silently.)
    if seed is not None and int(seed) >= 0:
        np.random.seed(int(seed))
    # Try to use sample or train as template
    template = None
    if sample_path and Path(sample_path).exists():
        template = sample_path
    elif train_path and Path(train_path).exists():
        template = train_path

    if template:
        print(f"Using template {template}")
        with rasterio.open(template) as src:
            meta = src.meta.copy()
            meta.update(dtype='float32', count=1, nodata=None, compress='lzw')
            # Create dummy data: zeros with some random faults
            # For demo, create random low prob with some lines
            data = np.random.rand(src.height, src.width).astype(np.float32) * 0.1
            # Add some fake fault lines
            for _ in range(20):
                x = np.random.randint(0, src.width)
                y = np.random.randint(0, src.height)
                length = np.random.randint(50, 200)
                # vertical or horizontal line
                if np.random.rand() > 0.5:
                    data[y:y+length, x] = np.random.rand() * 0.5 + 0.5
                else:
                    data[y, x:x+length] = np.random.rand() * 0.5 + 0.5
            data = np.clip(data, 0, 1).astype(np.float32)
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(data, 1)
            print(f"Generated dummy submission at {out_path} from template, shape {data.shape}")
    else:
        print("No template found, creating synthetic GeoTIFF with EPSG:32611, 100m")
        # Create synthetic: UTM 11N, approximate Nevada bounds
        # Use from_origin: west, north, xsize, ysize
        # Approx bounds for GeoDAWN region: need to estimate
        # We'll use 0,0 origin for demo, but note real submission must match training_features bounds
        transform = from_origin(300000, 4500000, 100, 100)  # 100m resolution
        crs = rasterio.crs.CRS.from_epsg(32611)
        data = np.random.rand(height, width).astype(np.float32) * 0.1
        # Add fake faults
        for _ in range(20):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            length = np.random.randint(50, 200)
            if np.random.rand() > 0.5:
                data[y:y+length, x] = 0.8
            else:
                data[y, x:x+length] = 0.8
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        print(f"Generated synthetic dummy at {out_path}, shape {height}x{width}, CRS EPSG:32611, 100m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_dummy.tif")
    parser.add_argument("--sample", default="data/sample_submission.tif")
    parser.add_argument("--train", default="data/training_features.tif")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the fake faults (default 42; -1 = unseeded)")
    args = parser.parse_args()
    generate_dummy(args.out, args.sample, args.train, args.width, args.height, seed=args.seed)
