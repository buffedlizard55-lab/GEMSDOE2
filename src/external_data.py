"""
External data handling with verified official sources.
All downloads are from public domain USGS or CC-licensed INGENIOUS.

Sources:
- GeoDAWN: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
- QFaults: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
- 3DEP: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services
- INGENIOUS: https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483

This module provides:
- download helpers
- DEM feature engineering (slope, curvature, TPI, TRI, hillshade)
"""

import os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter, generic_filter

def compute_slope(dem, resolution=100):
    """Slope in degrees from DEM."""
    # gradient
    dy, dx = np.gradient(dem, resolution)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
    return slope

def compute_curvature(dem, resolution=100):
    """Simple curvature via second derivative."""
    dy, dx = np.gradient(dem, resolution)
    dyy, dyx = np.gradient(dy, resolution)
    dxy, dxx = np.gradient(dx, resolution)
    # profile curvature approximation
    curvature = dxx + dyy
    return curvature

def compute_tpi(dem, radius=3):
    """Topographic Position Index: dem - mean(dem in radius)."""
    mean = generic_filter(dem, np.mean, size=2*radius+1, mode='nearest')
    return dem - mean

def compute_tri(dem):
    """Terrain Ruggedness Index: mean absolute diff to 8 neighbors."""
    # Simple 3x3
    tri = np.zeros_like(dem)
    # Use generic filter
    def tri_func(window):
        center = window[4]
        return np.mean(np.abs(window - center))
    tri = generic_filter(dem, tri_func, size=3, mode='nearest')
    return tri

def compute_hillshade(dem, azimuth=315, altitude=45, resolution=100):
    """Hillshade."""
    azimuth = 360 - azimuth
    x, y = np.gradient(dem, resolution)
    slope = np.pi/2 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180
    altituderad = altitude * np.pi / 180
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos((azimuthrad - np.pi/2) - aspect)
    return 255 * (shaded + 1) / 2

def detrended_elevation(dem, sigma=50):
    """Detrend by subtracting Gaussian smoothed version."""
    # sigma in pixels
    smoothed = gaussian_filter(dem, sigma=sigma, mode='nearest')
    return dem - smoothed

def load_dem_to_grid(path, target_meta, shape, *, resolution=Resampling.average):
    """Read a DEM mosaic and aggregate/reproject it onto the feature grid.

    The competition supplies URLs to 1 m tiles rather than a ready-made 100 m raster.
    This function deliberately accepts a *local mosaic* created by
    ``scripts/download_dem_tiles.py``; it never downloads an undocumented source at
    training time. ``Resampling.average`` reduces the 1 m elevation to the 100 m
    feature cells while keeping the operation reproducible.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DEM mosaic not found: {path}")
    destination = np.full(tuple(shape), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=target_meta["transform"],
            dst_crs=target_meta["crs"],
            dst_nodata=np.nan,
            resampling=resolution,
        )
    destination[~np.isfinite(destination)] = np.nan
    return destination


def _dem_derivatives(dem, resolution=100):
    """Return five deterministic DEM derivatives while preserving the footprint mask."""
    dem = np.asarray(dem, dtype=np.float32)
    valid = np.isfinite(dem)
    if not valid.any():
        return np.full(dem.shape + (5,), np.nan, dtype=np.float32)
    # Derivatives and neighbourhood statistics cannot operate on NaN. Fill only while
    # calculating; restore the original invalid footprint after every derivative.
    filled = np.where(valid, dem, float(np.nanmedian(dem))).astype(np.float32)
    features = np.stack([
        compute_slope(filled, resolution=resolution),
        compute_curvature(filled, resolution=resolution),
        compute_tpi(filled, radius=3),
        compute_tri(filled),
        detrended_elevation(filled, sigma=50),
    ], axis=-1).astype(np.float32)
    features[~valid, :] = np.nan
    return features


def augment_with_dem_features(X, dem=None, dem_channel_index=None, resolution=100):
    """Append slope/curvature/TPI/TRI/detrended elevation to ``X``.

    ``dem`` should be a grid-aligned ``(H,W)`` array. For backwards compatibility,
    omitting it uses ``dem_channel_index`` (or channel 0), but production code should
    pass the reprojected 1 m mosaic explicitly rather than guessing from a feature band.
    """
    X = np.asarray(X)
    if X.ndim == 3:
        if dem is None:
            dem = X[:, :, dem_channel_index] if dem_channel_index is not None else X[:, :, 0]
        derivatives = _dem_derivatives(dem, resolution=resolution)
        if derivatives.shape[:2] != X.shape[:2]:
            raise ValueError(f"DEM grid {derivatives.shape[:2]} != feature grid {X.shape[:2]}")
        return np.concatenate([X.astype(np.float32, copy=False), derivatives], axis=-1)
    if X.ndim == 2:
        return _dem_derivatives(X if dem is None else dem, resolution=resolution)
    raise ValueError(f"expected X with shape (H,W,C) or DEM with shape (H,W), got {X.shape}")


def augment_from_dem_path(X, dem_path, target_meta, resolution=100):
    """Load a local DEM mosaic on the exact feature grid, then append five bands."""
    dem = load_dem_to_grid(dem_path, target_meta, X.shape[:2], resolution=Resampling.average)
    return augment_with_dem_features(X, dem=dem, resolution=resolution)

def download_instructions():
    print("""
Official verified download instructions:

1. GeoDAWN (USGS):
   - ScienceBase landing: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
   - Files: 22103_area1_tiffs.zip, 22103_area2_tiffs.zip etc.
   - Citation: Glen & Earney 2024, DOI https://doi.org/10.5066/P93LGLVQ
   - Use: mag, radiometric grids for extra features.

2. Quaternary Faults:
   - Interactive map: https://www.usgs.gov/programs/earthquake-hazards/faults
   - GIS zip: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 -> Qfaults_GIS.zip
   - DOI: https://doi.org/10.5066/P9BCVRCK

3. 3DEP 1m DEM:
   - About: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services
   - Downloader: https://apps.nationalmap.gov/downloader/
   - LidarExplorer: https://apps.nationalmap.gov/lidar-explorer/
   - For competition region (Nevada UTM 11N), search bounding box from training_features.tif meta.

4. INGENIOUS:
   - GDR: https://gdr.openei.org/submissions/1391
   - DOI: https://doi.org/10.15121/1881483
   - Contains: conductivity, strain, gravity, earthquake density etc. Already in training_features but useful for validation.

All are public domain or CC.
""")

if __name__ == "__main__":
    download_instructions()
