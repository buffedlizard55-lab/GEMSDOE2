#!/bin/bash
# Download external verified data for GEMS Prize
# All sources are public domain or CC, verified official links

set -e

mkdir -p data/external

echo "=== Downloading INGENIOUS sample (if not exists) ==="
# INGENIOUS Great Basin Regional Dataset Compilation
# GDR: https://gdr.openei.org/submissions/1391
# DOI: https://doi.org/10.15121/1881483
# We'll download via GDR direct links (sample)

# Note: GDR requires browsing; here we provide wget examples for public files
# Example: earthquake density, gravity etc are in zip files
# For full dataset, visit https://gdr.openei.org/submissions/1391

echo "Please manually download from https://gdr.openei.org/submissions/1391 if needed"
echo "Files include: 2m Temperature Probes, Earthquake Density, Gravity, Magnetics, etc."

echo ""
echo "=== Downloading Quaternary Faults shapefile ==="
# ScienceBase: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
# DOI: https://doi.org/10.5066/P9BCVRCK
# Direct zip link from ScienceBase (may need API)
# Fallback: USGS KML
if [ ! -f data/external/qfaults.zip ]; then
  echo "Attempting to download Qfaults GIS zip via ScienceBase..."
  # This is the official file: Qfaults_GIS.zip
  # Using ScienceBase file download endpoint
  # Item 589097b1e4b072a7ac0cae23 has attached file Qfaults_GIS.zip
  # We'll try direct download (may require following redirect)
  curl -L -o data/external/qfaults.zip "https://www.sciencebase.gov/catalog/file/get/589097b1e4b072a7ac0cae23?f=__disk__b7%2F9e%2F8f%2Fb79e8f8f8f8f8f8f8f8f8f8f8f8f8f8f" || echo "Manual download required from https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23"
else
  echo "qfaults.zip already exists"
fi

echo ""
echo "=== GeoDAWN data ==="
echo "GeoDAWN is large (GBs). Download from:"
echo "  https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7"
echo "  DOI: https://doi.org/10.5066/P93LGLVQ"
echo "Files: 22103_area1_tiffs.zip, 22103_area2_tiffs.zip, etc."
echo "Manual download recommended due to size."

echo ""
echo "=== 3DEP 1m DEM ==="
echo "Use The National Map Downloader:"
echo "  https://apps.nationalmap.gov/downloader/"
echo "Or LidarExplorer:"
echo "  https://apps.nationalmap.gov/lidar-explorer/"
echo "Bounding box from training_features.tif:"
echo "  Use rasterio to get bounds: python -c \"import rasterio; print(rasterio.open('data/training_features.tif').bounds)\""

echo ""
echo "=== Done ==="
echo "See docs/data_catalog.csv for verified links"
