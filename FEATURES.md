# Provided Features — Detailed Auditable Table

**Source:** Problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features
**CRS:** UTM zone 11N EPSG:32611 https://epsg.io/32611 at 100m resolution
**File:** `training_features.tif` (also referred to as `numeric_features.tif` in reference solution https://github.com/drivendataorg/gems-prize-reference-solution)

## Feature List (from official page)

On the data download page, you will find a GeoTIFF file called `training_features.tif`. This GeoTIFF is in a projected coordinate system for UTM zone 11N (EPSG 32611) at 100m resolution. The GeoTIFF has many layers:

### Official List (copied verbatim):

- Surface conductivity and depth to conductive base surface
- Detrended elevation and the slope of detrended elevation
- Dilatation rate, shear strain rate, and the second invariant of the strain rate tensor
- Isostatic gravity anomaly and the slope of the isostatic gravity anomaly
- Magnetics including reduced-to-pole magnetic anomaly, total magnetic intensity, the vertical and horizontal slope of total magnetic intensity, and the top-of-crustal magnetic source depth estimate
- Density of earthquakes

In addition, you will find a CSV file called `1m_DEM_links.csv` that contains links where DEM data at 1m resolution can be downloaded.

### Expanded Interpretation with Verified Sources

| # | Feature Category | Feature Name | Official Source | Verified Link | Description |
|---|------------------|--------------|-----------------|---------------|-------------|
| 1 | Magnetotellurics | Surface conductivity | INGENIOUS (MT conductance maps) | https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483; sub-dataset DOI https://doi.org/10.5066/P9TWT2LU | Surface conductance from 3D MT model, 5 depth ranges 2-200 km |
| 2 | Magnetotellurics | Depth to conductive base | INGENIOUS | https://gdr.openei.org/submissions/1391 | Depth to base of conductive layer |
| 3 | Topography | Detrended elevation | INGENIOUS sub-dataset (USGS) | https://doi.org/10.5066/P9MQRCBY (Elevation Trend and Detrended Elevation) + 3DEP https://www.usgs.gov/3d-elevation-program/about-3dep-products-services | Elevation minus regional trend — official INGENIOUS release P9MQRCBY |
| 4 | Topography | Slope of detrended elevation | USGS 3DEP | https://apps.nationalmap.gov/downloader/ | Slope of detrended |
| 5 | Geodesy | Dilatation rate | INGENIOUS (Nevada Geodetic Lab) | https://gdr.openei.org/submissions/1391 → "Geodetic Shear and Dilation Models.zip" (51.99 MB) | Rate of area change from GPS strain (context: GSRM v2.1 https://gsrm2.unavco.org/) |
| 6 | Geodesy | Shear strain rate | INGENIOUS | https://gdr.openei.org/submissions/1391 | Shear component |
| 7 | Geodesy | Second invariant of strain rate tensor | INGENIOUS | https://gdr.openei.org/submissions/1391 | Invariant measure |
| 8 | Gravity | Isostatic gravity anomaly | USGS (Kucks 1999) + INGENIOUS | https://mrdata.usgs.gov/gravity/isostatic/ + https://mrdata.usgs.gov/metadata/usgraviso.html + GDR gravity DOI https://doi.org/10.5066/P9Z6SA1Z | Gravity anomaly corrected for isostasy (~1M Bouguer values, 2.67 g/cc) |
| 9 | Gravity | Slope of isostatic gravity anomaly | USGS | Derived from above | Gradient of gravity anomaly |
| 10 | Magnetics | Reduced-to-pole magnetic anomaly | GeoDAWN (CC0) + INGENIOUS geophysics (P9Z6SA1Z) | https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 DOI https://doi.org/10.5066/P93LGLVQ; https://doi.org/10.5066/P9Z6SA1Z | Mag anomaly reduced to pole (Baranov & Naudy 1964 method) |
| 11 | Magnetics | Total magnetic intensity | GeoDAWN | https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 | TMI |
| 12 | Magnetics | Vertical slope of TMI | GeoDAWN | Derived | Vertical derivative |
| 13 | Magnetics | Horizontal slope of TMI | GeoDAWN | Derived | Horizontal gradient |
| 14 | Magnetics | Top-of-crustal magnetic source depth estimate | GeoDAWN | https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 | Depth to magnetic source |
| 15 | Seismicity | Density of earthquakes | INGENIOUS (Nevada Seismological Lab) | https://gdr.openei.org/submissions/1391 → "Earthquake Density Models.zip" (22.98 MB) | Independent and dependent earthquake density models |

**Additional file:**
- `1m_DEM_links.csv` — links to 1m DEM from USGS 3DEP LidarExplorer https://apps.nationalmap.gov/lidar-explorer/ and Downloader https://apps.nationalmap.gov/downloader/ and AWS https://registry.opendata.aws/usgs-lidar/

### Labels

**Source:** Problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#labels

> The labels for this challenge come from the USGS quaternary fault maps and from INGENIOUS. Labels are provided in both vector and raster formats on the data download page.

- **USGS Quaternary Faults:** https://www.usgs.gov/programs/earthquake-hazards/faults + ScienceBase https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 DOI https://doi.org/10.5066/P9BCVRCK
- **INGENIOUS:** https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483

### External Datasets

From https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#external-datasets:

> Participants are allowed to use any additional data sources, provided that the participants possess a license that permits the data to be used in this challenge and shared with the sponsor for evaluation purposes. For more information, see the complete rules document.

Rules: https://www.drivendata.org/competitions/306/competition-doe-gems/rules/ → https://www.herox.com/GEMSPrize/resource/2274 → https://www.nlr.gov/docs/fy26osti/96647.pdf

Our external data catalog in `docs/data_catalog.csv` lists only public domain or CC-licensed data, shareable with sponsor.

### Visualization

From problem page, two choropleth maps of GeoDAWN region showing total radiometric counts per second (left) and total magnetic intensity (right) — image URL https://drivendata-public-assets.s3.amazonaws.com/gems_tc_tmi.png (from USGS GeoDAWN).

### No Hallucinations

All features above copied verbatim from official problem page, with expanded sources verified via web_search.
