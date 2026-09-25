#!/usr/bin/env python3
"""Pre-flight validation of the competition data in data/ - run right after downloading.

Checks the four things the official problem page states as requirements, so a bad
download is caught before hours of GPU training:
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
    - projected CRS UTM 11N (EPSG:32611)
    - 100 m resolution
    - identical bounds across features / labels / submission template
    - labels = positive pixels where a fault is present; template float32 in [0,1]
plus band inventory (the reference solution reads per-band `description` + `data_category`
tags, so a stack without them is worth knowing about before training).

Exit code is non-zero on failure so it can gate a pipeline (`prepare_data.py && train`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import (FEATURE_NAME_CANDIDATES, LABEL_NAME_CANDIDATES,  # noqa: E402
                         SAMPLE_NAME_CANDIDATES, resolve_path)

EXPECTED_CRS_EPSG = 32611
EXPECTED_RES = 100.0


def _describe(path: str) -> dict:
    with rasterio.open(path) as src:
        info = dict(path=path, width=src.width, height=src.height, count=src.count,
                    crs=str(src.crs), epsg=(src.crs.to_epsg() if src.crs else None),
                    res=(float(src.res[0]), float(src.res[1])), dtypes=src.dtypes,
                    nodata=src.nodata, bounds=tuple(round(b, 2) for b in src.bounds),
                    tags=[dict(src.tags(i) or {}) for i in range(1, min(src.count, 32) + 1)])
        a = src.read(1).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        info.update(minv=float(np.nanmin(a)), maxv=float(np.nanmax(a)),
                    nan_frac=float(np.isnan(a).mean()))
    return info


def check() -> int:
    problems, notes = [], []
    found = {}
    for kind, cands in (("features", FEATURE_NAME_CANDIDATES), ("labels", LABEL_NAME_CANDIDATES),
                        ("sample", SAMPLE_NAME_CANDIDATES)):
        try:
            found[kind] = resolve_path(None, cands)
        except FileNotFoundError:
            problems.append(f"missing {kind}: none of {list(cands)} under data/ "
                            f"- run `bash scripts/download_competition_data.sh`")
            continue
        info = _describe(found[kind])
        found[kind + "_info"] = info
        print(f"[{kind}] {info['path']}")
        print(f"    {info['width']}x{info['height']} count={info['count']} crs={info['crs']} "
              f"res={info['res']} dtype={info['dtypes'][0]} nodata={info['nodata']}")
        print(f"    bounds={info['bounds']}  band1 range=[{info['minv']:.4g},{info['maxv']:.4g}] "
              f"nan%={100 * info['nan_frac']:.2f}")
        if info["epsg"] not in (EXPECTED_CRS_EPSG, None):
            problems.append(f"{kind}: CRS EPSG:{info['epsg']} != {EXPECTED_CRS_EPSG}")
        elif info["epsg"] is None:
            notes.append(f"{kind}: CRS is not EPSG-codeable ({info['crs']}) - verify it is UTM 11N")
        if max(info["res"]) != EXPECTED_RES or min(info["res"]) != EXPECTED_RES:
            problems.append(f"{kind}: resolution {info['res']} != (100.0, 100.0)")
        if abs(info["maxv"]) > 1e30:
            notes.append(f"{kind}: |max| > 1e30 -> likely unreplaced nodata; loader maps <-1e30 to NaN")

    if "features_info" in found and "labels_info" in found:
        f, l = found["features_info"], found["labels_info"]
        if (f["width"], f["height"]) != (l["width"], l["height"]):
            problems.append(f"grid mismatch: features {f['width']}x{f['height']} vs labels "
                            f"{l['width']}x{l['height']} (spec: same bounds)")
        if f["bounds"] != l["bounds"]:
            problems.append(f"bounds mismatch: features {f['bounds']} vs labels {l['bounds']}")
        if l["count"] != 1:
            notes.append(f"labels raster has {l['count']} bands; training uses band 1")
        if not (0.0 <= l["minv"] and l["maxv"] <= 1.0 + 1e-6):
            notes.append(f"labels band 1 range [{l['minv']}, {l['maxv']}] - code thresholds y>0.5 "
                         "as fault (reference solution uses y<1 -> 0)")
        print(f"[bands] {len(f['tags'])} feature band tag sets; "
              f"first: {f['tags'][0] if f['tags'] else '{}'}")
        if f["tags"] and not any("description" in t for t in f["tags"]):
            notes.append("feature bands carry no 'description' tag (the reference solution prints "
                         "description/data_category) - band order will be inferred, verify by hand")
    if "sample_info" in found:
        s, f = found["sample_info"], found.get("features_info")
        if s["count"] != 1:
            problems.append(f"sample submission has {s['count']} bands, expected 1")
        if not (-1e-6 <= s["minv"] and s["maxv"] <= 1.0 + 1e-6):
            problems.append(f"sample submission values outside [0,1]: [{s['minv']}, {s['maxv']}]")
        if f and (s["width"], s["height"]) != (f["width"], f["height"]):
            problems.append(f"sample submission grid {s['width']}x{s['height']} != features "
                            f"{f['width']}x{f['height']}")

    for n in notes:
        print(f"NOTE: {n}")
    if problems:
        print("\nFAILED:")
        for pr in problems:
            print("  -", pr)
        return 1
    print("\nOK: data/ passes every pre-flight check; next: "
          "python -m src.train --config configs/config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
