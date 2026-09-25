#!/usr/bin/env python3
"""Rasterise the independent fault catalogue onto the competition grid and remove what the labels
already contain.

This is `docs/DISCOVERY_PLAN.md` §3(b), step 2 of 3:

    1. `scripts/fetch_proxy_faults.py`  -> independent USGS SGMC polylines (GeoJSON)
    2. THIS SCRIPT                      -> a coded raster on the exact competition grid:
                                            0 = no proxy fault
                                            1 = proxy fault ALREADY within R of a training label
                                            2 = proxy fault with NO training label within R
                                                ("the catalogue does not contain it")
    3. `scripts/eval_proxy_catalogue.py` -> score a prediction against code 2 (and report the
                                            baselines that prove the measurement is not vacuous)

Why code 1 exists instead of a plain mask: the whole value of the proxy depends on how much of it
the training labels *already* cover.  If 95 % of SGMC faults sat within 300 m of a labelled fault,
the proxy would be a restatement of the catalogue and any score against it would be as misleading as
catalogue DTI.  That fraction is measured here, committed, and printed - it is the honest error bar
on every number derived from this file.  It is also *expected* to be well below 1: the labels come
from the INGENIOUS Quaternary compilation, while SGMC_Structure is mostly pre-Quaternary bedrock
structure from the state geologic maps.

Grid alignment is not assumed: the template raster (sample_submission.tif / labels.tif /
training_features.tif - they are all on the same 3292x3730 EPSG:32611 grid, verified in
data/evidence/rasters.json) defines the transform, and the script refuses to write an output whose
CRS/transform/shape differ from it.

USAGE
    python scripts/build_proxy_catalogue.py --proxy data/external/sgmc_proxy_faults.geojson \
        --template data/labels.tif --labels data/labels.tif \
        --out data/evidence/proxy/proxy_catalogue.tif --stats data/evidence/proxy/proxy_stats.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.warp import transform as warp_transform
from scipy import ndimage

CODE_NONE, CODE_NEAR, CODE_ONLY = 0, 1, 2
CHUNK = 2000


def load_geojson(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    feats = doc.get("features", doc if isinstance(doc, list) else [])
    if not feats:
        raise SystemExit(f"{path}: no features - refusing to build an empty proxy")
    return feats


def _lines(geom: dict) -> list[list[list[float]]]:
    if geom["type"] == "LineString":
        return [geom["coordinates"]]
    if geom["type"] == "MultiLineString":
        return list(geom["coordinates"])
    return []


def project_features(feats: list[dict], dst_crs) -> tuple[list[tuple[dict, int]], dict]:
    """GeoJSON (EPSG:4326) -> (geometry dicts in the template CRS, RuleID) + count summary."""
    src_crs = "EPSG:4326"
    out: list[tuple[dict, int]] = []
    dropped = Counter()
    for f in feats:
        lines = _lines(f.get("geometry") or {})
        if not lines:
            dropped["non_line_geometry"] += 1
            continue
        xs = [x for ln in lines for x, _ in ln]
        ys = [y for ln in lines for _, y in ln]
        px, py = warp_transform(src_crs, dst_crs, xs, ys)
        new_lines, i = [], 0
        for ln in lines:
            seg = [[px[i + k], py[i + k]] for k in range(len(ln))]
            i += len(ln)
            if len({(round(a, 3), round(b, 3)) for a, b in seg}) >= 2:   # >=2 distinct vertices
                new_lines.append(seg)
        if not new_lines:
            dropped["degenerate_after_reprojection"] += 1
            continue
        geom = ({"type": "LineString", "coordinates": new_lines[0]} if len(new_lines) == 1
                else {"type": "MultiLineString", "coordinates": new_lines})
        try:
            out.append((geom, int(f["properties"].get("RuleID"))))
        except (KeyError, TypeError, ValueError):
            dropped["missing_rule_id"] += 1
    return out, {"projected": len(out), "dropped": dict(dropped)}


def rasterize_features(shapes: list[tuple[dict, int]], shape, transform) -> np.ndarray:
    """OR together all polylines on the grid (chunked: rasterize() allocates a full-grid array)."""
    acc = np.zeros(shape, dtype="uint8")
    for i in range(0, len(shapes), CHUNK):
        tmp = rasterize([(g, 1) for g, _ in shapes[i:i + CHUNK]], out_shape=shape,
                        transform=transform, fill=0, dtype="uint8")
        np.maximum(acc, tmp, out=acc)
    return acc


def component_stats(mask: np.ndarray, px_km: float) -> dict:
    """Connected components of a boolean mask: count, areas, longest extent, in pixels and km."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return {"components": 0, "px": 0, "km": 0.0, "largest_px": 0, "largest_km": 0.0}
    counts = np.bincount(lab.ravel())[1:]
    objs = ndimage.find_objects(lab)
    largest = int(counts.max())
    li = int(counts.argmax())
    sl = objs[li]
    extent_px = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
    total = int(counts.sum())
    return {
        "components": int(n), "px": total, "km": round(total * px_km, 3),
        "largest_px": largest, "largest_km": round(largest * px_km, 3),
        "largest_extent_px": int(extent_px), "largest_extent_km": round(extent_px * px_km, 3),
        "median_px": float(np.median(counts)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy", default="data/external/sgmc_proxy_faults.geojson")
    ap.add_argument("--template", default="data/labels.tif", help="raster defining the grid to write on")
    ap.add_argument("--labels", default="data/labels.tif", help="known-fault raster (what the catalogue has)")
    ap.add_argument("--out", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--stats", default="data/evidence/proxy/proxy_stats.json")
    ap.add_argument("--fetch-meta", default="data/evidence/proxy/fetch_meta.json",
                    help="sidecar from fetch_proxy_faults.py, copied into the stats for provenance")
    ap.add_argument("--R", type=int, default=3, help="metric radius in pixels (300 m at 100 m/px)")
    ap.add_argument("--window", default=None,
                    help="optional r,c,size crop written next to --out as proxy_window.tif, so the "
                         "sandbox can test against a real slice without the competition rasters")
    a = ap.parse_args()

    proxy_path, tmpl_path, labels_path = Path(a.proxy), Path(a.template), Path(a.labels)
    feats = load_geojson(proxy_path)

    with rasterio.open(tmpl_path) as tm:
        shape, transform, crs = (tm.height, tm.width), tm.transform, tm.crs
        px_m = float(abs(tm.transform.a))
    with rasterio.open(labels_path) as lb:
        if (lb.height, lb.width) != shape or lb.crs != crs or lb.transform != transform:
            raise SystemExit(f"{labels_path} is not on the {tmpl_path} grid - refusing to mix grids")
        labels = lb.read(1).astype("float32")
        nod = lb.nodata
    in_footprint = np.isfinite(labels)
    if nod is not None:
        in_footprint &= labels != nod
    known = in_footprint & (labels > 0.5)

    print(f"grid {shape[1]}x{shape[0]} {crs} at {px_m} m; labels: {int(known.sum())} fault px "
          f"({100.0 * known.mean():.3f} % of the grid, {100.0 * known.sum() / max(in_footprint.sum(), 1):.3f} % "
          "of the footprint)")

    shapes, proj = project_features(feats, crs)
    print(f"projected {proj['projected']} features {proj['dropped'] or ''}")
    mask = rasterize_features(shapes, shape, transform)
    print(f"rasterised {int(mask.sum())} px of proxy fault")

    dist_to_known = ndimage.distance_transform_edt(~known)
    near = dist_to_known <= a.R
    coded = np.where(mask > 0, np.where(near, CODE_NEAR, CODE_ONLY), CODE_NONE).astype("uint8")
    coded[~in_footprint] = CODE_NONE          # never claim a proxy fault outside the data footprint

    only = coded == CODE_ONLY
    near_px, only_px = int((coded == CODE_NEAR).sum()), int(only.sum())
    px_km = px_m / 1000.0
    # The mask counts EVERY rasterised proxy pixel, including the 58 % of the grid that is outside
    # the data footprint (NaN).  Coverage is therefore reported twice, and the headline number is
    # the in-footprint one: an out-of-footprint proxy fault can never be predicted from data that
    # does not exist there, so including it would flatter the overlap.
    in_fp_px = near_px + only_px
    outside_px = int(mask.sum()) - in_fp_px

    # Per-class accounting: how much of each published class the labels already cover.
    # Rasterised ONE CLASS AT A TIME (there are ~30 distinct RuleIDs, not ~15,000 features):
    # rasterize() allocates a full-grid int32 array per call, so a per-feature loop would spend
    # minutes allocating ~49 MB buffers and would measure the allocator, not the geology.
    by_rule: dict[int, list[dict]] = defaultdict(list)
    for geom, rule in shapes:
        by_rule[rule].append(geom)
    per_rule_feat = {str(r): len(g) for r, g in by_rule.items()}
    per_rule_px: dict[str, int] = {}
    for rule, geoms in by_rule.items():
        drawn = rasterize([(g, rule) for g in geoms], out_shape=shape, transform=transform,
                          fill=0, dtype="int32")
        per_rule_px[str(rule)] = int(((drawn > 0) & (coded == CODE_ONLY)).sum())

    fetch_meta = json.loads(Path(a.fetch_meta).read_text()) if Path(a.fetch_meta).exists() else None
    # Two fetchers, two provenance layouts, ONE consumer.  scripts/fetch_proxy_faults.py (SGMC)
    # nests the class vocabulary under `query.rule_id_to_class`; scripts/fetch_qfaults.py (QFaults,
    # catalogue B for the cross-catalogue measurement) nests it under `classes.rule_id_to_class`.
    # Reading only the first silently produced unnamed per-class rows for the second catalogue, so
    # both are accepted and the key actually used is recorded in the stats.
    names, names_key = {}, None
    for key in ("query.rule_id_to_class", "classes.rule_id_to_class"):
        cur = fetch_meta or {}
        for part in key.split("."):
            cur = (cur or {}).get(part) if isinstance(cur, dict) else None
        if cur:
            names, names_key = cur, key
            break

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="uint8",
                   crs=crs, transform=transform, nodata=None, tiled=True,
                   blockxsize=256, blockysize=256, compress="lzw")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(coded, 1)
        dst.set_band_description(1, "0=none 1=proxy near label 2=proxy only")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    window_meta = None
    if a.window:
        r, c, size = (int(v) for v in a.window.split(","))
        crop = coded[r:r + size, c:c + size]
        if crop.shape != (size, size):
            raise SystemExit(f"--window {a.window} does not fit the grid {shape}")
        wp = out.with_name("proxy_window.tif")
        wprofile = dict(profile, height=size, width=size, compress="lzw")
        wprofile["transform"] = window_transform(Window(c, r, size, size), transform)
        with rasterio.open(wp, "w", **wprofile) as dst:
            dst.write(crop, 1)
        window_meta = {"path": str(wp), "row": r, "col": c, "size": size,
                       "codes": {int(k): int(v) for k, v in zip(*np.unique(crop, return_counts=True))}}

    stats = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/build_proxy_catalogue.py",
        "purpose": ("rasterised independent fault catalogue on the competition grid, split by whether "
                    "the training labels already contain it (docs/DISCOVERY_PLAN.md 3b)"),
        "inputs": {
            "proxy_geojson": str(proxy_path),
            "proxy_sha256": hashlib.sha256(proxy_path.read_bytes()).hexdigest(),
            "features_in": len(feats),
            "features_projected": proj["projected"],
            "dropped": proj["dropped"],
            "template": str(tmpl_path),
            "labels": str(labels_path),
            "class_vocabulary_key": names_key,
            "R_pixels": a.R,
            "R_meters": a.R * px_m,
        },
        "grid": {"width": shape[1], "height": shape[0], "crs": str(crs),
                 "transform": list(transform)[:6], "pixel_m": px_m,
                 "footprint_px": int(in_footprint.sum())},
        "labels": {"fault_px": int(known.sum()),
                   "fault_km": round(float(known.sum()) * px_km, 3)},
        "proxy": {
            "mask_px": int(mask.sum()),
            "near_label_px": near_px,
            "proxy_only_px": only_px,
            "near_label_km": round(near_px * px_km, 3),
            "proxy_only_km": round(only_px * px_km, 3),
            "catalogue_already_covers_fraction": round(near_px / max(in_fp_px, 1), 4),
            "catalogue_already_covers_fraction_including_outside_footprint":
                round(near_px / max(int(mask.sum()), 1), 4),
            "outside_footprint_px": outside_px,
            "proxy_only_components": component_stats(only, px_km),
            "per_rule_id": {
                rid: {"features": per_rule_feat.get(rid, 0), "proxy_only_px": px,
                      "class": names.get(rid, "")}
                for rid, px in sorted(per_rule_px.items(), key=lambda kv: -kv[1])
            },
        },
        "output": {"path": str(out), "sha256": sha, "bytes": out.stat().st_size,
                   "codes": {"0": "no proxy fault", "1": "proxy fault within R of a training label",
                             "2": "proxy fault with no training label within R"}},
        "window": window_meta,
        "source": (fetch_meta or {}).get("source"),
        "caveats": [
            "The proxy is an independent *published* catalogue, not the hidden expert labels: it is "
            "systematically easier in places (faults already drawn on state maps) and harder in "
            "others (bedrock faults with no Quaternary surface expression).",
            "Code 1 pixels are NOT removed from the proxy - they are kept so the overlap can be "
            "audited; scripts/eval_proxy_catalogue.py scores code 2 by default.",
            "Code 2 means 'no *labelled* fault within R', which is a statement about the label "
            "raster, not a statement that the fault is unmapped in the literature.",
        ],
    }
    sp = Path(a.stats)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")

    print(f"proxy: {int(mask.sum())} px rasterised ({outside_px} outside the data footprint) -> "
          f"in-footprint {in_fp_px}: {near_px} near-label, {only_px} proxy-only "
          f"({100.0 * near_px / max(in_fp_px, 1):.1f} % of the in-footprint proxy is already in "
          "the catalogue)")
    print(f"proxy-only: {component_stats(only, px_km)}")
    print(f"wrote {out} ({out.stat().st_size} bytes, sha256 {sha[:16]}...) and {sp}")
    if only_px == 0:
        print("WARNING: every proxy fault is already inside the catalogue - the proxy cannot "
              "discriminate anything. Do not use it for selection.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
