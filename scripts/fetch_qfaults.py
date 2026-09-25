#!/usr/bin/env python3
"""Fetch a SECOND independent fault catalogue for the GeoDAWN footprint: the USGS Quaternary
Fault and Fold Database (QFaults), the "catalogue B" of the cross-catalogue transfer test.

WHY A SECOND CATALOGUE
----------------------
The repository already has one independent fault set - the USGS State Geologic Map Compilation
(SGMC) structure polylines, rasterised into `data/evidence/proxy/proxy_catalogue.tif` and used as
the new-fault-like population (`scripts/fetch_proxy_faults.py`, DOI 10.3133/ds1052).  It is the
only local stand-in for the scored faults, and every emission-policy number rests on it.  But a
single surrogate cannot tell us whether a policy travels or whether it merely fits THAT
catalogue's mapping style.  EXECUTIVE_SUMMARY.md §11 item 1 names the missing measurement:

    "Emit catalogue A (SGMC), score vs independent catalogue B (QFaults), report transfer as a
     prior on hidden-expert-set recall."

The proxy population cannot measure that idea by construction: the proxy IS catalogue A, so a
catalogue-copy submission scores 0.0 there (the acceptance control in
`scripts/eval_proxy_catalogue.py`).  A second, independently compiled catalogue can.

SOURCE (verified live 2026-09-18 from this repository's own HTTP requests)
--------------------------------------------------------------------------
Publication   U.S. Geological Survey, 2020, Quaternary fault and fold database for the nation:
              https://doi.org/10.5066/P9BCVRCK   (public domain, USGS)
ScienceBase   https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
Portal        https://www.usgs.gov/programs/earthquake-hazards/faults
Service       https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer
              currentVersion 11.5, mapName "2020 Release", capabilities "Map,Query,Data",
              supportedQueryFormats "JSON, geoJSON, PBF", maxRecordCount 2000
Layer         21 "National Database" (Feature Layer, esriGeometryPolyline).  Layers 1-20 are the
              per-state views (11 Nevada, 4 California, 16 Utah); layer 21 covers the whole
              nation in one query, so no state-boundary logic is needed and no feature can be
              missed at a state line.
Footprint     EPSG:4326 envelope of the competition grid, computed at run time from
              data/labels.tif with rasterio.warp.transform_bounds - never typed.  For the
              shipped grid (EPSG:32611, 243350/4135550 - 572550/4508550) that is
              -120.0372, 37.3312, -116.1409, 40.7279, which the service reported as
              **14,482 intersecting features** on 2026-09-18 (returnCountOnly=true).

WHY QFAULTS IS AN INDEPENDENT "B" AND WHAT IT IS NOT
----------------------------------------------------
* Independent of A: SGMC_Structure is mostly pre-Quaternary bedrock structure digitised from the
  state geologic maps (Crafford 2007 for Nevada); QFaults is Quaternary-age, seismic-hazard
  mapped, and compiled by different authors from different sources.  Their intersection is a
  measurement, not a tautology.
* NOT independent of the training labels: rules §3.3 says the labels come from the INGENIOUS
  Great Basin compilation, which itself distributes Quaternary fault layers (GDR submission 1391,
  DOI 10.15121/1881483).  So `scripts/build_proxy_catalogue.py` is run on this catalogue too and
  the scored truth is only its code-2 pixels - the QFaults traces with NO training label within
  R = 3 px.  That overlap fraction is measured and committed, never assumed: if QFaults and the
  labels turned out to be the same lines, the code-2 population would be empty and this script's
  downstream measurement would refuse to run.

WHAT IT DOES
------------
1. reads the layer metadata (name, geometry type, maxRecordCount, fields, and the renderer's
   `uniqueValueInfos` class vocabulary) so nothing about the service is hand-copied;
2. asks the service how many features intersect the footprint (`returnCountOnly=true`);
3. pages the query with resultOffset/resultRecordCount in EPSG:4326 GeoJSON until the fetched
   count EQUALS the service's own count - a silent truncation fails the run instead of shipping
   a partial catalogue;
4. assigns each feature an integer `RuleID` = its index in the service's own sorted class
   vocabulary (the `symbology` attribute, e.g. "latest Quaternary Moderately Constrained"), which
   lets `scripts/build_proxy_catalogue.py` rasterise and account for it per class unchanged;
5. writes the GeoJSON plus a provenance sidecar (every request URL, the counts, the class map,
   the sha256 of both outputs, and live link checks of the DOI/ScienceBase/portal pages).

Network note: earthquake.usgs.gov is not on the development sandbox's egress allowlist
(verified 2026-09-18: curl exit 35), so this runs on a GitHub-hosted runner via
`.github/workflows/cross-catalogue.yml`, exactly like the SGMC fetch does.

USAGE
    python scripts/fetch_qfaults.py --bbox-source data/labels.tif \
        --out data/external/qfaults_footprint.geojson \
        --meta data/evidence/xcat/fetch_meta.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- provenance (every value below was opened on 2026-09-18 or is already catalogued) -------
QFAULTS_DOI = "https://doi.org/10.5066/P9BCVRCK"
QFAULTS_SCIENCEBASE = "https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23"
QFAULTS_PORTAL = "https://www.usgs.gov/programs/earthquake-hazards/faults"
QFAULTS_CITATION = ("U.S. Geological Survey, 2020, Quaternary fault and fold database for the "
                    "nation: U.S. Geological Survey data release, https://doi.org/10.5066/P9BCVRCK")
SERVICE = "https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer"
LAYER = 21                       # "National Database" (polyline); see the module docstring
LICENSE = "Public Domain (U.S. Government work, USGS)"
USER_AGENT = "gemsdoe-xcat-fetch/1.0 (+https://github.com/buffedlizard55-lab/GEMSDOE)"
RULE_ID_FIELD = "symbology"      # the layer's own renderer field: age + location certainty
FALLBACK_CLASS_FIELDS = ("age", "linetype")


def http_json(url: str, timeout: int = 300, attempts: int = 4) -> dict:
    """GET a URL and parse JSON, retrying on transient failures and reporting the real error."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:       # noqa: S310 - fixed https
                raw = r.read()
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:                                          # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                wait = 5 * attempt
                print(f"  retry {attempt}/{attempts - 1} in {wait}s: {last}")
                time.sleep(wait)
    raise RuntimeError(f"failed after {attempts} attempts: {url}: {last}")


def head_check(urls: dict) -> dict:
    """Record the real HTTP status of each provenance link (never claim a link is OK untested)."""
    out = {}
    for label, url in urls.items():
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:            # noqa: S310
                out[url] = f"OK_{r.status} ({label})"
        except Exception as exc:                                          # noqa: BLE001
            out[url] = f"UNREACHABLE: {type(exc).__name__}: {exc}"
    for url, status in out.items():
        print(f"  link {status:<28} {url}")
    return out


def bbox_from_raster(path: Path, pad_deg: float = 0.0) -> tuple:
    """EPSG:4326 envelope of a competition raster, computed (not typed) at run time."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as src:
        b = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        grid = dict(path=str(path), crs=str(src.crs), width=src.width, height=src.height,
                    bounds=[float(v) for v in src.bounds])
    west, south, east, north = (float(b[0]) - pad_deg, float(b[1]) - pad_deg,
                                float(b[2]) + pad_deg, float(b[3]) + pad_deg)
    return (west, south, east, north), grid


def build_query(service: str, layer: int, bbox: tuple, offset: int, count: int,
                count_only: bool = False, out_fields: str = "*") -> str:
    params = {
        "where": "1=1",
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "f": "json" if count_only else "geojson",
    }
    if count_only:
        params["returnCountOnly"] = "true"
    else:
        params.update(outFields=out_fields, outSR="4326",
                      resultRecordCount=str(count), resultOffset=str(offset))
    return f"{service.rstrip('/')}/{layer}/query?" + urllib.parse.urlencode(params)


def class_vocabulary(layer_meta: dict) -> list:
    """The service's OWN class vocabulary, from its renderer - never a hand-copied list.

    QFaults symbology values are "<age> <location certainty>" strings, e.g. "latest Quaternary
    Moderately Constrained" (read from the layer's drawingInfo.renderer.uniqueValueInfos on
    2026-09-18).  A feature whose value is absent from the vocabulary still gets a stable id:
    it is appended in first-seen order and recorded, so nothing is silently dropped.
    """
    vocab: list = []
    renderer = ((layer_meta.get("drawingInfo") or {}).get("renderer") or {})
    for info in renderer.get("uniqueValueInfos", []) or []:
        v = info.get("value")
        if isinstance(v, str) and v not in vocab:
            vocab.append(v)
    for group in renderer.get("uniqueValueGroups", []) or []:
        for cls in group.get("classes", []) or []:
            for row in cls.get("values", []) or []:
                for v in row:
                    if isinstance(v, str) and v not in vocab:
                        vocab.append(v)
    return sorted(vocab)


def class_id_map(vocab: list, extra: dict) -> dict:
    """class value -> integer RuleID.  `extra` holds values the service vocabulary did not list."""
    m = {v: i + 1 for i, v in enumerate(vocab)}
    m.update({k: int(v) for k, v in extra.items()})
    return m


def feature_class(props: dict, vocab: list, extra: dict) -> int:
    """Integer RuleID for build_proxy_catalogue.py, derived from the service's own vocabulary.

    A value the renderer did not list is appended with the next free id and recorded in `extra`,
    so an unlisted class is measured and reported rather than silently dropped.
    """
    value = props.get(RULE_ID_FIELD)
    if not isinstance(value, str) or not value:
        value = " ".join(str(props.get(f, "")).strip() for f in FALLBACK_CLASS_FIELDS).strip()
    if not value:
        value = "<missing class>"
    ids = class_id_map(vocab, extra)
    if value not in ids:
        extra[value] = len(vocab) + len(extra) + 1
        ids = class_id_map(vocab, extra)
    return int(ids[value])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox-source", default="data/labels.tif",
                    help="competition raster whose EPSG:4326 envelope defines the query bbox")
    ap.add_argument("--service", default=SERVICE)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--page-size", type=int, default=1000,
                    help="features per request; capped by the service's own maxRecordCount")
    ap.add_argument("--pad-deg", type=float, default=0.0,
                    help="grow the bbox (a fault crossing the edge is still real geology)")
    ap.add_argument("--out", default="data/external/qfaults_footprint.geojson")
    ap.add_argument("--meta", default="data/evidence/xcat/fetch_meta.json")
    ap.add_argument("--max-features", type=int, default=400000, help="hard stop against runaway paging")
    a = ap.parse_args(argv)

    t0 = time.time()
    bbox, grid = bbox_from_raster(Path(a.bbox_source), pad_deg=a.pad_deg)
    print(f"footprint bbox (EPSG:4326) from {grid['path']} [{grid['crs']} {grid['width']}x{grid['height']}]: "
          f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}")

    layer_meta = http_json(f"{a.service.rstrip('/')}/{a.layer}?f=json")
    layer_name = layer_meta.get("name")
    geom_type = layer_meta.get("geometryType")
    max_records = int(layer_meta.get("maxRecordCount") or 1000)
    fields = [f.get("name") for f in (layer_meta.get("fields") or [])]
    if geom_type != "esriGeometryPolyline":
        raise SystemExit(f"layer {a.layer} is {geom_type}, expected esriGeometryPolyline - "
                         "rasterising anything else would not be a fault trace")
    page = max(1, min(a.page_size, max_records))
    vocab = class_vocabulary(layer_meta)
    print(f"layer {a.layer} '{layer_name}' {geom_type}; maxRecordCount={max_records} -> page={page}; "
          f"{len(fields)} fields; {len(vocab)} classes in the service's own vocabulary")

    count_url = build_query(a.service, a.layer, bbox, 0, 0, count_only=True)
    expected = int(http_json(count_url).get("count", -1))
    if expected <= 0:
        raise SystemExit(f"the service reports {expected} features in the footprint - refusing to "
                         "write an empty catalogue (check the bbox source and the layer id)")
    print(f"service reports {expected:,} features intersecting the footprint")

    feats: list = []
    urls: list = []
    offset = 0
    while True:
        url = build_query(a.service, a.layer, bbox, offset, page)
        urls.append(url)
        doc = http_json(url)
        batch = doc.get("features") or []
        feats.extend(batch)
        print(f"  offset {offset:>7}: +{len(batch)} (total {len(feats):,}/{expected:,})"
              f"{' exceededTransferLimit' if doc.get('exceededTransferLimit') else ''}")
        if not batch:
            break
        offset += len(batch)
        if offset >= expected:
            break
        if not doc.get("exceededTransferLimit") and len(batch) < page:
            break
        if len(feats) > a.max_features:
            raise SystemExit(f"fetched {len(feats)} > --max-features {a.max_features}: stopping, "
                             "the query is not converging")
        time.sleep(0.2)

    # INTEGRITY GATE: a silently truncated catalogue would understate transfer and look like a
    # negative result.  The service told us how many features exist; we must have all of them.
    if len(feats) != expected:
        raise SystemExit(f"fetched {len(feats):,} features but the service reports {expected:,} for "
                         "this bbox - refusing to write a partial catalogue")

    extra_classes: dict = {}
    per_class: dict = {}
    n_lines = 0
    for f in feats:
        props = f.get("properties") or {}
        rule = feature_class(props, vocab, extra_classes)
        props["RuleID"] = rule
        props["RuleClass"] = props.get(RULE_ID_FIELD) or " ".join(
            str(props.get(k, "")).strip() for k in FALLBACK_CLASS_FIELDS).strip() or "<missing class>"
        f["properties"] = props
        per_class[str(rule)] = per_class.get(str(rule), 0) + 1
        g = f.get("geometry") or {}
        if g.get("type") in ("LineString", "MultiLineString"):
            n_lines += 1
    if n_lines == 0:
        raise SystemExit("no line geometries in the response - nothing to rasterise")

    class_map = {str(v): k for k, v in sorted(class_id_map(vocab, extra_classes).items())}
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"type": "FeatureCollection",
           "features": feats,
           "properties": {
               "source": "USGS Quaternary Fault and Fold Database (QFaults), 2020 release",
               "citation": QFAULTS_CITATION,
               "doi": QFAULTS_DOI,
               "sciencebase": QFAULTS_SCIENCEBASE,
               "license": LICENSE,
               "service": a.service,
               "layer": a.layer,
               "layer_name": layer_name,
               "role": "catalogue B for the cross-catalogue transfer measurement "
                       "(scripts/measure_cross_catalogue_transfer.py)",
           }}
    payload = json.dumps(doc)
    out_path.write_text(payload, encoding="utf-8")
    out_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    links = head_check({"doi": QFAULTS_DOI, "sciencebase": QFAULTS_SCIENCEBASE,
                        "portal": QFAULTS_PORTAL, "service": f"{a.service}?f=json"})

    meta = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/fetch_qfaults.py",
        purpose=("second independent fault catalogue (QFaults) for the cross-catalogue transfer "
                 "measurement: emit catalogue A (SGMC), score against catalogue B (QFaults)"),
        source=dict(citation=QFAULTS_CITATION, doi=QFAULTS_DOI, sciencebase=QFAULTS_SCIENCEBASE,
                    portal=QFAULTS_PORTAL, license=LICENSE,
                    service=a.service, layer=a.layer, layer_name=layer_name,
                    geometry_type=geom_type, max_record_count=max_records,
                    page_size_used=page, fields=fields),
        bbox=dict(envelope_4326=[round(v, 6) for v in bbox], pad_deg=a.pad_deg, source=grid),
        counts=dict(service_reported=expected, fetched=len(feats), line_geometries=n_lines,
                    integrity_gate="fetched == service_reported (else the run fails)"),
        classes=dict(vocabulary_field=RULE_ID_FIELD, vocabulary_from="layer renderer uniqueValueInfos",
                     n_classes_in_vocabulary=len(vocab),
                     n_classes_appended_at_runtime=len(extra_classes),
                     rule_id_to_class=class_map, features_per_class=per_class),
        # Every request this run made, by kind.  `pages` alone was ambiguous (it counted neither
        # the metadata read nor the count query), so the three kinds are recorded separately and
        # the total is stated: an auditor should be able to replay this fetch from the sidecar.
        requests=dict(layer_metadata=f"{a.service.rstrip('/')}/{a.layer}?f=json",
                      count_only=count_url, pages=len(urls),
                      first_page=urls[0] if urls else None,
                      last_page=urls[-1] if urls else None, all_page_urls=urls,
                      total_requests=2 + len(urls)),
        link_checks=links,
        output=dict(path=str(out_path), sha256=out_sha, bytes=len(payload.encode("utf-8")),
                    features=len(feats)),
        elapsed_s=round(time.time() - t0, 1),
        next_steps=[
            "python scripts/build_proxy_catalogue.py --proxy " + str(out_path) +
            " --template data/labels.tif --labels data/labels.tif "
            "--out data/evidence/xcat/qfaults_catalogue.tif "
            "--stats data/evidence/xcat/qfaults_stats.json "
            "--fetch-meta " + str(Path(a.meta)),
            "python scripts/measure_cross_catalogue_transfer.py",
        ],
    )
    meta_path = Path(a.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {out_path} ({len(payload):,} bytes, {len(feats):,} features, sha256 {out_sha[:16]}…)")
    print(f"wrote {meta_path}")
    print(f"classes seen: {len(per_class)}; integrity: fetched {len(feats):,} == reported {expected:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
