#!/usr/bin/env python3
"""Fetch an INDEPENDENT fault catalogue for the GeoDAWN footprint (the "proxy catalogue").

WHY THIS EXISTS
---------------
The official rules put the scored population and the training labels in *different universes*
(verified verbatim in `scripts/verify_rules_quotes.py`, `data/evidence/rules_quotes.json`):

    §1.1  "In Phase 1, submissions will be evaluated against a privately withheld subset of the
           original new fault dataset compiled by expert reviewers."
    §1.1  "Submissions will be reevaluated against the full, revised new fault dataset ..."
    §3.3  "The training labels contain existing fault data at 100-m resolution ... obtained from
           the INGENIOUS project's Great Basin Regional Dataset Compilation."

So every DTI computed against `labels.tif` measures faults the *catalogue already contains*, while
the prize is won on faults it *does not contain*.  `docs/DISCOVERY_PLAN.md` §3(b) names the missing
instrument: a set of real, mapped faults that are absent from the training labels, used as a stand-in
truth set for model/submission selection.

This script fetches exactly that, from a source that is independent of the INGENIOUS compilation:

    USGS State Geologic Map Compilation (SGMC) of the conterminous United States
    Horton, J.D., San Juan, C.A., and Stoeser, D.B., 2017, USGS Data Series 1052
    DOI 10.3133/ds1052  (also released as DOI 10.5066/F7WH2N65)
    Feature class SGMC_Structure (polyline; "linear geologic features (mostly faults)")
    Nevada content: Crafford, A.E.J., 2007, Geologic map of Nevada, USGS Data Series 249.

It is published by USGS as an ArcGIS FeatureServer (public, anonymous, no key):
    https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/
        SB_5888bf4fe4b05ccb964bab9d_USGS_SGMC_feature/FeatureServer/1

WHY THIS SOURCE AND NOT ANOTHER
-------------------------------
* Independent of the labels' provenance (the INGENIOUS compilation), so the intersection with
  `labels.tif` is a *measurement*, not a tautology.  Whatever the labels already cover is removed
  downstream by `scripts/build_proxy_catalogue.py` — this script only fetches.
* Official, public domain (USGS), no credentials, and every feature carries its source citation
  (`REFERENCE`, `NGMDB1`, `DIGITAL_URL`), which is preserved in the output for manual review.
* Mostly pre-Quaternary bedrock structure: state geologic maps map faults that the Quaternary
  Fault and Fold Database, by construction, excludes — the same "missing from the catalogue"
  class as the scored faults.

WHAT IT DOES
------------
1. reads the layer metadata and builds the query from the service's own coded-value domain, so the
   fault classes are not a hand-copied list: every RuleID whose published name contains "fault"
   (plus "shear zone"/"lineament" optionally) is selected, and the mapping is recorded;
2. pages through the query for the raster footprint in EPSG:4326, EPSG:4326 out (RFC 7946 GeoJSON);
3. writes GeoJSON + a provenance sidecar (request URLs, counts per RuleID, sha256 of the output).

Network note: this must run where the host is reachable.  The development sandbox's egress
allowlist does not include services.arcgis.com, so this script runs on a GitHub-hosted runner
(`.github/workflows/proxy-eval.yml`) exactly like the competition-data data bridge does.

USAGE
    python scripts/fetch_proxy_faults.py --bbox-source data/labels.tif \
        --out data/external/sgmc_proxy_faults.geojson --meta data/evidence/proxy/fetch_meta.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Official publication (recorded in every report so the service URL can never become an unlabelled
# mirror: the DOI is the citable object, the FeatureServer is where the bytes came from).
SGMC_DOI = "https://doi.org/10.3133/ds1052"
SGMC_DATA_DOI = "https://doi.org/10.5066/F7WH2N65"
SGMC_METHODS = "https://pubs.usgs.gov/publication/ds1052"
# Every URL below was opened on 2026-09-16 (see data/evidence/independent_verification.json):
# the publication page lists these appendices verbatim, and the data DOI resolves to ScienceBase
# item 5888bf4fe4b05ccb964bab9d - the same item id embedded in the FeatureServer name below,
# which is how the service is tied to the official data release rather than to a look-alike.
SGMC_REPORT_PDF = "https://pubs.usgs.gov/ds/1052/ds1052.pdf"
SGMC_ATTRIBUTE_DICTIONARY = "https://pubs.usgs.gov/ds/1052/ds20171052_appendix5.pdf"
SGMC_ALL_FIELD_DEFINITIONS = "https://pubs.usgs.gov/ds/1052/ds20171052_appendix2_v1_1.pdf"
# USGS recommends this 2026 GeMS-format replacement for new work; the FeatureServer queried here
# belongs to the 2017 release (ver. 1.1) and is kept because it is the one that is queryable.
SGMC_2026_GEMS_DOI = "https://doi.org/10.5066/P1A3DQZK"
SERVICE = ("https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/"
           "SB_5888bf4fe4b05ccb964bab9d_USGS_SGMC_feature/FeatureServer")
LAYER = 1                       # SGMC_Structure
USER_AGENT = "gemsdoe-proxy-fetch/1.0 (+https://github.com/buffedlizard55-lab/GEMSDOE)"


def http_json(url: str, timeout: int = 180, attempts: int = 4) -> dict:
    """GET a JSON document with retries.  Raises on HTTP error or an ArcGIS 'error' payload."""
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 - fixed https
                payload = json.loads(r.read().decode("utf-8"))
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(f"service error: {payload['error']}")
            return payload
        except Exception as exc:                                          # noqa: BLE001
            last = exc
            if i < attempts - 1:
                time.sleep(2 ** i * 3)
    raise RuntimeError(f"failed after {attempts} attempts: {url}: {last}")


def resolve_doi(doi_url: str) -> str:
    """Follow a DOI to its landing page and return the final URL (no credentials, no API key)."""
    req = urllib.request.Request(doi_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:                    # noqa: S310 - fixed https
        return r.geturl()


def service_identity(service: str) -> dict:
    """Prove the FeatureServer belongs to the official data release, by measurement.

    The service is named `SB_<sciencebase-item-id>_USGS_SGMC_feature`; the data release DOI resolves
    to that ScienceBase item.  Both facts are observed here, not asserted: if the DOI ever resolves
    somewhere that does not carry the item id, the proxy stops being traceable to USGS and this
    function says so, loudly.

    MEASURED 2026-09-16: doi.org answers a scripted client with HTTP 403 (it wants a browser), so
    the first attempt uses a browser User-Agent and the second goes straight to the ScienceBase
    item API for the same id.  Whichever route succeeds is recorded - and if both fail the result
    carries `match: null` rather than a comfortable-looking true.
    """
    item_id = next((tok for tok in service.split("/") if tok.startswith("SB_")), "")
    expected = item_id[3:].split("_")[0]
    out = {"doi": SGMC_DATA_DOI, "item_id_in_service": expected, "match": None, "route": None}
    if not expected:
        out["error"] = f"service name carries no ScienceBase item id: {service}"
        return out
    try:
        landing = resolve_doi(SGMC_DATA_DOI)
        out.update(route="doi.org redirect", resolved=landing,
                   match=bool(expected in landing))
    except Exception as exc:                                              # noqa: BLE001
        out["doi_route_error"] = f"{type(exc).__name__}: {exc}"
        try:
            url = f"https://www.sciencebase.gov/catalog/item/{expected}?format=json"
            doc = http_json(url)
            out.update(route="sciencebase item api", resolved=url,
                       item_title=doc.get("title"),
                       match=bool(doc.get("id") == expected and doc.get("title")))
        except Exception as exc2:                                         # noqa: BLE001
            out["item_route_error"] = f"{type(exc2).__name__}: {exc2}"
    out["meaning"] = ("the FeatureServer queried is the published service of this data release"
                      if out["match"] else
                      "NOT ESTABLISHED - the service could not be tied to the DOI on this run; "
                      "cite the data release only if a reviewer can reproduce the link by hand")
    return out


def raster_bbox_4326(raster: Path) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in EPSG:4326 for a raster's footprint."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(raster) as src:
        if src.crs is None:
            raise SystemExit(f"{raster}: no CRS, cannot build a geographic query envelope")
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)


def rule_id_domain(layer_meta: dict) -> dict[str, str]:
    """The service's own RuleID coded-value domain (code -> published class name)."""
    out: dict[str, str] = {}
    for fld in layer_meta.get("fields", []):
        if fld.get("name") != "RuleID":
            continue
        for cv in (fld.get("domain") or {}).get("codedValues", []) or []:
            out[str(cv["code"])] = str(cv.get("name", ""))
    return out


def fault_rule_ids(layer_meta: dict, extra_names: tuple[str, ...] = ()) -> dict[str, str]:
    """RuleID -> published class name, for classes that are faults (or in `extra_names`).

    The names come from the service's own coded-value domain, i.e. from USGS, not from this file.
    """
    want = ("fault",) + tuple(n.lower() for n in extra_names)
    return {code: name for code, name in rule_id_domain(layer_meta).items()
            if any(w in name.lower() for w in want)}


def esri_paths_to_geojson(geom: dict) -> dict | None:
    """ESRI JSON polyline -> GeoJSON LineString / MultiLineString (coordinates are passed through).

    True curves are returned densified by the service when outSR is requested, so `paths` is always
    a dense vertex list here; nothing is interpolated locally.
    """
    paths = geom.get("paths") or []
    lines = [[[float(x), float(y)] for x, y in p] for p in paths if len(p) >= 2]
    if not lines:
        return None
    if len(lines) == 1:
        return {"type": "LineString", "coordinates": lines[0]}
    return {"type": "MultiLineString", "coordinates": lines}


def build_query(service: str, layer: int, bbox, rule_ids: dict, offset: int, page: int) -> str:
    """The exact request URL, so a reviewer can paste it into a browser and see the same bytes."""
    params = {
        "where": "RuleID IN (%s)" % ",".join(sorted(rule_ids, key=int)) if rule_ids else "1=1",
        "geometry": ",".join(f"{v:.6f}" for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID,STATE,DESCRIPTION,REFERENCE,DIGITAL_URL,NGMDB1,RuleID",
        "returnGeometry": "true",
        "outSR": "4326",
        "geometryPrecision": "6",
        "resultOffset": str(offset),
        "resultRecordCount": str(page),
        "orderByFields": "OBJECTID",
        "f": "json",
    }
    return f"{service.rstrip('/')}/{layer}/query?" + urllib.parse.urlencode(params)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--service", default=SERVICE)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--bbox-source", default="data/labels.tif",
                    help="raster whose CRS+bounds define the query envelope (the competition grid)")
    ap.add_argument("--out", default="data/external/sgmc_proxy_faults.geojson")
    ap.add_argument("--meta", default="data/evidence/proxy/fetch_meta.json")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--extra-classes", default="",
                    help="comma-separated extra published class names to include, e.g. 'Lineament'")
    ap.add_argument("--rule-ids", default=None,
                    help="override the domain-derived selection (comma-separated codes)")
    a = ap.parse_args()

    extra = tuple(s.strip() for s in a.extra_classes.split(",") if s.strip())
    bbox = raster_bbox_4326(Path(a.bbox_source))

    # The report covers the SGMC_Structure attribute semantics; a link that 404s is worse than no
    # link, and this repository has already published one guessed USGS PDF path that did not exist.
    # So the documented links are checked here, on the same run that fetches the faults.
    link_check = {}
    for label, url in (("report", SGMC_REPORT_PDF),
                       ("appendix5", SGMC_ATTRIBUTE_DICTIONARY),
                       ("appendix2", SGMC_ALL_FIELD_DEFINITIONS)):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:           # noqa: S310 - fixed https
                link_check[url] = f"OK_{r.status} ({label})"
        except Exception as exc:                                          # noqa: BLE001
            link_check[url] = f"UNREACHABLE: {type(exc).__name__}: {exc}"
    ident = service_identity(a.service)
    print(f"  service identity: {ident['item_id_in_service']} in {ident['resolved']} -> "
          f"{ident['match']}")
    for url, status in link_check.items():
        print(f"  doc link {status:<24} {url}")
    if any(s.startswith("UNREACHABLE") for s in link_check.values()):
        raise SystemExit("a documented USGS link is unreachable - fix the link, do not publish it")

    layer_meta = http_json(f"{a.service.rstrip('/')}/{a.layer}?f=json")

    if a.rule_ids:
        domain = rule_id_domain(layer_meta)
        chosen = {str(c).strip(): domain.get(str(c).strip(), f"RuleID {c} (--rule-ids override, "
                                                             "name not in the service domain)")
                  for c in a.rule_ids.split(",") if str(c).strip()}
        selection = "explicit --rule-ids (name taken from the service coded-value domain)"
    else:
        chosen = fault_rule_ids(layer_meta, extra)
        domain, selection = chosen, "service coded-value domain, names containing 'fault'"

    if not chosen:
        raise SystemExit("no RuleIDs selected - inspect the layer metadata before trusting an empty proxy")
    print(f"query envelope (EPSG:4326): {tuple(round(v, 6) for v in bbox)}")
    print(f"rule classes kept ({len(chosen)}): {json.dumps(chosen, sort_keys=True)}")

    features: list[dict] = []
    urls: list[str] = []
    offsets: list[int] = []
    offset = 0
    while True:
        url = build_query(a.service, a.layer, bbox, chosen, offset, a.page_size)
        urls.append(url)
        page = http_json(url)
        batch = page.get("features", [])
        offsets.append(len(batch))
        for f in batch:
            geom = esri_paths_to_geojson(f.get("geometry") or {})
            if geom is None:
                continue
            attrs = f.get("attributes") or {}
            features.append({
                "type": "Feature",
                "id": attrs.get("OBJECTID"),
                "properties": {k: attrs.get(k) for k in
                               ("OBJECTID", "STATE", "DESCRIPTION", "RuleID", "REFERENCE",
                                "DIGITAL_URL", "NGMDB1")},
                "geometry": geom,
            })
        print(f"  offset {offset}: {len(batch)} features (total kept {len(features)})")
        offset += len(batch)
        if not batch or len(batch) < a.page_size or len(features) >= a.max_features:
            break

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"type": "FeatureCollection", "features": features}
    payload = json.dumps(doc).encode("utf-8")
    out.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    from collections import Counter
    per_rule = Counter(str(f["properties"].get("RuleID")) for f in features)
    per_state = Counter(str(f["properties"].get("STATE")) for f in features)
    meta = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/fetch_proxy_faults.py",
        "purpose": ("independent fault catalogue (proxy) for the GeoDAWN footprint - stands in for "
                    "the scored 'new fault' population, which is not downloadable"),
        "source": {
            "dataset": "USGS State Geologic Map Compilation (SGMC), Data Series 1052 (ver. 1.1)",
            "citation": ("Horton, J.D., San Juan, C.A., and Stoeser, D.B., 2017, The State Geologic "
                         "Map Compilation (SGMC) geodatabase of the conterminous United States "
                         "(ver. 1.1, August 2017): U.S. Geological Survey Data Series 1052, 46 p."),
            "doi": SGMC_DOI, "data_doi": SGMC_DATA_DOI, "methods_url": SGMC_METHODS,
            "report_pdf": SGMC_REPORT_PDF,
            "attribute_dictionary": SGMC_ATTRIBUTE_DICTIONARY,
            "all_field_definitions": SGMC_ALL_FIELD_DEFINITIONS,
            "successor_release_2026": SGMC_2026_GEMS_DOI,
            "service": a.service, "layer": a.layer,
            "layer_name": layer_meta.get("name"),
            "license": "US public domain (USGS); no credentials used",
            "official_identity": ident,
            "documented_links_checked": link_check,
        },
        "query": {
            "bbox_epsg4326": [round(v, 6) for v in bbox],
            "bbox_source": str(a.bbox_source),
            "bbox_source_crs": "the CRS of --bbox-source (competition grid: EPSG:32611)",
            "where": urls[0].split("where=")[1].split("&")[0] if urls else None,
            "selection_rule": selection,
            "rule_id_to_class": chosen,
            "first_url": urls[0] if urls else None,
            "request_urls": urls,
            "pages": len(urls),
            "features_per_page": offsets,
        },
        "result": {
            "features": len(features),
            "per_rule_id": dict(sorted(per_rule.items(), key=lambda kv: int(kv[0]))),
            "per_rule_id_name": {k: chosen.get(k, "") for k in per_rule},
            "per_state": dict(per_state),
            "out": str(out),
            "bytes": len(payload),
            "sha256": sha,
        },
        "caveats": [
            "The proxy is built from published geologic maps (surface mapping at 1:250,000 or "
            "finer); the scored faults were drawn by experts from the GeoDAWN geophysics. The two "
            "populations are related but not identical - see docs/DISCOVERY_PLAN.md 3(b).",
            "Overlap with the training labels is not removed here; scripts/build_proxy_catalogue.py "
            "does that and reports the overlap.",
            "The competition data must not be redistributed, so this script writes only the "
            "independent USGS layer; the labels are read locally to compute the overlap.",
        ],
    }
    mp = Path(a.meta)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(payload)} bytes, sha256 {sha[:16]}...) and {mp}")
    print(f"features {len(features)}  states {dict(per_state)}")
    if not features:
        print("WARNING: zero features - the query selected nothing; do not use this as a proxy",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
