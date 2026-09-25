#!/usr/bin/env python3
"""
Download USGS 3DEP 1m DEM tiles for the GEMS/GeoDAWN region.

Two modes
---------
1) Tile list mode (default): downloads tiles listed in data/dem_links.json
   (partial capture of the competition's Digital-elevation-model-links-JSON;
   canonical URLs S3-verified). Filters: --zone --x --y --project --limit.

2) --complete-listing: enumerates ALL 1m tiles for the three projects from the
   official S3 bucket (ListObjectsV2, paginated) and writes
   data/dem_links_full.json — use this to replace the partial competition
   capture with the complete authoritative list, then download from it.

Bucket: prd-tnm.s3.amazonaws.com (USGS The National Map staged products,
public; ~90-380 MB per tile; the full region is tens of GB).

Examples
--------
  python scripts/download_dem_tiles.py --complete-listing
  python scripts/download_dem_tiles.py --limit 4            # smoke test
  python scripts/download_dem_tiles.py --project NV_WestCentral_EarthMRI_2020_D20 --y 443
"""
import argparse, csv, json, sys, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "https://prd-tnm.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
PROJECTS = ["CA_SierraNevada_B22", "NV_WestCentral_EarthMRI_2020_D20", "NV_Humboldt_2021_D21"]


def http_get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "GEMSDOE/1.0 (competition use; USGS public data)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_keys(project, prefix=""):
    """Yield (key, size_bytes) for a project prefix via paginated ListObjectsV2."""
    token = None
    while True:
        q = {"list-type": "2", "prefix": f"StagedProducts/Elevation/1m/Projects/{project}/TIFF/{prefix}", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = http_get(BUCKET + "/?" + urllib.parse.urlencode(q))
        root = ET.fromstring(xml)
        for c in root.findall("s3:Contents", NS):
            yield c.findtext("s3:Key", "", NS), int(c.findtext("s3:Size", "0", NS))
        trunc = root.findtext("s3:IsTruncated", "false", NS) == "true"
        token = root.findtext("s3:NextContinuationToken", None, NS)
        if not trunc or not token:
            return


def complete_listing(out):
    doc = {"metadata": {"generated": time.strftime("%Y-%m-%d"),
                        "source": "prd-tnm.s3.amazonaws.com ListObjectsV2 (official USGS TNM staged products)",
                        "note": "Complete authoritative tile list for the three projects referenced by the competition DEM-links file"},
           "tiles": []}
    for p in PROJECTS:
        n = 0
        for key, size in list_keys(p):
            if not key.endswith(".tif"):
                continue
            fname = key.rsplit("/", 1)[-1]
            doc["tiles"].append(dict(project=p, filename=fname, canonical_url=f"{BUCKET}/{key}",
                                     size_bytes=size, s3_verified=True))
            n += 1
        print(f"{p}: {n} tiles")
    Path(out).write_text(json.dumps(doc, indent=2))
    print(f"wrote {out} ({len(doc['tiles'])} tiles, "
          f"{sum(t['size_bytes'] for t in doc['tiles'])/1e9:.1f} GB total)")


def download(tiles, outdir, log):
    outdir.mkdir(parents=True, exist_ok=True)
    new_log = not log.exists()
    with log.open("a", newline="") as lf:
        w = csv.writer(lf)
        if new_log:
            w.writerow(["timestamp", "filename", "status", "bytes", "url"])
        for t in tiles:
            dest = outdir / t["filename"]
            if dest.exists() and dest.stat().st_size == t.get("size_bytes", dest.stat().st_size):
                print(f"skip (exists): {t['filename']}")
                continue
            for attempt in range(1, 4):
                try:
                    t0 = time.time()
                    data = http_get(t["canonical_url"], timeout=1800)
                    dest.write_bytes(data)
                    w.writerow([time.strftime("%FT%T"), t["filename"], "ok", len(data), t["canonical_url"]])
                    lf.flush()
                    print(f"ok {t['filename']} {len(data)/1e6:.0f}MB in {time.time()-t0:.0f}s")
                    break
                except Exception as e:
                    w.writerow([time.strftime("%FT%T"), t["filename"], f"retry{attempt}", 0, str(e)[:120]])
                    lf.flush()
                    print(f"attempt {attempt} failed: {e}")
                    time.sleep(5 * attempt)
            else:
                print(f"FAILED: {t['filename']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", default=str(ROOT / "data/dem_links.json"))
    ap.add_argument("--out", default=str(ROOT / "data/dem"))
    ap.add_argument("--complete-listing", action="store_true")
    ap.add_argument("--zone", type=int), ap.add_argument("--x", type=int), ap.add_argument("--y", type=int)
    ap.add_argument("--project"), ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.complete_listing:
        complete_listing(str(ROOT / "data/dem_links_full.json"))
        return

    doc = json.loads(Path(a.links).read_text())
    tiles = doc["tiles"]
    if a.zone: tiles = [t for t in tiles if t["zone"] == a.zone]
    if a.x: tiles = [t for t in tiles if t["x"] == a.x]
    if a.y: tiles = [t for t in tiles if t["y"] == a.y]
    if a.project: tiles = [t for t in tiles if t["project"] == a.project]
    if a.limit: tiles = tiles[:a.limit]
    if not tiles:
        print("no tiles matched"); sys.exit(1)
    print(f"downloading {len(tiles)} tiles "
          f"({sum(t.get('size_bytes',0) for t in tiles)/1e9:.1f} GB) -> {a.out}")
    download(tiles, Path(a.out), Path(a.out) / "download_log.csv")


if __name__ == "__main__":
    main()
