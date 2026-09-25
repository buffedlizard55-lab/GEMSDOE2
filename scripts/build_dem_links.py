#!/usr/bin/env python3
"""
Build data/dem_links.json from the verbatim PDF-text evidence file.

Design (no-hallucination policy):
- Tokens are extracted MECHANICALLY by regex from data/evidence/dem_links_pdf_chunk0.txt
  (verbatim capture of the competition's Digital-elevation-model-links-JSON.pdf, chunk 1/13).
- The canonical-URL rule (path project == FILENAME project) was proven by direct S3
  ListObjectsV2 spot checks (recorded below, exactly as returned by prd-tnm.s3.amazonaws.com).
- Nothing is invented: tokens that could not be fully parsed go to "unresolved_fragments".
"""
import json, re, hashlib
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "data/evidence/dem_links_pdf_chunk0.txt"
OUT = ROOT / "data/dem_links.json"

# S3 ListObjectsV2 spot checks performed 2026-09-12 via platform fetch of
# https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix=StagedProducts/Elevation/1m/Projects/...
# (key_count / size_bytes / etag / last_modified exactly as returned)
S3_VERIFICATIONS = {
    ("10", "75", "441", "CA_SierraNevada_B22"): dict(key_count=1, size_bytes=238657987, etag="40af28674a3ffb56512bebba59914cf0-46", last_modified="2026-02-13T03:48:25.000Z"),
    ("10", "75", "442", "NV_WestCentral_EarthMRI_2020_D20"): dict(key_count=1, size_bytes=208488258, etag="5da90763a80dcd914ead99eeead3523e", last_modified="2026-02-14T02:31:01.000Z", note="PDF row had path project CA_SierraNevada_B22; CA path variant returned key_count=0 — canonical uses filename project (rule proven)"),
    ("11", "26", "449", "NV_Humboldt_2021_D21"): dict(key_count=1, size_bytes=90967138, etag="a6a000fbc287aabeac8a5695843f6f0f", last_modified="2026-02-14T02:04:00.000Z"),
    ("11", "27", "443", "NV_WestCentral_EarthMRI_2020_D20"): dict(key_count=1, size_bytes=376446112, etag="748dc44573f44f1cc5fab6d35c8b06e5-72", last_modified="2026-02-14T02:33:09.000Z"),
    ("11", "24", "442", "CA_SierraNevada_B22"): dict(key_count=1, size_bytes=137528564, etag="374f509a4a866aea011973941d4c65c5", last_modified="2026-02-13T03:59:01.000Z", note="PDF row was truncated at 'USGS_1M_11_x24y442_CA_'; existence + identity resolved by this S3 check"),
    ("11", "27", "430", "NV_WestCentral_EarthMRI_2020_D20"): dict(key_count=1, size_bytes=264004305, etag="e18cef183f7abc80c6c50f17664156e7-51", last_modified="2026-02-14T02:32:47.000Z", note="PDF fragment lost the 'USGS_1M_' prefix and one zone digit; zone 11 variant verified to exist"),
}

CANON = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/{proj}/TIFF/{fname}"

def main():
    raw = EVID.read_text(encoding="utf-8")
    # cut off the added provenance header
    marker = "# ---- VERBATIM BELOW ----"
    body = raw.split(marker, 1)[1] if marker in raw else raw

    # normalize PDF line-wrap artifacts INSIDE tokens: spaces within "prd- tnm",
    # "USGS_1M_1 1_x..", "NV _Humboldt" (this only de-spaces, never adds characters)
    norm = body.replace("prd- tnm", "prd-tnm").replace("prd.tnm", "prd-tnm").replace("prdtnm", "prd-tnm")
    norm = re.sub(r"(USGS_1M_\d)\s+(\d)_", r"\1\2_", norm)      # broken zone digits
    norm = re.sub(r"(NV)\s+_(Humboldt)", r"\1_\2", norm)        # broken project name

    tiles = OrderedDict()
    # full tokens (path project + filename)
    pat_full = re.compile(
        r"StagedProducts/Elevation/1m/Projects/([A-Za-z0-9_]+)/TIFF/"
        r"(USGS_1M_(\d{1,2})_x(\d+)y(\d+)_([A-Za-z0-9_]+?)\.tif)"
    )
    for path_proj, fname, z, x, y, fproj in pat_full.findall(norm):
        key = (z, x, y, fproj)
        e = tiles.setdefault(key, dict(occurrences=0, pdf_path_projects=set()))
        e["occurrences"] += 1
        e["pdf_path_projects"].add(path_proj)

    # leading-truncated fragment: "1_x27y430_<proj>.tif" (lost "USGS_1M_" + zone digit)
    pat_frag = re.compile(r"(?<![\w])(\d)_(x\d+)(y\d+)_([A-Za-z0-9_]+?)\.tif")
    resolved_fragments = []
    for zd, xs, ys, proj in pat_frag.findall(norm):
        key_candidate = (None, xs[1:], ys[1:], proj)
        # only treat as unresolved if no full token with same x/y/proj exists
        if not any(k[1] == xs[1:] and k[2] == ys[1:] and k[3] == proj for k in tiles):
            resolved_fragments.append(dict(x=int(xs[1:]), y=int(ys[1:]), project=proj,
                                           fragment_zone_digit=zd,
                                           disposition="zone recovered via S3 verification (see tiles: zone 11 variant exists)" if (xs[1:], ys[1:], proj) == ("27", "430", "NV_WestCentral_EarthMRI_2020_D20") else "duplicate/trailing fragment of an already-captured tile or unresolvable tail"))

    # bare tail fragments that cannot be reconstructed at all (standalone only —
    # negative lookbehind excludes matches inside full filenames like NV_Humboldt_...)
    unresolved = []
    for m in re.finditer(r"(?<![\w])_Humboldt_2021_D21\.tif", norm):
        unresolved.append("_Humboldt_2021_D21.tif (x/y/zone lost in PDF parse — cannot reconstruct; will be captured by full-listing download)")

    # Promote PDF rows that were truncated/broken but were RESOLVED by direct S3
    # verification (truncated/broken evidence exists in the verbatim text):
    promoted = []
    for (z, x, y, proj), v in S3_VERIFICATIONS.items():
        key = (z, x, y, proj)
        if key in tiles:
            continue
        evidence = None
        if f"x{x}y{y}_{proj}" in norm or f"x{x}y{y}_{proj[:3]}" in norm:
            evidence = "broken/truncated row in PDF text"
        if evidence:
            promoted.append(((z, x, y, proj), v, evidence))
    for (z, x, y, proj), v, evidence in promoted:
        tiles[(z, x, y, proj)] = dict(occurrences=1, pdf_path_projects=set(), _promoted=(v, evidence))

    out_tiles = []
    for (z, x, y, proj), meta in tiles.items():
        fname = f"USGS_1M_{int(z):02d}_x{x}y{y}_{proj}.tif"
        url = CANON.format(proj=proj, fname=fname)
        v = meta.get("_promoted", (None, None))[0] or S3_VERIFICATIONS.get((z, x, y, proj), {})
        promoted_note = meta.get("_promoted", (None, None))[1]
        mismatch = sorted(p for p in meta["pdf_path_projects"] if p != proj)
        notes = ([v["note"]] if v.get("note") else [])
        if promoted_note:
            notes.append(f"Tile recovered from {promoted_note}; identity + existence proven by direct S3 ListObjectsV2 check")
        out_tiles.append(dict(
            zone=int(z), x=int(x), y=int(y), project=proj, filename=fname,
            canonical_url=url,
            occurrences_in_pdf_chunk=meta["occurrences"],
            s3_verified=bool(v.get("key_count")),
            s3_size_bytes=v.get("size_bytes"),
            s3_etag=v.get("etag"),
            s3_last_modified=v.get("last_modified"),
            irregularities=(["pdf path project != filename project: " + ",".join(mismatch) + " (canonical uses filename project — S3-proven)"] if mismatch else [])
                            + notes,
        ))
    out_tiles.sort(key=lambda t: (t["zone"], t["x"], t["y"], t["project"]))

    doc = dict(
        metadata=dict(
            generated="2026-09-12",
            source="Competition file Digital-elevation-model-links-JSON.pdf (Dropbox mirror of DrivenData data tab '1m_DEM_links'); original JSON hosted at drivendata-prod.s3.amazonaws.com (per PDF title)",
            evidence="data/evidence/dem_links_pdf_chunk0.txt (verbatim chunk 1 of 13; Dropbox st signature expired mid-capture — PARTIAL list)",
            status="PARTIAL (~1/13 of the competition list). Complete list obtainable on an unrestricted machine via scripts/download_dem_tiles.py --complete-listing (S3 ListObjectsV2 per project) or by re-fetching the competition JSON after DrivenData login.",
            canonical_url_rule="path project == FILENAME project; host prd-tnm.s3.amazonaws.com (PDF shows garbled host variants prdtnm/prd.tnm/prd- tnm — PDF parse noise)",
            rule_proof="S3 ListObjectsV2: filename-project path returns the object (key_count=1); mismatched CA path returned key_count=0",
            tiles_captured=len(out_tiles),
            s3_spot_checks=6,
        ),
        tiles=out_tiles,
        fragments_seen=resolved_fragments,
        unresolved_fragments=unresolved,
        irregularities=[
            "Duplicate rows in competition JSON (same tile repeated up to 5x) — captured as occurrences_in_pdf_chunk",
            "Path/filename project mismatches in competition JSON (e.g. Projects/CA_SierraNevada_B22/.../USGS_1M_10_x75y442_NV_WestCentral_EarthMRI_2020_D20.tif) — S3-proven wrong; canonical URL uses filename project",
            "Host written inconsistently (prdtnm / prd.tnm / prd- tnm) in PDF text — canonical host prd-tnm.s3.amazonaws.com verified by direct S3 queries",
            "Dropbox st share signature expired mid-capture; only chunk 1/13 obtained",
        ],
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"tiles: {len(out_tiles)}  verified: {sum(1 for t in out_tiles if t['s3_verified'])}  "
          f"fragments: {len(resolved_fragments)}  unresolved: {len(unresolved)}")
    for t in out_tiles:
        print(("  OK " if t["s3_verified"] else "     ") + t["filename"])

if __name__ == "__main__":
    main()
