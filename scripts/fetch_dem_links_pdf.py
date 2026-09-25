#!/usr/bin/env python3
"""Recover `data/dem_links.json` (1 m DEM tile links) from scratch, on any machine.

WHY THIS SCRIPT EXISTS
----------------------
An earlier session captured the competition's `Digital elevation model links JSON.pdf`
(Dropbox mirror of the data tab) and mechanically regex-extracted tile URLs into
`data/dem_links.json` + a verbatim evidence file.  Those artefacts lived under `data/`,
which `.gitignore` excluded except `data/README.md` - so they were never committed and did
not survive the sandbox.  Docs that referenced them therefore pointed at non-existent
files.  This is the audit finding; this script is the fix: the artefacts can be rebuilt
from the competition-provided mirror in one command, and the new .gitignore keeps the JSON
(+ the verbatim evidence text) in git so the audit trail is durable.

Mirror URL (from the competition data tab as supplied to this project; `rlkey` is durable,
the `st` signature expires within minutes - do not paste `st`).
Verification status at last check (2026-09-12, from this sandbox): the share link resolves
and Dropbox reports the file title "Digital elevation model links JSON.pdf", but binary
download and full-text PDF rendering are blocked by the egress allowlist -> run on an
unrestricted machine.

Usage:
    python scripts/fetch_dem_links_pdf.py                # download + parse -> data/dem_links.json
    python scripts/fetch_dem_links_pdf.py --pdf my.pdf   # parse an already-downloaded PDF
    python scripts/fetch_dem_links_pdf.py --complete-listing-instead
        -> prints the USGS The National Map alternative (no competition file needed)

The alternative authoritative enumeration (independent of the competition file) is the
ArcGIS REST index of the 1 m StagedProducts DEM; see scripts/download_dem_tiles.py
--complete-listing.  Cross-check both; report mismatches, never silently reconcile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_URL = ("https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/"
           "Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&dl=1")
# TNM 1m DEM staged products: the project index used by scripts/download_dem_tiles.py
TNM_INDEX = ("https://prd-tnm.s3.amazonaws.com/?list-type=2&delimiter=/&prefix="
             "StagedProducts/Elevation/1m/Projects/")
TILE_RE = re.compile(r"USGS_1M_\d+_[xy]\d+_[A-Za-z]{2}_[A-Za-z0-9_]+")
URL_RE = re.compile(r"https?://[^\s\"'<>,)]+")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download(url: str, dest: Path) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "GEMSDOE-data-recovery/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:      # noqa: S310 (fixed https host)
        data = r.read()
    dest.write_bytes(data)
    return data


def pdf_text(path: Path) -> str:
    """Extract text; prefers pypdf, falls back to pdftotext, then to raw-stream regex."""
    try:
        from pypdf import PdfReader                          # pip install pypdf
        rd = PdfReader(str(path))
        return "\n".join((pg.extract_text() or "") for pg in rd.pages)
    except Exception as e:                                   # noqa: BLE001
        print(f"pypdf unavailable/failed ({e}); trying pdftotext")
    try:
        out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True)
        return out.stdout
    except Exception as e:                                   # noqa: BLE001
        print(f"pdftotext failed ({e}); falling back to byte-level regex on the raw file")
        return path.read_bytes().decode("latin-1", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "data" / "raw" / "Digital-elevation-model-links-JSON.pdf"))
    ap.add_argument("--out", default=str(ROOT / "data" / "dem_links.json"))
    ap.add_argument("--evidence", default=str(ROOT / "data" / "evidence" / "dem_links_pdf_text.txt"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--complete-listing-instead", action="store_true")
    a = ap.parse_args()

    if a.complete_listing_instead:
        print("Enumerate 1 m DEM projects straight from the USGS bucket (no competition file):\n  " + TNM_INDEX)
        print("then per project:\n  https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix=StagedProducts/Elevation/1m/Projects/<NAME>/Thumbnail/")
        print("or run:  python scripts/download_dem_tiles.py --complete-listing")
        return 0

    pdf = Path(a.pdf)
    rec: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "source": PDF_URL, "provenance_note": "competition-provided Dropbox mirror of the data-tab DEM links file",
                 "tile_urls": [], "other_urls": [], "unresolved_fragments": []}
    if pdf.exists() and a.no_download:
        rec["pdf_sha256"] = sha256(pdf.read_bytes())
        rec["pdf_bytes"] = pdf.stat().st_size
    else:
        print(f"downloading {PDF_URL}")
        try:
            data = download(PDF_URL, pdf)
            rec["pdf_sha256"] = sha256(data)
            rec["pdf_bytes"] = len(data)
        except Exception as e:                                # noqa: BLE001
            sys.exit(f"download failed ({e}). This sandbox blocks dropbox.com - run on an "
                     "unrestricted machine, or use --complete-listing-instead.")
    print(f"pdf: {pdf} ({rec.get('pdf_bytes')} bytes, sha256 {rec.get('pdf_sha256','')[:16]})")

    text = pdf_text(pdf)
    Path(a.evidence).parent.mkdir(parents=True, exist_ok=True)
    Path(a.evidence).write_text(text)
    rec["evidence_file"] = str(Path(a.evidence).relative_to(ROOT))
    rec["evidence_sha256"] = sha256(text.encode())
    rec["tile_tokens_found"] = sorted(set(TILE_RE.findall(text)))

    urls = sorted(set(URL_RE.findall(text)))
    rec["tile_urls"] = [u for u in urls if "USGS_1M_" in u or "Elevation/1m" in u]
    rec["other_urls"] = [u for u in urls if u not in set(rec["tile_urls"])]
    # any tile token whose URL could not be assembled is recorded, never guessed
    covered = {t for u in rec["tile_urls"] for t in TILE_RE.findall(u)}
    rec["unresolved_fragments"] = sorted(set(rec["tile_tokens_found"]) - covered)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}: {len(rec['tile_urls'])} tile URLs, {len(rec['other_urls'])} other URLs, "
          f"{len(rec['unresolved_fragments'])} unresolved fragments (flagged, not filled in)")
    if rec["unresolved_fragments"]:
        print("  -> resolve those with the TNM listing or the DrivenData data tab; do NOT hand-type them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
