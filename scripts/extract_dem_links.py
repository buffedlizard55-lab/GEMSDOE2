#!/usr/bin/env python3
"""Recover the competition's 1 m DEM tile list — with ZERO trust in OCR characters.

THE PROBLEM
-----------
The competition ships the DEM list as `1m_DEM_links.csv` on the (login-walled) data tab.
The public Dropbox mirror is a PDF, and — measured on the runner, 2026-09-14 — that PDF
has **no text layer at all**:

    pypdf 41 chars / 0 'http'   pdftotext 0 chars / 0 'http'   pdfplumber 41 chars / 0 'http'

It is a raster scan, so OCR is the only way in.  But OCR output cannot be trusted
character-by-character.  Real observed corruption in this file:

    202@_D20          -> 2020_D20        ('0' read as '@')
    USGS_ 1M 10 x75   -> USGS_1M_10_x75  (spaces injected for underscores)
    QLi / QL1         -> ambiguous digit/letter
    x24y44@           -> x24y440
    .tif” / . tit”    -> .tif            (smart quotes, letter confusion)
    prd- \n tnm.s3    -> prd-tnm.s3      (line-wrap hyphenation)

A URL assembled from those bytes is a *hallucination risk*: it looks plausible and 404s,
or worse, silently points at the wrong tile.

THE APPROACH
------------
Do not reconstruct URLs from OCR text.  Instead:

  1. OCR the PDF (tesseract, 300 DPI).
  2. From the OCR text extract only the two things that are *structurally* recoverable
     and self-checking: the PROJECT name and the TILE GRID ID (`x<NN>y<NNN>`).  Both are
     normalised with a small, explicit repair table ('@'->'0', whitespace->'_', etc.).
  3. Enumerate the REAL USGS 3DEP S3 bucket (`prd-tnm`, public ListObjectsV2) for each
     project directory.  This is the authoritative source of truth for what tiles exist.
  4. Match each OCR-derived (project, tile) pair against the real key list.  A URL is
     emitted ONLY if it exists in the bucket listing.
  5. Anything that cannot be matched is reported in `unmatched` as an irregularity for
     manual review — never guessed at.

The result: every emitted URL is a verbatim key from the official USGS bucket, and the
count of what OCR claimed vs. what was confirmed is reported honestly.

Sources
-------
  USGS 3DEP product page:  https://www.usgs.gov/3d-elevation-program
  Bucket (public, no auth): https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix=StagedProducts/Elevation/1m/Projects/
  The National Map downloader: https://apps.nationalmap.gov/downloader/

Usage (runs on the GitHub runner; the dev sandbox cannot reach S3):
  python scripts/extract_dem_links.py --pdf data/Digital-elevation-model-links-JSON.pdf \
      --out data/dem_links.json --evidence data/evidence
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

BUCKET = "prd-tnm"
BUCKET_HOST = f"https://{BUCKET}.s3.amazonaws.com"
PREFIX = "StagedProducts/Elevation/1m/Projects/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# OCR character repairs. Applied ONLY to project names and tile ids, both of which are
# then checked against the real bucket listing, so a bad repair cannot survive.
OCR_CHAR_FIXES = [
    ("@", "0"),      # '0' consistently mis-read as '@' in this scan
    ("\u201c", '"'), ("\u201d", '"'),
]
TILE_RE = re.compile(r"x(\d{2})y(\d{3})", re.I)
PROJECT_RE = re.compile(r"Projects/([A-Za-z0-9_@. -]+?)/(?:TIFF|TIFE|TlFF)", re.I)


def ocr_pdf(pdf: Path, dpi: int = 300) -> str:
    """Rasterise + OCR. Reports the text-layer extractors' failure for the record."""
    stats = {}
    try:
        from pypdf import PdfReader

        t = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages)
        stats["pypdf"] = {"chars": len(t), "http": t.lower().count("http")}
    except Exception as e:  # noqa: BLE001
        stats["pypdf"] = {"error": repr(e)}
    try:
        t = subprocess.run(["pdftotext", "-layout", "-nopgbrk", str(pdf), "-"],
                           capture_output=True, text=True, timeout=1800).stdout
        stats["pdftotext"] = {"chars": len(t), "http": t.lower().count("http")}
    except Exception as e:  # noqa: BLE001
        stats["pdftotext"] = {"error": repr(e)}

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), f"{td}/pg"],
                       check=True, timeout=7200)
        pages = sorted(Path(td).glob("pg*.png"))
        print(f"OCR: {len(pages)} pages at {dpi} DPI")
        out = []
        for i, img in enumerate(pages, 1):
            r = subprocess.run(["tesseract", str(img), "stdout", "--psm", "6"],
                               capture_output=True, text=True, timeout=900)
            out.append(r.stdout)
            if i % 10 == 0 or i == len(pages):
                print(f"  page {i}/{len(pages)}")
    text = "\n".join(out)
    stats["ocr_tesseract"] = {"chars": len(text), "http": text.lower().count("http")}
    globals()["_EXTRACTOR_STATS"] = stats
    print("extractor stats:", json.dumps(stats))
    return text


def normalise(s: str) -> str:
    for a, b in OCR_CHAR_FIXES:
        s = s.replace(a, b)
    s = re.sub(r"[ \t]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_. ")


def parse_entries(text: str):
    """-> list of (project_guess, tile_id) and the raw counts, from OCR'd URL-ish lines.

    URLs are wrapped across lines in the print, so the text is first de-wrapped, then each
    `https://...tif` run is examined for a project segment and an x##y### tile id.
    """
    t = text
    for a, b in OCR_CHAR_FIXES:
        t = t.replace(a, b)
    t = re.sub(r"-\s*\n\s*", "", t)              # hyphenated wrap
    t = re.sub(r"\n", " ", t)                    # everything onto one line
    t = re.sub(r"https\s*:\s*/\s*/", "https://", t)
    chunks = re.split(r"(?=https://)", t)
    entries, no_tile, no_proj = [], 0, 0
    for ch in chunks:
        if "prd" not in ch.lower() or "Projects/" not in ch:
            continue
        ch = ch[: ch.lower().find(".tif") + 4] if ".tif" in ch.lower() else ch[:400]
        pm = PROJECT_RE.search(ch)
        tm = TILE_RE.search(ch)
        if not tm:
            no_tile += 1
            continue
        if not pm:
            no_proj += 1
            continue
        entries.append((normalise(pm.group(1)), f"x{tm.group(1)}y{tm.group(2)}"))
    return entries, {"chunks_without_tile_id": no_tile, "chunks_without_project": no_proj}


def s3_list(prefix: str) -> list[str]:
    """Full ListObjectsV2 enumeration of a bucket prefix (paginated). Public, no auth."""
    keys, token = [], None
    while True:
        url = f"{BUCKET_HOST}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, timeout=120) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{S3_NS}Contents"):
            k = c.find(f"{S3_NS}Key")
            if k is not None:
                keys.append(k.text)
        trunc = root.find(f"{S3_NS}IsTruncated")
        if trunc is None or trunc.text != "true":
            break
        nt = root.find(f"{S3_NS}NextContinuationToken")
        if nt is None:
            break
        token = nt.text
    return keys


def best_project(guess: str, real_projects: list[str]) -> str | None:
    """Map an OCR'd project name to a real bucket directory.

    Exact match first; then case-insensitive; then a conservative similarity on the
    alphanumeric-only form. Anything below 0.9 similarity is refused (returns None) so a
    wrong project cannot be invented.
    """
    if guess in real_projects:
        return guess
    low = {p.lower(): p for p in real_projects}
    if guess.lower() in low:
        return low[guess.lower()]
    import difflib

    key = re.sub(r"[^a-z0-9]", "", guess.lower())
    cands = {re.sub(r"[^a-z0-9]", "", p.lower()): p for p in real_projects}
    m = difflib.get_close_matches(key, list(cands), n=1, cutoff=0.9)
    return cands[m[0]] if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="data/dem_links.json")
    ap.add_argument("--evidence", default="data/evidence")
    a = ap.parse_args()
    pdf = Path(a.pdf)
    if not pdf.exists():
        print(f"missing PDF: {pdf}")
        return 1

    text = ocr_pdf(pdf)
    ev = Path(a.evidence)
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "dem_links_raw_text.txt").write_text(text)

    entries, parse_stats = parse_entries(text)
    print(f"parsed {len(entries)} (project, tile) pairs; stats={parse_stats}")
    proj_counts = Counter(p for p, _ in entries)
    print("OCR project variants:", json.dumps(proj_counts.most_common(12), indent=1))

    # --- authoritative listing of the real project directories --------------------
    print("listing real bucket projects ...")
    top = s3_list(PREFIX)
    real_projects = sorted({k[len(PREFIX):].split("/")[0] for k in top if "/" in k[len(PREFIX):]})
    print(f"bucket has {len(real_projects)} projects under {PREFIX}")

    mapping, unresolved_projects = {}, []
    for g in proj_counts:
        r = best_project(g, real_projects)
        mapping[g] = r
        if r is None:
            unresolved_projects.append(g)
    resolved = sorted({v for v in mapping.values() if v})
    print(f"OCR project variants {len(proj_counts)} -> {len(resolved)} real projects; "
          f"{len(unresolved_projects)} unresolved")

    # --- enumerate the TIFF directory of each resolved project --------------------
    keys_by_project = {}
    for p in resolved:
        ks = [k for k in s3_list(f"{PREFIX}{p}/TIFF/") if k.lower().endswith(".tif")]
        keys_by_project[p] = ks
        print(f"  {p}: {len(ks)} tif keys")

    # index real keys by (project, tile id)
    index = {}
    for p, ks in keys_by_project.items():
        for k in ks:
            m = TILE_RE.search(k.rsplit("/", 1)[-1])
            if m:
                index.setdefault((p, f"x{m.group(1)}y{m.group(2)}".lower()), []).append(k)

    records, unmatched = [], []
    seen = set()
    for guess, tile in entries:
        proj = mapping.get(guess)
        if proj is None:
            unmatched.append({"reason": "project not resolvable in bucket",
                              "ocr_project": guess, "tile": tile})
            continue
        hits = index.get((proj, tile.lower()))
        if not hits:
            unmatched.append({"reason": "tile id not present in that project's bucket listing",
                              "project": proj, "tile": tile})
            continue
        for k in hits:
            if k in seen:
                continue
            seen.add(k)
            records.append({
                "url": f"{BUCKET_HOST}/{k}",
                "key": k,
                "project": proj,
                "tile": tile.lower(),
                "filename": k.rsplit("/", 1)[-1],
                "source": "USGS 3DEP public bucket listing (ListObjectsV2), matched to OCR of the competition PDF",
            })

    records.sort(key=lambda r: r["key"])
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/extract_dem_links.py on a GitHub-hosted runner",
        "source_pdf": pdf.name,
        "source_pdf_has_text_layer": False,
        "extractor_stats": globals().get("_EXTRACTOR_STATS"),
        "method": (
            "PDF has no text layer -> tesseract OCR at 300 DPI -> extract (project, x##y### tile) "
            "structure only -> resolve every URL against the authoritative USGS 3DEP S3 "
            "ListObjectsV2 listing. URLs are verbatim bucket keys; none are reconstructed from "
            "OCR characters."
        ),
        "bucket_prefix": PREFIX,
        "ocr_pairs_parsed": len(entries),
        "parse_stats": parse_stats,
        "ocr_project_variants": proj_counts.most_common(),
        "project_mapping_ocr_to_bucket": mapping,
        "unresolved_project_variants": unresolved_projects,
        "n_unique_tiles_confirmed": len(records),
        "total_bytes_note": "sizes not fetched here; use --verify-sizes or scripts/download_dem_tiles.py",
        "n_unmatched": len(unmatched),
        "unmatched_irregularities": unmatched[:200],
        "records": records,
        "verified": [r["url"] for r in records],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {a.out}: {len(records)} confirmed tiles, {len(unmatched)} unmatched")
    return 0


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (used in s3_list)

    raise SystemExit(main())
