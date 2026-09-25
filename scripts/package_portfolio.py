#!/usr/bin/env python3
"""Package the measured submission portfolio into `docs/` with unique names, Notes and ZIPs.

WHY
---
A download a reader cannot identify is a download they cannot tell apart from their last
submission.  The DrivenData "New submission" dialog asks for a file and a short Note ("e.g.
clustering with k=25"); this script produces both, deterministically, from bytes that were scored
first (see `scripts/measure_submission_portfolio.py`).

WHAT IT GUARANTEES PER FILE (asserted, not asserted-about)
---------------------------------------------------------
  * single band, float32, EPSG:32611, 100 m, transform identical to the official template
  * every pixel inside the template's valid region is finite and in [0, 1]
  * every pixel outside it is NaN, and GDAL_NODATA is NaN (the problem page's "data outside the
    bounds is null or nan")
  * a `.sha256` sidecar and a single-member `.zip` that re-hashes to the same digest
  * a unique file name (`<strategy>-<UTC stamp>-<sha8>.tif`) and a copy-pasteable Note
    (`<strategy> | <one-line description> | <sha8>`)

Every check is re-derived from the bytes at run time and recorded in
`data/evidence/portfolio_files.json`, which `scripts/build_site.py` renders.  A file that fails a
check is NOT copied into `docs/` and the script exits 1.

USAGE
    python scripts/package_portfolio.py
    python scripts/package_portfolio.py --check      # verify only; write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TEMPLATE = ROOT / "data/sample_submission.tif"

#: (key, strategy label, source artifact, one-line description for the Note)
#: The first entry is the submission this repository recommends and the page leads with; the
#: other two exist because they are the arms the union is built from, so a reader can submit any
#: of the three and compare them on the leaderboard.  Candidates that only sat at the random
#: floor (`data/evidence/suture/`, the SUTURE lineament arm) or that carry catalogue echo
#: (`docs/gemsdoe2_turbo_submission.tif`) are deliberately absent.
PORTFOLIO = [
    ("gemsdoe2-dual-family-union", "GEMSDOE2 dual-family fault union",
     "data/evidence/union/submission.tif",
     "11-fold ensemble union 6-fold blend, catalogue-masked"),
    # the *conformant* copy: the raw blend's own bytes carried 3,061 NaN inside the template's
    # valid region and 1,540 finite px outside it, i.e. exactly the rejection class this repository
    # was hit by; the raw file is deliberately not offered
    ("gemsdoe2-precision-arm", "GEMSDOE2 precision arm",
     "data/evidence/runs/35042805806/submission_conformant.tif",
     "6-fold blend, highest independent-split score, thinnest emission"),
    ("gemsdoe2-recall-arm", "GEMSDOE2 recall arm",
     "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif",
     "11-fold U-Net ensemble, widest coverage"),
    # the pre-registered *bet* arm: the union plus a one-pixel corridor along the supplied
    # catalogue, built for the population the organizers named (new-fault truth within 300 m of a
    # known trace) and invisible to every local protocol by construction -- see
    # `data/evidence/extension/extension_report.json` for H1/H2 and the local ceiling (1.000)
    ("gemsdoe2-extension-arm", "GEMSDOE2 extension arm",
     "data/evidence/extension/submission.tif",
     "union + 300 m corridor on known traces, bets on corrections/extensions"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_format(path: Path, template: np.ndarray, tmpl_meta: dict) -> dict:
    with rasterio.open(path) as src:
        a = src.read(1)
        meta = dict(dtype=src.dtypes[0], count=src.count, crs=str(src.crs),
                    transform=tuple(round(v, 6) for v in src.transform[:6]),
                    nodata=src.nodata, shape=(src.height, src.width))
    inside = np.isfinite(template)
    fin = np.isfinite(a)
    inside_vals = a[inside]
    checks = {
        "single_band": meta["count"] == 1,
        "float32": meta["dtype"] == "float32",
        "crs_matches_template": meta["crs"] == tmpl_meta["crs"],
        "transform_matches_template": meta["transform"] == tmpl_meta["transform"],
        "shape_matches_template": meta["shape"] == tmpl_meta["shape"],
        "finite_inside_valid_region": bool(fin[inside].all()),
        "values_in_0_1_inside": bool(inside_vals[fin[inside]].size
                                     and (inside_vals[fin[inside]] >= 0).all()
                                     and (inside_vals[fin[inside]] <= 1).all()),
        "nan_outside_valid_region": bool((~fin[~inside]).all()),
        "nodata_is_nan": bool(meta["nodata"] is None or np.isnan(meta["nodata"])),
    }
    # the check that the platform reported as a range error on 2026-09-24
    checks["no_nonfinite_inside"] = int((~fin & inside).sum()) == 0
    return {"checks": checks, "ok": all(checks.values()), "meta": meta,
            "finite_px": int(fin.sum()), "emitted_px": int((np.nan_to_num(a, nan=0.0) > 0).sum()),
            "nan_px": int((~fin).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only; copy nothing")
    ap.add_argument("--json", default="data/evidence/portfolio_files.json")
    args = ap.parse_args(argv)

    if not TEMPLATE.exists():
        print(f"FAIL: {TEMPLATE} missing - run python scripts/assemble_data_bridge.py",
              file=sys.stderr)
        return 2
    with rasterio.open(TEMPLATE) as t:
        template = t.read(1)
        tmpl_meta = dict(crs=str(t.crs), transform=tuple(round(v, 6) for v in t.transform[:6]),
                         shape=(t.height, t.width))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    entries, failures = [], 0
    for key, label, rel, blurb in PORTFOLIO:
        src = ROOT / rel
        if not src.exists():
            print(f"  MISSING {rel}", file=sys.stderr)
            failures += 1
            continue
        res = check_format(src, template, tmpl_meta)
        sha = sha256_file(src)
        # The file committed in docs/ is content-addressed and therefore stable: links do not rot
        # when the packaging is re-run.  The *submission* name (what the browser saves and what the
        # leaderboard shows) additionally carries the packaging instant, so two uploads a reader
        # makes from this page are distinguishable in their own submission history.
        docs_file = f"{key}-{sha[:8]}.tif"
        unique = f"{key}-{stamp}-{sha[:8]}.tif"
        note = f"{label} | {blurb} | {sha[:8]}"
        entry = {
            "key": key, "label": label, "source": rel, "present": True,
            "unique_name": unique, "docs_file": docs_file, "note": note, "sha256": sha,
            "bytes": src.stat().st_size, "format": res,
            "docs_path": f"docs/{docs_file}", "docs_zip": f"docs/{docs_file[:-4]}.zip",
        }
        if not res["ok"]:
            failed = [k for k, v in res["checks"].items() if not v]
            print(f"  FAIL {rel}: {failed}", file=sys.stderr)
            failures += 1
            entries.append(entry | {"copied": False})
            continue

        if not args.check:
            dst = DOCS / docs_file
            shutil.copyfile(src, dst)
            (DOCS / (docs_file + ".sha256")).write_text(f"{sha}  {docs_file}\n")
            zip_path = DOCS / f"{docs_file[:-4]}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(dst, arcname=docs_file)
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                inner = zf.read(docs_file)
            zip_sha = hashlib.sha256(inner).hexdigest()
            entry["zip_members"] = names
            entry["zip_inner_sha256"] = zip_sha
            entry["zip_inner_matches"] = zip_sha == sha
            entry["zip_bytes"] = zip_path.stat().st_size
            entry["copied"] = True
        else:
            entry["copied"] = False
        print(f"  OK   {docs_file}  ({entry['bytes']:,} B, {res['emitted_px']:,} emitted px, "
              f"sha256 {sha[:12]}...)  submit as {unique}")
        entries.append(entry)

    # stale copies of this portfolio's own files (an older sha of the same key) are removed, so a
    # re-run cannot leave two files a reader might upload by mistake
    if not args.check:
        keep = {e.get("docs_file") for e in entries if e.get("docs_file")}
        keep |= {k + ".sha256" for k in keep if k}
        keep |= {k[:-4] + ".zip" for k in keep if k and k.endswith(".tif")}
        for key, _label, _rel, _blurb in PORTFOLIO:
            for p in DOCS.glob(f"{key}-*"):
                if p.name not in keep:
                    p.unlink()
                    print(f"  removed stale docs/{p.name}")

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/package_portfolio.py",
        "template": {"path": str(TEMPLATE.relative_to(ROOT)), **tmpl_meta,
                     "sha256": sha256_file(TEMPLATE)},
        "checks": (
            "single band, float32, EPSG:32611, 100 m, template transform and shape, every pixel "
            "inside the template's valid region finite and in [0,1], every pixel outside it NaN, "
            "GDAL_NODATA = NaN"),
        "format_bullets_source": ("https://www.drivendata.org/competitions/306/"
                                  "competition-doe-gems/page/967/#submission-format"),
        "submission_dialog": ("https://www.drivendata.org/competitions/306/"
                              "competition-doe-gems/submissions/"),
        "files": entries,
        "all_ok": failures == 0,
    }
    if not args.check:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")
    if failures:
        print(f"FAILED: {failures} file(s) did not pass the format gate", file=sys.stderr)
        return 1
    print("all portfolio files pass the format gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
