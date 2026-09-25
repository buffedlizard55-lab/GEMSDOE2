#!/usr/bin/env python3
"""Conform a submission GeoTIFF to the official sample submission's validity mask.

WHAT THIS FIXES (the defect that produced a real platform rejection, 2026-09-24)
---------------------------------------------------------------------------------
A file downloaded from this project's own site was rejected by DrivenData with

    Predicted values must be in range [0, 1]

Every *finite* value of that file was in [0, 1].  The violation was the placement of
NaN: 3,061 px inside the sample submission's valid region (exactly the labels raster's
valid mask - the region the platform scores) were NaN, and NaN is not in [0, 1].  The
same file also had 1,540 finite px outside that region where the template is NaN, and
it carried no GDAL_NODATA tag while the template carries "nan".

This script applies ``src.submission_io.conform_to_template`` - fill NaN inside with
0.0, mask finite outside to NaN, clip to [0, 1] - and (in write mode) records a
before/after evidence JSON next to the file so the change is auditable rather than
silent.  Scoring is unaffected by construction: this repository's metric reduces
through ``np.nan_to_num(..., nan=0.0)`` and the scored populations are subsets of the
template's valid region (measured, see the function's docstring).

USAGE
    python scripts/sanitize_submission.py --pred FILE             # check only (exit 1 if not conformant)
    python scripts/sanitize_submission.py --pred FILE --write     # conform in place + evidence + sidecar
    python scripts/sanitize_submission.py --pred FILE --out OTHER # conform to a new path

Exit codes: 0 conformant / written, 1 violations found in check mode, 2 missing input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_io import (clean_profile, conformance_findings,  # noqa: E402
                               conform_to_template, sha256_file)


def sha256_pixels(path: Path) -> str:
    with rasterio.open(path) as src:
        arr = src.read()
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<f4").tobytes()).hexdigest()


def refresh_postwrite(path: Path) -> None:
    """Rewrite postwrite.json (the reblend workflow's post-write audit) for the new bytes.

    postwrite.json is rendered on the site as the profile of *the file as written*;
    leaving it describing the pre-sanitation bytes would publish a stale nodata/count.
    Same schema as .github/workflows/reblend.yml's post-write audit step.
    """
    pw = path.with_name("postwrite.json")
    if not pw.exists():
        return
    with rasterio.open(path) as src:
        a = src.read(1)
        info = dict(path=path.name, bytes=path.stat().st_size, width=src.width,
                    height=src.height, count=src.count, dtype=src.dtypes[0],
                    crs=str(src.crs), res=[float(r) for r in src.res],
                    nodata=(None if src.nodata is None or
                            (isinstance(src.nodata, float) and np.isnan(src.nodata))
                            else src.nodata),
                    nodata_label=("nan" if src.nodata is not None and
                                  isinstance(src.nodata, float) and np.isnan(src.nodata)
                                  else None),
                    tiled=bool(src.profile.get("tiled")),
                    block_shapes=[list(b) for b in (src.block_shapes or [])],
                    finite_px=int(np.isfinite(a).sum()),
                    nonzero_px=int(np.count_nonzero(np.nan_to_num(a))),
                    prob_mass=float(np.nansum(a)),
                    min=float(np.nanmin(a)), max=float(np.nanmax(a)))
    if info["nodata_label"] == "nan":
        info["nodata"] = "nan"          # what the file actually carries (GDAL_NODATA)
    else:
        info.pop("nodata_label", None)
    pw.write_text(json.dumps(info, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="submission GeoTIFF to check/conform")
    ap.add_argument("--sample", default=str(ROOT / "data/sample_submission.tif"),
                    help="official sample submission (the validity template)")
    ap.add_argument("--write", action="store_true",
                    help="conform in place (default is check-only)")
    ap.add_argument("--out", default=None,
                    help="write the conformed raster to this path instead of in place")
    ap.add_argument("--evidence", default=None,
                    help="evidence JSON path (default: <pred dir>/sanitize.json)")
    args = ap.parse_args()

    pred = Path(args.pred)
    sample = Path(args.sample)
    if not pred.exists():
        print(f"MISSING: {pred}", file=sys.stderr)
        return 2
    if not sample.exists():
        print(f"MISSING template: {sample} (place data with scripts/assemble_data_bridge.py)",
              file=sys.stderr)
        return 2

    with rasterio.open(sample) as ds:
        template = ds.read(1)
        sample_nodata = ds.nodata
        sample_sha = sha256_file(sample)
    with rasterio.open(pred) as ds:
        field = ds.read(1)
        profile = ds.profile.copy()
        pred_nodata = ds.nodata
        band_description = ds.descriptions[0] if ds.descriptions else None
        tags = ds.tags()

    findings = conformance_findings(field, template)
    def _nodata_label(value):
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return "nan"
        return value

    findings["nodata_tag"] = _nodata_label(pred_nodata)
    findings["template_nodata_tag"] = _nodata_label(sample_nodata)
    nodata_ok = findings["nodata_tag"] == findings["template_nodata_tag"]
    findings["nodata_matches_template"] = bool(nodata_ok)
    conformant = findings["conformant"] and nodata_ok

    if not (args.write or args.out):
        print(json.dumps(findings, indent=1))
        if conformant:
            print(f"CONFORMANT: {pred} matches the template's validity mask")
            return 0
        print(f"NOT CONFORMANT: {pred} - rerun with --write to fix "
              f"(nan_inside={findings['nan_inside_px']}, "
              f"finite_outside={findings['finite_outside_px']}, "
              f"out_of_range={findings['out_of_range_px']}, "
              f"nodata_matches={nodata_ok})")
        return 1

    conformed, stats = conform_to_template(field, template)
    before_sha = sha256_file(pred)
    before_px = sha256_pixels(pred)
    before_bytes = pred.stat().st_size

    out = Path(args.out) if args.out else pred
    # Preserve the container layout (tiled 256 / lzw) and only fix what is wrong: the
    # nodata tag.  clean_profile() re-legalises block geometry, never inheriting stripes.
    block = int(profile.get("blockxsize") or 0)
    tiled = bool(profile.get("tiled", True))
    tile = block if (block and block % 16 == 0) else 256
    profile = clean_profile(dict(profile), dtype="float32", nodata=sample_nodata,
                            tiled=tiled, tile=tile)
    tmp = out.with_suffix(out.suffix + ".sanitizing")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(conformed, 1)
        if band_description:
            dst.set_band_description(1, band_description)
        if tags:
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
    with rasterio.open(tmp) as ds:
        back = ds.read(1)
        if back.shape != conformed.shape or not np.array_equal(
                np.isfinite(back), np.isfinite(conformed)):
            tmp.unlink(missing_ok=True)
            print("ABORT: read-back of the conformed file does not match", file=sys.stderr)
            return 1
        if not conformance_findings(back, template)["conformant"]:
            tmp.unlink(missing_ok=True)
            print("ABORT: conformed file still fails the template check", file=sys.stderr)
            return 1
    tmp.replace(out)

    after_sha = sha256_file(out)
    after_px = sha256_pixels(out)
    evidence = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/sanitize_submission.py",
        command=f"python scripts/sanitize_submission.py --pred {args.pred} --sample {args.sample} "
                + ("--write" if not args.out else f"--out {args.out}"),
        why=("the platform rejected the pre-sanitation file with 'Predicted values must be in "
             "range [0, 1]': NaN inside the template's valid (scored) region; see "
             "src/submission_io.conform_to_template"),
        template=dict(path=args.sample, sha256=sample_sha),
        before=dict(path=str(pred), sha256=before_sha, sha256_pixels=before_px,
                    bytes=before_bytes, findings=findings),
        after=dict(path=str(out), sha256=after_sha, sha256_pixels=after_px,
                   bytes=out.stat().st_size,
                   findings=conformance_findings(back, template)),
        changes=stats,
        scoring_impact=("none by construction: src/metrics.py scores through "
                        "np.nan_to_num(..., nan=0.0) and every scored population is a "
                        "subset of the template's valid region"),
    )
    ev = Path(args.evidence) if args.evidence else out.with_name("sanitize.json")
    ev.write_text(json.dumps(evidence, indent=1) + "\n")

    # keep the sha256sum-format sidecar next to the file truthful
    sidecar = out.with_suffix(".sha256")
    if not sidecar.exists() and out.name == "submission.tif":
        sidecar = out.with_name("submission.sha256")
    if sidecar.exists():
        old = sidecar.read_text().split()
        path_field = old[1] if len(old) > 1 else str(out)
        sidecar.write_text(f"{after_sha}  {path_field}\n")

    refresh_postwrite(out)
    print(json.dumps(evidence["changes"], indent=1))
    print(f"WROTE {out} ({evidence['after']['bytes']} B, sha256 {after_sha[:16]}…)")
    print(f"EVIDENCE {ev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
