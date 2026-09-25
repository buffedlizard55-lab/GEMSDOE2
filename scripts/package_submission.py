#!/usr/bin/env python3
"""Package a submission GeoTIFF the way the platform's submit dialog accepts it.

The dialog on the competition's submissions page reads (quoted by the team, who can see it logged
in; the account-gated page is not fetchable from the dev sandbox): "You can submit a single-band
GeoTIFF (.tif) file, or a .zip file containing a single GeoTIFF, with your predictions."  Rules
§3.2 states the GeoTIFF form; the .zip alternative is the platform's own convenience, and both
end up as the same raster.

This script exists so the CLI route and the browser route
(`docs/geotiff_writer.js` → `buildZip`) produce the same container, and so "the zip holds exactly
that file" is a check rather than an assumption:

    python scripts/package_submission.py                       # zips the committed artifact
    python scripts/package_submission.py --tif submission.tif --out submission.zip
    python scripts/package_submission.py --check submission.zip --tif submission.tif

The default method is STORED (no recompression): the raster is already deflate-compressed inside,
and a zip that silently re-encodes the payload is one more thing to go wrong before an upload.
Exit 0 on success; 1 when the produced/checked archive does not read back as exactly one GeoTIFF
equal to the input bytes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIF = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
MEMBER = "submission.tif"
#: fixed so the archive is reproducible: the same input bytes always give the same zip bytes
#: fixed so the archive is reproducible: the same input bytes always give the same zip bytes, and
#: docs/geotiff_writer.js's buildZip() defaults to the same instant (2020-01-01T00:00:00Z)
FIXED_DATE_TIME = (2020, 1, 1, 0, 0, 0)


def write_zip(tif: Path, out: Path, method: int) -> None:
    """The ONLY place a zip is written here, so the determinism check compares like with like.

    (It used to be inline twice, and the two copies differed by force_zip64 - which wrote a data
    descriptor in one and not the other, so "re-package and diff" reported non-determinism in the
    packager when the packager was fine.)
    """
    with zipfile.ZipFile(out, "w") as z:
        zi = zipfile.ZipInfo(MEMBER, date_time=FIXED_DATE_TIME)
        zi.compress_type = method
        zi.external_attr = 0o644 << 16
        with z.open(zi, "w", force_zip64=False) as dst, tif.open("rb") as src:
            shutil.copyfileobj(src, dst, 1 << 20)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(tif: Path, out: Path, method: str | None = None) -> dict:
    """Read the archive back and prove it contains exactly this file, byte for byte."""
    rep = dict(zip=str(out), tif=str(tif), member=MEMBER, ok=False)
    if not out.exists():
        rep["error"] = "archive does not exist"
        return rep
    want = tif.read_bytes()
    try:
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            bad = z.testzip()
            info = z.getinfo(MEMBER) if MEMBER in names else None
            data = z.read(MEMBER) if info else b""
    except Exception as ex:                       # noqa: BLE001 - a bad zip is the thing under test
        rep["error"] = f"{type(ex).__name__}: {ex}"
        return rep
    rep.update(names=names, corrupt_member=bad, method_stored=bool(info and info.compress_type == zipfile.ZIP_STORED),
               member_bytes=info.file_size if info else None,
               member_sha256=hashlib.sha256(data).hexdigest() if data else None,
               zip_bytes=out.stat().st_size, tif_bytes=len(want))
    problems = []
    if names != [MEMBER]:
        problems.append(f"the archive must hold exactly one member named {MEMBER}, found {names}")
    if bad is not None:
        problems.append(f"ZipFile.testzip() reports a corrupt member: {bad}")
    if data != want:
        problems.append("the member is not byte-equal to the .tif that was packaged")
    if method == "stored" and not rep["method_stored"]:
        problems.append("the archive was asked to store (no recompression) but did not")
    if not zipfile.is_zipfile(out):
        problems.append("not recognised as a zip container")
    # ...and the member, once extracted, must still be a GeoTIFF a raster reader accepts.
    # (GDAL cannot open a .zip as a raster: the file has to come out of the archive first. That is
    # worth stating here because the platform accepts the archive and unpacks it itself.)
    if data:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="gems-zipcheck-") as td:
            member = Path(td) / MEMBER
            member.write_bytes(data)
            try:
                import rasterio
                with rasterio.open(member) as src:
                    rep["gdal_sees_after_unzip"] = dict(width=src.width, height=src.height,
                                                        count=src.count, dtype=src.dtypes[0],
                                                        crs=str(src.crs),
                                                        epsg=(src.crs.to_epsg() if src.crs else None),
                                                        res=[float(src.res[0]), float(src.res[1])],
                                                        nodata=str(src.nodata))
            except ImportError:
                rep["gdal_sees_after_unzip"] = "rasterio not installed here; container checks only"
            except Exception as ex:               # noqa: BLE001
                problems.append(f"GDAL cannot read the unzipped raster: {type(ex).__name__}: {ex}")
    rep["problems"] = problems
    rep["ok"] = not problems
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", default=DEFAULT_TIF)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--method", choices=["stored", "deflate"], default="stored",
                    help="stored (default) does not recompress the raster; deflate re-encodes it")
    ap.add_argument("--check", action="store_true", help="verify an existing archive and write nothing")
    ap.add_argument("--allow-archive-differences", action="store_true",
                    help="when checking a zip written by another route (the browser's buildZip), "
                         "compare the member bytes and skip the byte-for-byte archive comparison")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    tif = Path(a.tif)
    if not tif.is_absolute():
        tif = ROOT / tif
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out

    if a.check:
        rep = verify(tif, out, a.method)
        if a.allow_archive_differences and rep.get("member_sha256"):
            rep["member_sha256_of_input"] = sha256(tif)
            if rep["member_sha256"] == rep["member_sha256_of_input"]:
                rep["problems"] = [p for p in rep.get("problems", [])
                                   if "byte-equal" not in p]
                rep["ok"] = not rep["problems"]
        print(json.dumps(rep, indent=1) if a.json else
              f"{'OK  ' if rep['ok'] else 'BAD '} {out} · {rep.get('zip_bytes', '?')} B · "
              f"member {'=' if rep['ok'] else '≠'} {tif}")
        for p in rep.get("problems", []):
            print("  -", p)
        return 0 if rep["ok"] else 1

    if not tif.exists():
        print(f"MISSING {tif}\n  place it with python scripts/assemble_data_bridge.py, or point "
              f"--tif at a submission you produced", file=sys.stderr)
        return 2

    method = zipfile.ZIP_STORED if a.method == "stored" else zipfile.ZIP_DEFLATED
    tmp = out.with_suffix(out.suffix + ".part")
    write_zip(tif, tmp, method)
    tmp.replace(out)

    rep = verify(tif, out, a.method)
    rep["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rep["method"] = a.method
    rep["member_name_fixed"] = MEMBER
    rep["zip_date_time"] = list(FIXED_DATE_TIME)
    rep["tif_sha256"] = sha256(tif)
    rep["zip_sha256"] = sha256(out)
    # Re-package once and compare BEFORE anything is printed: "deterministic" is only worth
    # claiming if it was just measured, and the JSON, the human line and the exit code must all
    # describe the same verdict (they used to disagree, because this ran after the print).
    if rep["ok"]:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="gems-zip-") as td:
            rerun = Path(td) / "rerun.zip"
            write_zip(tif, rerun, method)
            rep["deterministic"] = sha256(rerun) == rep["zip_sha256"]
        if not rep["deterministic"]:
            rep["problems"] = rep.get("problems", []) + [
                "packaging is not deterministic: two runs of this script on identical input bytes "
                "wrote different archives"]
            rep["ok"] = False
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print(f"{'OK  ' if rep['ok'] else 'BAD '}wrote {out} ({rep['zip_bytes']:,} B, {a.method}) "
              f"holding {MEMBER} ({rep['tif_bytes']:,} B, sha256 {rep['tif_sha256'][:16]}…)")
        if "deterministic" in rep:
            print(f"  determinism: re-packaged bytes "
                  f"{'identical' if rep['deterministic'] else 'DIFFER'} ({rep['zip_sha256'][:16]}…)")
        for pr in rep.get("problems", []):
            print("  -", pr)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
