#!/usr/bin/env python3
"""Encode the shippable submission artifact so a browser can rebuild it byte-of-pixels.

WHY THIS EXISTS
---------------
The platform wants a single-band float32 GeoTIFF on a fixed grid (see docs/how_to_submit.html).
Every route this repository documents to produce one (curl the committed artifact, fire a GitHub
Actions workflow, run the CPU classifier) needs *a machine with a Python environment*.  This
script removes that requirement: it writes a lossless description of the artifact into `docs/`
that `docs/geotiff_writer.js` can turn back into a valid GeoTIFF in any browser, with no server
and no install.

WHAT IS SHIPPED
---------------
    docs/submission_meta.json   grid + georeferencing + integrity pins + provenance (small)
    docs/submission_field.bin   the pixel field, run-length encoded (532,174 B on the 2026-09-22
                                artifact, i.e. 49,118,960 raw bytes of float32 -> 4.3 % of that)

THE ENCODING (version-stamped, and asserted on both sides)
----------------------------------------------------------
`submission_field.bin` is a stream of runs in row-major order, each written as

    uleb128(run_length)  u8(value_code)

with `value_code` 0 -> 0.0f, 1 -> 1.0f, 2 -> IEEE-754 NaN (quiet, 0x7fc00000), the same bit
pattern GDAL/rasterio use for a NaN float32.  Runs must tile the grid exactly:
`sum(run_length) == width * height`, otherwise this script refuses to write.

HONESTY CONSTRAINT: this encoding can only represent a field whose finite values are exactly
{0.0, 1.0}.  If the artifact is a continuous probability field, the script **fails loudly**
rather than quantising it, because a quantised raster silently presented as "the artifact" is
precisely the kind of claim this project has been burned by.  A continuous field must be fetched
as the artifact itself (route A) or regenerated (routes B-D).

USAGE
    python scripts/build_submission_payload.py                # write docs/submission_{meta.json,field.bin}
    python scripts/build_submission_payload.py --check        # assert the committed files match the artifact
    python scripts/build_submission_payload.py --verify       # decode and compare pixel-for-pixel
    python scripts/build_submission_payload.py --print-encoding

Exit codes: 0 ok, 1 drift/refusal (the artifact cannot be represented losslessly, or the
committed payload no longer matches it), 2 the artifact is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]

#: the artifact the site treats as "the file to upload" - the same constant build_site.py uses.
#: Session 27: the union of the two detector families (`scripts/union_submission.py`), which is the
#: artifact data/evidence/submission_portfolio.json recommends under the organizers' mask, and whose
#: own bytes pass scripts/validate_submission.py (log committed beside it).
ARTIFACT = "data/evidence/union/submission.tif"
SIDECAR = "data/evidence/union/submission.tif.sha256"
BLEND_REPORT = "data/evidence/union/union_report.json"
VALIDATION_LOG = "data/evidence/union/validation.log"
SAMPLE = "data/sample_submission.tif"

META_REL = "docs/submission_meta.json"
BLOB_REL = "docs/submission_field.bin"
ENCODING = "gems-rle-v1"
NAN_U32 = 0x7FC00000          # quiet NaN, little-endian float32


# --------------------------------------------------------------------------- helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _uleb128(n: int) -> bytes:
    if n < 0:
        raise ValueError(f"run lengths are unsigned, got {n}")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def encode_runs(field: np.ndarray) -> tuple[bytes, list[tuple[int, int]]]:
    """Run-length encode a 3-valued float field into the gems-rle-v1 byte stream.

    Returns (blob, [(length, code), ...]).  Raises ValueError if the field is not losslessly
    representable - the caller must not swallow that.
    """
    flat = field.reshape(-1)
    nan_mask = np.isnan(flat)
    finite = flat[~nan_mask]
    if finite.size:
        uniq = np.unique(finite)
        bad = [float(u) for u in uniq if not (u == 0.0 or u == 1.0)]
        if bad:
            raise ValueError(
                "the field is not binary: finite values include "
                f"{bad[:8]}{'' if len(bad) <= 8 else ' …'} ({len(uniq)} distinct values). "
                "gems-rle-v1 can only store {0.0, 1.0, NaN}; quantising here would ship a "
                "different prediction than the artifact whose hash this payload pins. "
                f"Refusing. (Artifact: {ARTIFACT})")
    codes = np.zeros(flat.shape, dtype=np.int8)
    codes[nan_mask] = 2
    codes[flat == 1.0] = 1
    chg = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], chg))
    lengths = np.diff(np.concatenate((starts, [codes.size])))
    vals = codes[starts]
    blob = bytearray()
    for ln, code in zip(lengths.tolist(), vals.tolist()):
        blob += _uleb128(int(ln))
        blob.append(int(code))
    return bytes(blob), list(zip(lengths.tolist(), vals.tolist()))


def decode_runs(blob: bytes, width: int, height: int) -> np.ndarray:
    """Decode the byte stream back to a float32 field (Python twin of the JS decoder).

    Every failure is a ValueError with a message, mirroring `decodeRuns` in docs/geotiff_writer.js:
    the two decoders are supposed to agree on what they refuse, and a bare IndexError from one side
    and a clean error from the other is exactly the drift a reviewer could not see.
    """
    out = np.empty(width * height, dtype=np.float32)
    pos = n = 0
    target = width * height
    runs = 0
    while pos < len(blob):
        v = shift = 0
        while True:
            if pos >= len(blob):
                raise ValueError(f"truncated uleb128 at run {runs}")
            b = blob[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift > 49:
                raise ValueError(f"unreasonable run length at run {runs}")
        if pos >= len(blob):
            raise ValueError(f"run {runs} has no value code")
        code = blob[pos]
        pos += 1
        if v == 0:
            raise ValueError(f"zero-length run at {runs}")
        if code == 0:
            val = np.float32(0.0)
        elif code == 1:
            val = np.float32(1.0)
        elif code == 2:
            val = np.float32(struct.unpack("<f", struct.pack("<I", NAN_U32))[0])
        else:
            raise ValueError(f"unknown value code {code} at run {runs}")
        if n + v > target:
            raise ValueError(f"run {runs} (length {v}) overruns the grid")
        out[n:n + v] = val
        n += v
        runs += 1
        if n == target and pos < len(blob):
            raise ValueError(f"{len(blob) - pos} trailing bytes after the grid is filled")
    if n != target:
        raise ValueError(f"runs cover {n} px of a {target} px grid")
    return out.reshape(height, width)


# --------------------------------------------------------------------------- build
def read_artifact(root: Path, rel: str = ARTIFACT):
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(str(p))
    with rasterio.open(p) as src:
        band = src.read(1)
        info = dict(width=src.width, height=src.height, count=src.count, dtype=src.dtypes[0],
                    crs=str(src.crs), epsg=(src.crs.to_epsg() if src.crs else None),
                    transform=list(src.transform), res=[float(src.res[0]), float(src.res[1])],
                    # NaN is not valid JSON (JSON.parse in a browser rejects the bare token);
                    # the template writes its outside-footprint nodata as the string "nan".
                    nodata=(None if src.nodata is None else
                            ("nan" if isinstance(src.nodata, float) and np.isnan(src.nodata)
                             else src.nodata)),
                    bounds=[float(x) for x in src.bounds],
                    driver=src.driver, tiled=src.profile.get("tiled"),
                    blockxsize=src.profile.get("blockxsize"), blockysize=src.profile.get("blockysize"),
                    compression=src.profile.get("compress"),
                    tags={str(k): str(v) for k, v in (src.tags() or {}).items()},
                    band_description=str(src.descriptions[0]) if src.descriptions and src.descriptions[0] else None)
    return p, band, info


def build_meta(root: Path, artifact: Path, band: np.ndarray, info: dict,
               blob: bytes, runs: list[tuple[int, int]]) -> dict:
    """Everything the browser needs to rebuild the file, and every pin it needs to check itself."""
    finite = band[np.isfinite(band)]
    raw = np.ascontiguousarray(band, dtype="<f4").tobytes()
    recorded = None
    side = root / SIDECAR
    if side.exists():
        recorded = side.read_text().strip().split()[0]
    val_log = root / VALIDATION_LOG
    val = None
    if val_log.exists():
        text = val_log.read_text()
        val = dict(log_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   passed="Validation PASSED" in text,
                   lines=len([l for l in text.splitlines() if l.strip()]))
    blend = None
    bp = root / BLEND_REPORT
    if bp.exists():
        d = json.loads(bp.read_text())
        keys = ("n_folds", "folds", "usable_folds", "policy", "shaping", "dti", "scores",
                "emission", "provenance", "weights")
        blend = {k: d[k] for k in keys if k in d}
    sample = None
    sp = root / SAMPLE
    if sp.exists():
        with rasterio.open(sp) as s:
            sample = dict(grid=[s.width, s.height], transform=list(s.transform),
                          epsg=(s.crs.to_epsg() if s.crs else None),
                          shape_matches=(s.width == info["width"] and s.height == info["height"]),
                          transform_matches=(list(s.transform) == [float(x) for x in info["transform"]]))
    n_on = int(np.count_nonzero(finite == 1.0))
    return dict(
        schema=1,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        generator="scripts/build_submission_payload.py",
        purpose="lets docs/how_to_submit.html rebuild the shippable GeoTIFF in a browser "
                "with no install and no server; every pin below is measured from the artifact",
        encoding=dict(format=ENCODING, file=BLOB_REL, bytes=len(blob),
                      layout="row-major runs: uleb128(length), u8(code)",
                      codes={"0": "float32 0.0", "1": "float32 1.0",
                             "2": "float32 NaN (quiet, 0x7fc00000)"},
                      n_runs=len(runs),
                      expected_pixels=int(info["width"] * info["height"])),
        grid=dict(width=int(info["width"]), height=int(info["height"]),
                  count=int(info["count"]), dtype=info["dtype"], epsg=info["epsg"],
                  crs=info["crs"], res=info["res"], transform=info["transform"],
                  bounds=info["bounds"], nodata=info["nodata"],
                  origin=[float(info["transform"][2]), float(info["transform"][5])],
                  tiepoint=[0.0, 0.0, 0.0,
                            float(info["transform"][2]), float(info["transform"][5]), 0.0],
                  pixelscale=[float(info["transform"][0]), float(-info["transform"][4]), 0.0],
                  band_description=info["band_description"], gdal_tags=info["tags"],
                  source_tiff=dict(tiled=info["tiled"], blockxsize=info["blockxsize"],
                                   blockysize=info["blockysize"],
                                   compression=info["compression"], driver=info["driver"])),
        artifact=dict(path=ARTIFACT, bytes=artifact.stat().st_size,
                      sha256=sha256_file(artifact), recorded_sha256=recorded,
                      recorded_in=SIDECAR, hash_matches=bool(recorded) and
                      recorded == sha256_file(artifact)),
        field=dict(pixels=int(band.size), zero_px=int(np.count_nonzero(finite == 0.0)),
                   one_px=n_on, nan_px=int(np.isnan(band).sum()),
                   min=float(finite.min()) if finite.size else None,
                   max=float(finite.max()) if finite.size else None,
                   float32_bytes=len(raw), float32_sha256=hashlib.sha256(raw).hexdigest()),
        blob=dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()),
        checks=dict(sample_submission=sample, validation=val),
        provenance=dict(policy=info["tags"].get("policy"), kind=info["tags"].get("kind"),
                        script=info["tags"].get("script"), blend_report=blend),
        how_this_was_made=dict(
            builder="python scripts/build_submission_payload.py",
            writer="docs/geotiff_writer.js (same code path runs under node in the test suite)",
            parity_test="tests/test_site_generator.py",
            evidence="data/evidence/site_generator.json",
        ),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--artifact", default=ARTIFACT)
    ap.add_argument("--check", action="store_true",
                    help="compare the committed docs/submission_meta.json + field blob with a rebuild")
    ap.add_argument("--verify", action="store_true",
                    help="decode the committed blob and compare it pixel-for-pixel with the artifact")
    ap.add_argument("--print-encoding", action="store_true")
    ap.add_argument("--out-dir", default=None, help="write the two files somewhere else (tmp dirs in tests)")
    a = ap.parse_args(argv)
    root = Path(a.root)

    if a.print_encoding:
        print(__doc__)
        return 0

    try:
        artifact, band, info = read_artifact(root, a.artifact)
    except FileNotFoundError as ex:
        print(f"MISSING  {ex}\n"
              f"  the artifact is committed in the repository; `git pull` or run the CPU route "
              f"(python scripts/baseline_submission.py) to produce one")
        return 2

    try:
        blob, runs = encode_runs(band)
    except ValueError as ex:
        print(f"REFUSED  {ex}")
        return 1
    meta = build_meta(root, artifact, band, info, blob, runs)

    # the round trip is verified HERE, before anything is written: decode what we just encoded
    back = decode_runs(blob, info["width"], info["height"])
    same_nan = bool(np.array_equal(np.isnan(back), np.isnan(band)))
    finite_equal = bool(np.array_equal(np.where(np.isnan(back), 0, back),
                                       np.where(np.isnan(band), 0, band)))
    exact_bits = bytes(back.astype("<f4").tobytes()) == bytes(np.ascontiguousarray(band, dtype="<f4").tobytes())
    meta["round_trip"] = dict(nan_positions_match=same_nan, finite_values_match=finite_equal,
                              float32_bytes_identical=exact_bits)

    if a.verify:
        meta_p = root / META_REL
        blob_p = root / BLOB_REL
        if not (meta_p.exists() and blob_p.exists()):
            print(f"MISSING  {META_REL} / {BLOB_REL} - run without --check/--verify to create them")
            return 2
        committed = decode_runs(blob_p.read_bytes(), info["width"], info["height"])
        bits = bytes(committed.astype("<f4").tobytes()) == bytes(np.ascontiguousarray(band, dtype="<f4").tobytes())
        cm = json.loads(meta_p.read_text())
        print(f"artifact      {a.artifact}  sha256 {meta['artifact']['sha256'][:16]}…")
        print(f"blob sha256   {meta['blob']['sha256'][:16]}… committed {cm['blob']['sha256'][:16]}… "
              f"{'MATCH' if cm['blob']['sha256'] == meta['blob']['sha256'] else 'DIFFER'}")
        print(f"decoded pixels {committed.size} · NaN {int(np.isnan(committed).sum()):,} · "
              f"1.0 {int(np.count_nonzero(committed == 1.0)):,} · float32 bits identical: {bits}")
        return 0 if bits and cm["blob"]["sha256"] == meta["blob"]["sha256"] else 1

    if a.check:
        drift = []
        for rel, want in ((META_REL, json.dumps(meta, indent=1, sort_keys=True, allow_nan=False)),
                          (BLOB_REL, blob)):
            p = root / rel
            if not p.exists():
                drift.append(f"{rel}: not committed")
                continue
            got = p.read_bytes()
            want_b = want.encode() if isinstance(want, str) else want
            if got != want_b:
                # The manifest carries two kinds of field, and only one of them is a pin:
                #   * grid / artifact / blob / encoding / field / round_trip / generator describe
                #     THE PAYLOAD and must match a rebuild anywhere, or the payload has drifted;
                #   * `generated_utc` is a stamp, and `checks` records what the *build machine*
                #     measured against data/ -- the 418 MB placed stack, which is not committed.
                #     A fresh clone or a CI runner without that placement cannot reproduce those
                #     numbers, so demanding them here reported DRIFT on a payload that was fine
                #     (found by the Tests workflow on run 35803568817, not by reading).
                if rel.endswith(".json"):
                    gj, wj = json.loads(got), json.loads(want)
                    for d in (gj, wj):
                        d.pop("generated_utc", None)
                        d.pop("checks", None)
                    drift_ok = gj != wj
                else:
                    drift_ok = True
                if drift_ok:
                    drift.append(f"{rel}: {len(got)} B on disk vs {len(want_b)} B rebuilt "
                                 f"(sha {hashlib.sha256(got).hexdigest()[:12]} vs "
                                 f"{hashlib.sha256(want_b).hexdigest()[:12]})")
        if drift:
            print("DRIFT — the committed site payload no longer reproduces the artifact:")
            for d in drift:
                print("  -", d)
            print("  fix: python scripts/build_submission_payload.py && rebuild the site")
            return 1
        print(f"OK — {META_REL} + {BLOB_REL} reproduce {a.artifact} "
              f"({meta['encoding']['n_runs']:,} runs, {meta['field']['one_px']:,} px at 1.0, "
              f"float32 bits identical after a decode round trip)")
        if meta["checks"] != json.loads((root / META_REL).read_text()).get("checks", {}):
            print("INFO — the `checks` block (this checkout's data/, the validator log) differs from "
                  "the committed one because this machine has a different environment; the payload "
                  "pins above are what a submission depends on")
        return 0

    out_dir = Path(a.out_dir) if a.out_dir else root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "submission_meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True, allow_nan=False))
    (out_dir / "submission_field.bin").write_bytes(blob)
    print(f"wrote {out_dir / 'submission_meta.json'} "
          f"({len(json.dumps(meta, indent=1, allow_nan=False)):,} B)")
    print(f"wrote {out_dir / 'submission_field.bin'} ({len(blob):,} B, {meta['encoding']['n_runs']:,} runs)")
    print(f"round trip: float32 bits identical = {exact_bits}; "
          f"1.0 px = {meta['field']['one_px']:,}; NaN px = {meta['field']['nan_px']:,}")
    if meta["artifact"]["recorded_sha256"]:
        print(f"artifact hash {'matches' if meta['artifact']['hash_matches'] else 'MISMATCHES'} "
              f"the recorded sidecar ({meta['artifact']['sha256'][:16]}…)")
    return 0 if exact_bits else 1


if __name__ == "__main__":
    raise SystemExit(main())
