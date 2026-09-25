"""Torch-free GeoTIFF submission writer with a fail-loud post-write verification.

WHY THIS MODULE EXISTS (defect found 2026-09-16, evidence in this repo)
----------------------------------------------------------------------
The 6-fold ensemble workflow (Actions run 35042805806) trained all six folds, then died at
the LAST step with

    rasterio._err.CPLE_AppDefinedError: _TIFFVSetField:submission.tif: Bad value 3292 for "TileWidth" tag
    rasterio.errors.RasterBlockError: The height and width of TIFF dataset blocks must be multiples of 16

Root cause: the writer copied the *sample submission's* rasterio profile and then forced
``TILED="YES"``.  A striped GeoTIFF reports ``blockxsize == width`` (3292 here); GDAL then
refuses to reinterpret 3292 as a tile width, because TIFF tile dimensions must be multiples
of 16.  Nothing was written, the workflow still reported success (the crash was masked by a
``| tee`` pipeline without ``pipefail``), and a 110-byte GDAL stub was committed to the
branch as if it were a submission.

So a submission writer has exactly two jobs, and this module does both:
  1. build a *clean* profile - never inherit block geometry from another file;
  2. prove the file it just wrote is readable, correctly gridded and non-degenerate, and
     raise if it is not (an empty/invalid submission must never look like success).

A third job was added 2026-09-25 (defect found by an actual platform rejection, see
``conform_to_template``): the field written to disk must be *conformant with the official
sample submission's validity mask* - finite inside it, NaN outside it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

__all__ = ["TILE", "clean_profile", "write_submission", "sha256_file",
           "conform_to_template", "conformance_findings", "cli_validate"]

TILE = 256  # multiple of 16 -> always a legal TIFF tile dimension

# keys rasterio copies from a source dataset that must NOT be reused verbatim when the
# block layout changes (they describe the *source* file's storage, not our output's)
_STALE_KEYS = ("blockxsize", "blockysize", "blockyshapes", "strides")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_pixels(path) -> str:
    """Hash the PIXELS only, not the container.

    WHY THIS IS NOT REDUNDANT WITH sha256(file).  Measured 2026-09-16: two runs of the identical
    fold artifacts produced submissions with **bit-identical pixels** (this hash matched) but
    different file hashes, because one float64 in a GeoTIFF metadata tag differed by one ULP
    (the selected floor, 0.4696741044002384 vs ...2383, from geomspace under a different numpy
    build).  The container hash is the right thing to attach to a submitted file - it is what a
    reviewer can re-check - but the pixel hash is the right thing to compare two runs with, so both
    are recorded.  NaN is hashed as NaN (not 0) so a footprint difference cannot hide.
    """
    with rasterio.open(path) as src:
        arr = src.read()
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<f4").tobytes()).hexdigest()


def clean_profile(src_profile: dict | None = None, *, height: int | None = None,
                  width: int | None = None, crs=None, transform=None,
                  dtype: str = "float32", compress: str = "lzw",
                  tiled: bool = True, tile: int = TILE, nodata=None) -> dict:
    """Build a legal GTiff profile for a submission raster.

    ``src_profile`` (e.g. from ``sample_submission.tif``) contributes georeferencing and
    layout *intent* only - never its block geometry.  Tile blocks are pinned to a multiple
    of 16 for every axis, which is what the failed run got wrong.
    """
    prof: dict = {}
    if src_profile:
        prof.update({k: v for k, v in src_profile.items() if k not in _STALE_KEYS})
        prof["driver"] = "GTiff"
    prof.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress=compress)
    if height is not None:
        prof["height"] = int(height)
    if width is not None:
        prof["width"] = int(width)
    if crs is not None:
        prof["crs"] = crs
    if transform is not None:
        prof["transform"] = transform
    for k in _STALE_KEYS:
        prof.pop(k, None)
    if tiled:
        prof["tiled"] = True
        # a raster narrower/shorter than the tile is legal (GDAL pads), but the tile
        # dimension itself must stay a multiple of 16
        prof["blockxsize"] = int(tile)
        prof["blockysize"] = int(tile)
    else:
        prof["tiled"] = False
    prof.pop("interleave", None)
    return prof


def write_submission(path, array: np.ndarray, profile: dict, *, blockcheck: bool = True,
                     band_description: str | None = None, tags: dict | None = None) -> dict:
    """Write ``array`` to ``path`` and verify the bytes on disk.

    Returns a verification dict (path, bytes, sha256, grid, dtype, counts, ranges).
    Raises ``RuntimeError`` - never returns quietly - if the written file does not read
    back as a single-band float32 raster on the requested grid with any finite mass.
    """
    path = Path(path)
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"submission array must be 2-D, got shape {arr.shape}")
    h, w = arr.shape
    if int(profile.get("height", h)) != h or int(profile.get("width", w)) != w:
        raise ValueError(f"profile {profile.get('width')}x{profile.get('height')} "
                         f"does not match array {w}x{h}")
    arr = arr.astype(np.float32, copy=False)
    bad = np.isfinite(arr) & ((arr < -1e-6) | (arr > 1 + 1e-6))
    if bad.any():
        raise ValueError(f"values outside [0,1] (n={int(bad.sum())}, "
                         f"max={float(np.nanmax(arr))}) - spec requires 0..1")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        # rasterio has no append mode ("a" raises ValueError), so tags must be set here
        if band_description:
            dst.set_band_description(1, band_description)
        if tags:
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"writer produced no bytes at {path}")

    # ---- read the bytes back: the step that would have caught the 110-byte stub --------
    with rasterio.open(path) as src:
        if src.width != w or src.height != h:
            raise RuntimeError(f"read-back grid {src.width}x{src.height} != written {w}x{h}")
        if src.count != 1:
            raise RuntimeError(f"read-back has {src.count} bands, expected 1")
        if src.dtypes[0] != "float32":
            raise RuntimeError(f"read-back dtype {src.dtypes[0]}, expected float32")
        back = src.read(1)
        if (src.crs is None) and (profile.get("crs") is not None):
            raise RuntimeError("read-back lost the CRS")
        if blockcheck and src.block_shapes and src.profile.get("tiled"):
            for bh, bw in src.block_shapes:
                if bh % 16 or bw % 16:
                    raise RuntimeError(f"illegal tiled block {bh}x{bw} (must be multiples of 16)")
        finite = int(np.isfinite(back).sum())
        info = dict(
            path=str(path), bytes=int(path.stat().st_size), sha256=sha256_file(path),
            width=int(src.width), height=int(src.height), count=int(src.count),
            dtype=str(src.dtypes[0]), crs=str(src.crs), res=[float(r) for r in src.res],
            transform=list(src.transform), nodata=src.nodata,
            tiled=bool(src.profile.get("tiled")),
            block_shapes=[list(b) for b in (src.block_shapes or [])],
            finite_px=finite, total_px=int(back.size), nan_px=int(back.size - finite),
            nonzero_px=int(np.count_nonzero(np.nan_to_num(back))),
            prob_mass=float(np.nansum(back)),
        )
    if info["finite_px"] == 0:
        raise RuntimeError("read-back raster has no finite pixels")
    if info["nonzero_px"] == 0:
        raise RuntimeError("read-back raster is all zeros (total-fault-absence template) - "
                           "refusing to report success")
    return info


def dump_json(obj, path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    return str(p)


def conform_to_template(array, template) -> tuple[np.ndarray, dict]:
    """Align a prediction field to the official sample submission's validity mask.

    WHY THIS EXISTS (platform rejection observed 2026-09-24)
    ---------------------------------------------------------
    DrivenData rejected the file this repository shipped, with the platform's own error
    text: ``Predicted values must be in range [0, 1]``.  Every finite value of the
    shipped raster was already in [0, 1] - what was wrong was *where* the NaNs were:

    * 3,061 px inside the sample submission's valid region (which is exactly the labels
      raster's valid mask, i.e. the scored region) were NaN, and NaN is not in [0, 1];
    * 1,540 px outside that region were finite where the template is NaN;
    * the raster carried no GDAL_NODATA tag while the template carries ``"nan"``.

    The rule the platform's own template encodes (problem description, "Submission
    format": *"the same bounds as the training data, and data outside the bounds is null
    or nan"* and *"values between 0 and 1"*) is one mask, taken from the template itself:

        template finite -> prediction must be finite and in [0, 1]
        template NaN    -> prediction must be NaN

    This function implements exactly that and reports what it changed.  The fill value
    inside is 0.0 - "no predicted fault where the model had no data" - which is what
    every scorer in this repository already counted (``src/metrics.py`` scores through
    ``np.nan_to_num(..., nan=0.0)``, and the scored populations are subsets of the
    template's valid region, measured 2026-09-24), so conforming a field never changes
    a recorded measurement.

    Returns ``(conformed, stats)``.  ``stats`` counts ``filled_inside``,
    ``masked_outside``, ``clipped`` and ``unchanged`` pixels.  Raises ValueError on a
    shape mismatch - silently aligning a differently-shaped grid is how off-by-one
    submissions are born.
    """
    pred = np.asarray(array)
    ref = np.asarray(template)
    if pred.shape != ref.shape:
        raise ValueError(f"prediction {pred.shape} does not match template {ref.shape}")
    out = pred.astype(np.float32, copy=True)
    ref_valid = np.isfinite(ref)
    finite = np.isfinite(out)
    filled = ref_valid & ~finite                  # NaN inside -> 0.0 (the platform error)
    masked = ~ref_valid & finite                  # finite outside -> NaN (template mask)
    clipped = ref_valid & finite & ((out < 0.0) | (out > 1.0))
    changed = filled | masked | clipped
    out[filled] = 0.0
    out[masked] = np.nan
    if clipped.any():
        out[clipped] = np.clip(out[clipped], 0.0, 1.0)
    stats = dict(
        pixels=int(out.size),
        template_valid_px=int(ref_valid.sum()),
        filled_inside=int(filled.sum()),
        masked_outside=int(masked.sum()),
        clipped=int(clipped.sum()),
        unchanged=int(out.size - int(np.count_nonzero(changed))),
    )
    return out, stats


def conformance_findings(array, template) -> dict:
    """Report (without changing anything) the violations ``conform_to_template`` would fix.

    Used by check-only modes so a caller can *judge* a file - a gate - without ever
    mutating it.  Same mask semantics as ``conform_to_template``.
    """
    pred = np.asarray(array)
    ref = np.asarray(template)
    if pred.shape != ref.shape:
        raise ValueError(f"prediction {pred.shape} does not match template {ref.shape}")
    ref_valid = np.isfinite(ref)
    finite = np.isfinite(pred)
    out_of_range = finite & ((pred < 0.0) | (pred > 1.0))
    return dict(
        pixels=int(pred.size),
        template_valid_px=int(ref_valid.sum()),
        nan_inside_px=int((ref_valid & ~finite).sum()),
        finite_outside_px=int((~ref_valid & finite).sum()),
        out_of_range_px=int(out_of_range.sum()),
        conformant=bool(not (ref_valid & ~finite).any()
                         and not (~ref_valid & finite).any()
                         and not out_of_range.any()),
    )


# --------------------------------------------------------------------------
# The runner-side gate: `python -m src.submission_io validate-conformant FILE`.
# Exit 0 only when FILE is finite inside the official template's valid region,
# NaN outside it, in [0, 1], and declares the same GDAL_NODATA tag as the
# template.  This is the exact invariant whose absence let the 2026-09-24
# platform rejection ("Predicted values must be in range [0, 1]") through;
# scripts/validate_submission.py enforces the same rules with friendlier
# output, and this entry point exists for CI and runners that already live
# on this module's contract.
def _same_nodata(a, b) -> bool:
    """Nodata equality where NaN == NaN (the template's tag is 'nan')."""
    if a is None or b is None:
        return a is None and b is None
    try:
        if np.isnan(a) and np.isnan(b):
            return True
    except TypeError:
        pass
    return a == b


def cli_validate(argv=None) -> int:
    """Check-only conformance gate.  Returns 0 conformant, 1 not, 2 usage."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="python -m src.submission_io",
        description="fail-loud checks for the submission writer contract",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser(
        "validate-conformant",
        help="exit 1 unless FILE is finite inside the sample template's valid "
             "region, NaN outside it, in [0,1], with a matching GDAL_NODATA tag",
    )
    v.add_argument("pred", type=Path, help="GeoTIFF to judge (never modified)")
    v.add_argument("--sample", type=Path, default=None,
                   help="template mask (default: data/sample_submission.tif)")
    args = ap.parse_args(argv)

    pred: Path = args.pred
    sample: Path = args.sample or (
        Path(__file__).resolve().parents[1] / "data" / "sample_submission.tif")
    if not pred.exists():
        print(f"MISSING {pred}", file=sys.stderr)
        return 2
    if not sample.exists():
        print(f"MISSING template {sample}", file=sys.stderr)
        return 2
    with rasterio.open(pred) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    with rasterio.open(sample) as ds:
        tpl = ds.read(1)
        tpl_nodata = ds.nodata
    findings = conformance_findings(arr, tpl)
    findings["nodata"] = None if nodata is None else str(nodata)
    findings["nodata_ok"] = _same_nodata(nodata, tpl_nodata)
    findings["path"] = str(pred)
    findings["conformant"] = bool(findings["conformant"] and findings["nodata_ok"])
    print(json.dumps(findings, indent=1))
    if not findings["conformant"]:
        print(
            "NOT CONFORMANT: "
            f"{findings['nan_inside_px']} NaN inside the valid region, "
            f"{findings['finite_outside_px']} finite outside, "
            f"{findings['out_of_range_px']} out of [0,1], "
            f"nodata_ok={findings['nodata_ok']} — fix: "
            "python scripts/sanitize_submission.py --pred {pred} --write".format(pred=pred),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via cli_validate() in tests
    raise SystemExit(cli_validate())
