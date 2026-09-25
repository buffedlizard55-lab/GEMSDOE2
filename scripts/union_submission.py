#!/usr/bin/env python3
"""Build the shipped submission: the union of the two independent detector families.

WHAT IT DOES
------------
Takes the two *families* of trained detector this repository has produced and emits both, as one
binary field:

  recall arm     data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif
                 11-fold U-Net++/DeepLabV3+ ensemble, 172,974 px, covers 58 % of the held-out
                 segments within 3 px (`data/evidence/submission_portfolio.json`)
  precision arm  data/evidence/runs/35042805806/submission.tif
                 6-fold blend of a different model set (`scripts/blend_submission.py`, log at
                 data/evidence/runs/35042805806/blend_report.json), 21,492 px, covers 30 % with
                 8x fewer pixels

Why the union and not either arm: it is the best of the three fields on L (third-party protocol),
S (held-out segments, catalogue off) and P (the organizers' mask applied) -- see `measurement` in
the report this script writes -- and it is the best of the three once the hidden new-fault set is
at least as large as the local stand-in (`size_robustness`).  It is *not* the best on the strictest
local truth population, I (held-out segments more than 5 px from the catalogue), where the six-fold
arm alone wins 0.1173 to 0.0955 and buys 3.45x the random floor per pixel against the union's
1.39x; that is why the six-fold arm stays downloadable as an alternate arm rather than being
replaced by the union.  The metric pays `+k` for emission within 300 m of a truth pixel and charges
`0.2` for emission that is not, so what a pixel is worth is the truth it buys: the six-fold arm's
pixels buy ~3x more truth per pixel, and the eleven-fold arm buys coverage the six-fold arm does
not have; the union is the hedge between those two failure modes at an unknown truth-set size.

WHAT IT REFUSES TO DO
---------------------
It does not paint the supplied catalogue (organizers mask those pixels: forum 11516 -- that is the
"catalogue echo" the TURBO file still ships), it does not grade values to hedge, and it does not
tune a distance band.  Those three are measured and reported here instead:

  * binary vs graded: a graded field (1.0 on the union, v elsewhere) is measured over
    v in {0.0, 0.25, 0.5, 0.75, 1.0};
  * the near-trace band (emission restricted to within D px of the supplied catalogue) is measured
    over D in {8, 16, 32, 64, inf}, and the report says in plain words why its large local gain is
    a property of this proxy and not a licence to ship it: the proxy's "held-out" truth *is* a
    subset of the supplied catalogue (`data/evidence/seghold/eval_labels.tif` is carved out of
    `data/labels.tif`), so masking the prediction on the catalogue -- which is exactly what the
    platform does -- zeroes it on the truth itself, and any field that hugs the catalogue's edges
    looks better than it is.

The written file is conformant by construction (`src/submission_io.conform_to_template`): single
band float32 EPSG:32611, every pixel inside the template's valid region finite and in [0, 1], every
pixel outside it NaN, GDAL_NODATA = NaN.  That is the fix for the platform's rejection
"Predicted values must be in range [0, 1]" -- and unlike the two inputs, the union's own bytes are
re-validated here, not assumed.

USAGE
    python scripts/union_submission.py            # measure and write data/evidence/union/
    python scripts/union_submission.py --check    # measure only
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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.segment_holdout_eval import dti  # noqa: E402
from src.submission_io import conform_to_template  # noqa: E402

RECALL_ARM = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
PRECISION_ARM = "data/evidence/runs/35042805806/submission.tif"
TEMPLATE = "data/sample_submission.tif"
LABELS = "data/labels.tif"
SPLIT_DIR = "data/evidence/seghold"
OUT_DIR = "data/evidence/union"
INDEPENDENT_PX = 5.0
BANDS = (8, 16, 32, 64)
GRADES = (0.0, 0.25, 0.5, 0.75, 1.0)


def read(path) -> np.ndarray:
    with rasterio.open(ROOT / path) as src:
        return src.read(1).astype(np.float32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="measure only; write no raster")
    ap.add_argument("--json", default=f"{OUT_DIR}/union_report.json")
    args = ap.parse_args(argv)

    for rel in (RECALL_ARM, PRECISION_ARM, TEMPLATE, LABELS, f"{SPLIT_DIR}/eval_labels.tif"):
        if not (ROOT / rel).exists():
            print(f"FAIL: {rel} missing - run python scripts/assemble_data_bridge.py",
                  file=sys.stderr)
            return 2

    template = read(TEMPLATE)
    footprint = np.isfinite(template)
    supplied = read(LABELS) >= 1
    ev = read(f"{SPLIT_DIR}/eval_labels.tif")
    train_known = read(f"{SPLIT_DIR}/train_labels.tif") == 1
    valid = footprint & ~train_known
    d_train = ndimage.distance_transform_edt(~train_known).astype(np.float32)
    d_cat = ndimage.distance_transform_edt(~supplied).astype(np.float32)

    union = (read(RECALL_ARM) > 0) | (read(PRECISION_ARM) > 0)
    recall = read(RECALL_ARM) > 0
    precision = read(PRECISION_ARM) > 0

    truth_S = ev == 1
    independent = d_train > INDEPENDENT_PX
    truth_I = truth_S & independent
    comp = ndimage.label(truth_S, structure=np.ones((3, 3)))[0]
    ids = np.unique(comp[truth_S])
    tune_ids = set(ids[: len(ids) // 2].tolist())
    truth_tune = np.isin(comp, list(tune_ids)) & truth_S & independent
    truth_measure = truth_I & ~truth_tune
    valid_L = (ev == 0) | (ev == 1)

    def populations(field: np.ndarray) -> dict:
        """L / S / I / P for one field, plus the term-by-term detail behind them."""
        p = np.where(valid, field, 0.0).astype(np.float32)
        p_masked = np.where(valid & ~supplied, field, 0.0).astype(np.float32)
        L = dti(p, truth_S, valid=valid_L)
        S, tS = dti(p, truth_S, valid=valid, return_terms=True)
        I = dti(p, truth_I, valid=valid)
        P = dti(p_masked, truth_S, valid=valid)
        return {"L": L, "S": S, "I": I, "P_platform_mask": P,
                "tp": tS["tp"], "fp": tS["fp"], "fn": tS["fn"],
                "emitted_px": int((field > 0).sum()),
                "emitted_on_catalogue_px": int(((field > 0) & supplied).sum())}

    table = {
        "recall arm (11-fold, shipped by earlier sessions)": populations(recall.astype(np.float32)),
        "precision arm (6-fold blend)": populations(precision.astype(np.float32)),
        "union of the two arms (SHIPPED)": populations(union.astype(np.float32)),
    }
    print(f"{'field':48s} {'emitted':>8s} {'tp':>7s} {'L':>7s} {'S':>7s} {'I':>7s} {'P':>7s}")
    for name, row in table.items():
        print(f"{name:48s} {row['emitted_px']:>8,} {row['tp']:>7.1f} {row['L']:>7.4f} "
              f"{row['S']:>7.4f} {row['I']:>7.4f} {row['P_platform_mask']:>7.4f}")

    # -- what a graded field would buy (measured, not shipped) --------------------------------
    graded = {}
    for v in GRADES:
        field = np.where(union & (d_cat <= 8), 1.0, np.where(union, v, 0.0)).astype(np.float32)
        graded[f"1.0 on union within 8 px of the catalogue, {v:g} elsewhere"] = populations(field)

    # -- what a hard distance band would buy (measured, not shipped) ---------------------------
    bands = {}
    for D in BANDS + (float("inf"),):
        band = union & (d_cat <= D) if np.isfinite(D) else union
        bands[f"union restricted to within {D:g} px of the supplied catalogue"] = (
            populations(band.astype(np.float32)))
    out = np.where(union & footprint, 1.0, np.where(footprint, 0.0, np.nan)).astype(np.float32)
    conformed, conform_stats = conform_to_template(out, template)

    # independent-vs-random control for the shipped bytes, on the same population
    rng = np.random.default_rng(0)
    n = int((union & valid & ~supplied).sum())
    idx = rng.choice(np.flatnonzero(valid & ~supplied), n, replace=False)
    rand = np.zeros(union.shape, np.float32)
    rand.ravel()[idx] = 1.0
    random_control = populations(rand)
    kappa = [0.5, 1, 2, 4, 8]
    tS = table["union of the two arms (SHIPPED)"]
    size_robustness = {
        "note": ("DTI as a function of the hidden new-fault set's size, scaled to the local "
                 "stand-in |G| = %d px: TP and FN both scale, FP does not.  The leaderboard's |G| "
                 "is unknown, which is why this is reported instead of one number."
                 % int(truth_I.sum())),
        "kappa_G_over_local_proxy": {str(k): (tS["tp"] * k /
                                             (tS["tp"] * k + 0.2 * tS["fp"] + 0.8 * tS["fn"] * k))
                                     for k in kappa},
    }
    measured = {
        "split": {"dir": SPLIT_DIR, "held_out_px": int(truth_S.sum()),
                  "independent_px": int(truth_I.sum()),
                  "tune_px": int(truth_tune.sum()), "measure_px": int(truth_measure.sum())},
        "arms_and_union": table,
        "graded_variants_measured_not_shipped": graded,
        "distance_bands_measured_not_shipped": bands,
        "random_control_same_support": random_control,
        "size_robustness": size_robustness,
        "why_the_band_is_not_shipped": (
            "the proxy's held-out truth is carved out of the supplied catalogue, so it lies *on* "
            "the pixels the platform masks; a prediction restricted to the catalogue's edges "
            "therefore scores locally for a reason that says nothing about the hidden new-fault "
            "set, which the organizers describe as faults that are not in the supplied set (with "
            "corrections of known traces as only one explicit sub-target).  Shipping the band "
            "would trade real recall for a local artefact."),
    }
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/union_submission.py",
        "inputs": {rel: {"sha256": sha256_file(ROOT / rel), "bytes": (ROOT / rel).stat().st_size}
                   for rel in (RECALL_ARM, PRECISION_ARM, TEMPLATE, LABELS)},
        "population_definitions": {
            "L": "third-party held-out-segment protocol (gems-eval)",
            "S": "footprint minus the supplied catalogue; truth = held-out segments",
            "I": f"S with truth restricted to held-out segments more than {INDEPENDENT_PX:g} px "
                 "from the training catalogue",
            "P_platform_mask": "S with the prediction masked on the supplied catalogue, i.e. the "
                               "organizers' pixel-exact mask applied",
        },
        "measurement": measured,
        "conformance": {"stats": conform_stats,
                        "checks": "single band, float32, EPSG:32611, template transform and "
                                  "shape, finite in [0,1] inside, NaN outside, GDAL_NODATA = NaN"},
    }
    if not args.check:
        out_dir = ROOT / OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(ROOT / TEMPLATE) as tmpl:
            profile = tmpl.profile.copy()
        profile.update(dtype="float32", count=1, nodata=float("nan"), compress="deflate")
        out_path = out_dir / "submission.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(conformed, 1)
        # read the bytes back and compare the *field*: NaNs are not equal to themselves under
        # np.array_equal, so the NaN masks are compared explicitly (the first version of this check
        # reported False on a perfectly conformant file).
        readback = read(out_path)
        nan_same = bool((np.isnan(readback) == np.isnan(conformed)).all())
        values_same = bool(np.array_equal(np.nan_to_num(readback, nan=-1.0),
                                          np.nan_to_num(conformed, nan=-1.0)))
        sha = sha256_file(out_path)
        report["written"] = {
            "path": str(out_path.relative_to(ROOT)), "bytes": out_path.stat().st_size,
            "sha256": sha, "emitted_px": int((conformed > 0).sum()),
            "readback_identical_to_written": bool(nan_same and values_same),
            "readback_nan_mask_identical": nan_same, "readback_values_identical": values_same,
        }
        (out_dir / "submission.tif.sha256").write_text(f"{sha}  submission.tif\n")
        print(f"\nwrote {out_path.relative_to(ROOT)}  {out_path.stat().st_size:,} B  "
              f"sha256 {sha[:12]}...  emitted {(conformed > 0).sum():,} px")
        print(f"conformance: filled_inside {conform_stats.get('filled_inside')} "
              f"masked_outside {conform_stats.get('masked_outside')} "
              f"clipped {conform_stats.get('clipped')}")
    out_json = ROOT / args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
