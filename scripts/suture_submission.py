#!/usr/bin/env python3
"""SUTURE v1 - a DTI-native submission built from structure-tensor lineament response.

WHAT IS NEW HERE
----------------
Every candidate this repository has shipped so far was *shaped* by a policy picked on the
repository's own surrogate populations (the SGMC proxy catalogue, the supplied labels, or their
union).  None of them was chosen under the scoring rule the organizers actually stated.  On
2026-09-21 the organizer answered exactly that question on the forum:

    "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation,
     so they do not count towards penalty terms."                       -- chrisk-dd, forum 11516 #2
    "The mask is indeed pixel-exact - it is identical to the provided set of training fault
     labels."                                                           -- chrisk-dd, forum 11516 #4
    "A predicted pixel that is near a known fault trace but far from a new-fault ground truth
     pixel will be fully penalized, i.e., the buffer does not apply to known faults."
                                                                        -- chrisk-dd, forum 11516 #4

So the false-positive term is charged over footprint pixels that are not themselves supplied-label
pixels, and a prediction that hugs the supplied catalogue pays full price unless new truth is
there.  This script builds a field for that scoring rule and *selects its emission policy under
it*, on a held-out half of the catalogue's own segments, then measures the chosen policy once on
the other half.

THE FIELD
---------
Two independent components, fused by rank (scale-free, so no arbitrary weights):

 1. ALR - anisotropic lineament response.  For each geophysical band u, the structure tensor
    J = G_sigma * (grad u  (x)  grad u) has eigenvalues l1 >= l2.  `l2` is large exactly where the
    band is line-like rather than blob-like or flat (Bigun & Granlund 1987; Jahne 1993), and the
    coherence (l1 - l2) / (l1 + l2) says how one-dimensional the local structure is.  Faults are
    linear geophysical boundaries, so l2 * coherence is the physically motivated response.
    Computed at three scales (2, 4, 8 px = 200 m / 400 m / 800 m) because the fault systems in the
    GeoDAWN region range from short splays to long range-front structures.
 2. The shipped deep-ensemble field (`data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`),
    which is the repository's best-measured detector of *unmapped* structure and contributes recall
    that a purely classical edge detector does not have.

THE EMISSION
------------
The metric rewards thin, sparse, line-like predictions: TP_w saturates per truth pixel while every
off-line pixel is charged at alpha = 0.2 (see src/metrics.py and the problem page's worked
example).  So the fused field is skeletonised (one-pixel lines, value 1.0) at a support fraction
swept over a small pre-registered grid, and the winner is chosen on the tune half.  The chosen
policy is then scored once, unchanged, on the measure half.

WHAT IT DOES NOT CLAIM
----------------------
The held-out segments are pixels of the *supplied* catalogue, so any local score here is an
imitation of the board, not the board.  A field that re-draws the catalogue scores well here and
poorly on the leaderboard; that is why the organizers' mask is applied (predicted mass next to the
training catalogue is charged) and why the report prints the emitted-pixel count beside every
score.  Only DrivenData can say whether the file beats 0.3049.

USAGE
    python scripts/suture_submission.py --out-dir data/evidence/suture
    python scripts/suture_submission.py --supports 0.004,0.008,0.015 --no-ensemble
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

from scripts.segment_holdout_eval import (build_split, dti,  # noqa: E402
                                         skeleton_top_fraction)

# Bands whose lineament response is physically meaningful for fault mapping, by category.
DEFAULT_BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "tmi_vg", "tc", "tmi",              # magnetics
    "iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg",          # gravity
    "det_elev_slope", "cond_surf", "depth_to_base_surf",              # topography / subsurface
]
ENSEMBLE_FIELD = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
R_PX = 3.0
ALPHA, BETA = 0.2, 0.8


# --------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_band(features: Path, index: int) -> np.ndarray:
    with rasterio.open(features) as src:
        return src.read(index).astype(np.float32)


def band_index_map(features: Path) -> dict[str, int]:
    with rasterio.open(features) as src:
        return {src.tags(i).get("band_name", f"band{i}"): i for i in range(1, src.count + 1)}


def lineament_response(band: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Structure-tensor line response and coherence for one band at one scale.

    Returns (l2 * coherence, coherence), both zero wherever the band is not finite.
    """
    finite = np.isfinite(band)
    if finite.sum() < 100:
        z = np.zeros(band.shape, np.float32)
        return z, z.copy()
    u = np.where(finite, band, 0.0).astype(np.float32)
    # standardise (median / MAD, robust to the long-tailed geophysical distributions) so the
    # structure-tensor products stay in float32 range -- without this the gradient products
    # overflow for bands whose raw units are large.
    med = float(np.median(u[finite]))
    mad = float(np.median(np.abs(u[finite] - med))) * 1.4826
    u = np.clip((u - med) / (mad if mad > 1e-12 else 1.0), -10.0, 10.0).astype(np.float32)
    s = ndimage.gaussian_filter(u, sigma=sigma, mode="nearest")
    gy, gx = np.gradient(s)
    # structure tensor components, smoothed by the same scale
    jxx = ndimage.gaussian_filter(gx * gx, sigma=sigma, mode="nearest")
    jyy = ndimage.gaussian_filter(gy * gy, sigma=sigma, mode="nearest")
    jxy = ndimage.gaussian_filter(gx * gy, sigma=sigma, mode="nearest")
    del s, gy, gx
    tr = jxx + jyy
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy**2, 0.0)).astype(np.float32)
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    del jxx, jyy, jxy
    coh = np.where(l1 > 1e-12, (l1 - l2) / np.maximum(l1, 1e-12), 0.0).astype(np.float32)
    resp = (l2 * coh).astype(np.float32)
    del l1, tr
    resp[~finite] = 0.0
    coh[~finite] = 0.0
    return resp, coh


def rank_normalise(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map the values inside `mask` onto their rank in [0, 1]; 0 elsewhere."""
    out = np.zeros(x.shape, np.float32)
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return out
    v = x.ravel()[idx]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(order.size, np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    out.ravel()[idx] = (ranks / max(order.size - 1, 1)).astype(np.float32)
    return out


def build_field(features: Path, band_names, sigmas, ensemble: Path | None,
                template: Path | None = None) -> tuple[np.ndarray, dict, dict, np.ndarray]:
    with rasterio.open(features) as src:
        shape = (src.height, src.width)
        profile = src.profile.copy()
        index_of = {src.tags(i).get("band_name", f"band{i}"): i for i in range(1, src.count + 1)}
    # The scored region is the submission template's valid region (the problem page: "same bounds
    # as the training data, and data outside the bounds is null or nan").  A band's -3.4e38
    # nodata sentinel is a *finite* float, so `isfinite` is the wrong test for it.
    if template is not None and Path(template).exists():
        with rasterio.open(template) as src:
            foot_src = src.read(1)
        footprint = (np.isfinite(foot_src) if np.isnan(src.nodata if src.nodata is not None
                                                       else np.nan)
                     else np.abs(foot_src) < 1e30)
    else:
        with rasterio.open(features) as src:
            footprint = np.abs(src.read(1)) < 1e30
    used = [b for b in band_names if b in index_of]
    missing = [b for b in band_names if b not in index_of]
    acc_resp = np.zeros(shape, np.float32)
    n_terms = 0
    per_band = []
    for name in used:
        band = read_band(features, index_of[name])
        # every band carries its own nodata (-3.4e38); inside band 1's footprint a different band
        # can still be nodata, and those sentinels overflow the structure tensor
        band = np.where(np.abs(band) > 1e30, np.nan, band)
        band[~footprint] = np.nan
        sub = np.zeros(shape, np.float32)
        coh_max = np.zeros(shape, np.float32)
        for sigma in sigmas:
            resp, coh = lineament_response(band, float(sigma))
            # normalise each scale by a robust high quantile so no single scale dominates
            q = float(np.quantile(resp[footprint], 0.999)) if footprint.any() else 0.0
            if q > 0:
                sub += np.clip(resp / q, 0.0, 1.0)
            coh_max = np.maximum(coh_max, coh)
            del resp
            n_terms += 1
        sub /= max(len(sigmas), 1)
        sub *= coh_max
        acc_resp += sub
        per_band.append({"band": name, "scale_sum": float(sub[footprint].mean())})
        del band, sub, coh_max
    if n_terms == 0 and (ensemble is None or not Path(ensemble).exists()):
        sys.exit("FAIL: no requested bands were found in the feature stack and no ensemble given")
    lineament = acc_resp / max(len(used), 1)
    del acc_resp

    parts = {} if len(used) == 0 else {"lineament": lineament}
    if ensemble is not None and Path(ensemble).exists():
        with rasterio.open(ensemble) as src:
            ens = src.read(1).astype(np.float32)
        parts["ensemble"] = np.where(np.isfinite(ens), ens, 0.0)
        del ens

    fused = np.zeros(shape, np.float32)
    for name, arr in parts.items():
        fused += rank_normalise(arr, footprint) / len(parts)
    fused[~footprint] = np.nan
    meta = {
        "bands_used": used, "bands_missing": missing, "sigmas": list(sigmas),
        "components": list(parts.keys()), "per_band_mean_scaled_response": per_band,
        "footprint_px": int(footprint.sum()),
        "template": str(template) if template else None,
    }
    del parts
    return fused, meta, profile, footprint


# --------------------------------------------------------------------------------------
def policy_scores(field: np.ndarray, footprint: np.ndarray, train_known: np.ndarray,
                  truth_mask: np.ndarray, supports) -> list[dict]:
    """Skeletonise the field at each support fraction and score it under the organizers' rule.

    Scored population = footprint minus the supplied catalogue; truth = the held-out segments
    given in `truth_mask` (already restricted to those far from the training catalogue).
    """
    valid_S = footprint & ~train_known
    rows = []
    for s in supports:
        thin = skeleton_top_fraction(field, float(s), valid=footprint)
        p = np.where(np.isfinite(thin), thin, 0.0).astype(np.float32)
        score, terms = dti(p, truth_mask, valid=valid_S, return_terms=True)
        # the same emission scored with the field's own probabilities (no skeleton)
        rows.append({
            "support_fraction": float(s),
            "emitted_px": int((p[valid_S] > 0).sum()),
            "emitted_fraction_of_footprint": float((p[valid_S] > 0).sum() / max(footprint.sum(), 1)),
            "dti_S": score,
            "tp_w": terms["tp"], "fp_w": terms["fp"], "fn_w": terms["fn"],
            "n_truth": terms["n_truth"],
        })
        del thin, p
    return rows


def write_submission(field: np.ndarray, footprint: np.ndarray, out: Path, profile: dict) -> str:
    """Write the file exactly as the problem page's submission-format bullet list describes.

    "same bounds as the training data, and data outside the bounds is null or nan"
    "a single layer with datatype of 32-bit float (float32) with values between 0 and 1"
    """
    arr = np.where(footprint, np.clip(np.nan_to_num(field, nan=0.0), 0.0, 1.0), np.nan).astype(np.float32)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", crs=profile["crs"], transform=profile["transform"],
                nodata=np.nan, compress="deflate", tiled=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr, 1)
    return sha256_file(out)


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/training_features.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--split-dir", default="data/evidence/seghold")
    ap.add_argument("--ensemble", default=ENSEMBLE_FIELD)
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--no-lineament", action="store_true",
                    help="skip the structure-tensor component (ablation: ensemble only)")
    ap.add_argument("--sigma", default="2,4,8")
    ap.add_argument("--bands", default=",".join(DEFAULT_BAND_NAMES))
    ap.add_argument("--supports", default="0.002,0.004,0.008,0.015,0.025,0.04,0.06")
    ap.add_argument("--out-dir", default="data/evidence/suture")
    ap.add_argument("--out-name", default="submission.tif",
                    help="file name under --out-dir (keep ablations side by side)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--independent-px", type=float, default=5.0,
                    help="truth pixels within this distance of the training catalogue are "
                         "excluded from selection (the label-echo path)")
    args = ap.parse_args(argv)

    t0 = time.time()
    features, labels = Path(args.features), Path(args.labels)
    for p in (features, labels):
        if not p.exists():
            print(f"FAIL: {p} missing - run python scripts/assemble_data_bridge.py", file=sys.stderr)
            return 2

    # 1. held-out segments, split once, cached
    split_dir = Path(args.split_dir)
    if not (split_dir / "eval_labels.tif").exists():
        build_split(labels, split_dir, 0.3, args.seed, 3)
    with rasterio.open(split_dir / "eval_labels.tif") as s:
        ev = s.read(1)
    with rasterio.open(split_dir / "train_labels.tif") as s:
        train_known = s.read(1) == 1
    footprint = ev >= 0

    # tune / measure halves of the held-out segments, disjoint by construction (component parity)
    comp, n_comp = ndimage.label(ev == 1, structure=np.ones((3, 3)))
    ids = np.arange(1, n_comp + 1)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    tune_ids = set(ids[: len(ids) // 2].tolist())
    dist_to_train = ndimage.distance_transform_edt(~train_known).astype(np.float32)
    independent = dist_to_train > args.independent_px
    tune_truth = np.isin(comp, list(tune_ids)) & (ev == 1) & independent
    measure_truth = (ev == 1) & ~tune_truth & independent
    dropped = int(((ev == 1) & ~independent).sum())
    print(f"split: {n_comp} held-out segments -> tune {int(tune_truth.sum())} px / "
          f"measure {int(measure_truth.sum())} px  ({dropped} px dropped: within "
          f"{args.independent_px:g} px of the training catalogue)")

    # 2. the field
    sigmas = [float(x) for x in args.sigma.split(",") if x.strip()]
    band_names = [] if args.no_lineament else [b for b in args.bands.split(",") if b.strip()]
    ens = None if args.no_ensemble else Path(args.ensemble)
    field, meta, profile, footprint = build_field(features, band_names, sigmas, ens,
                                                  Path(args.template))
    meta["ensemble_fused"] = bool(ens is not None and Path(ens).exists())
    print(f"field built in {time.time() - t0:.0f}s from {len(meta['bands_used'])} bands "
          f"x {len(sigmas)} scales")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "field_raw.tif"
    with rasterio.open(raw_path, "w", driver="GTiff", height=field.shape[0], width=field.shape[1],
                       count=1, dtype="float32", crs=profile["crs"],
                       transform=profile["transform"], nodata=np.nan, compress="deflate") as dst:
        dst.write(field.astype(np.float32), 1)

    # 3. sweep on the tune half, then one measurement on the untouched half
    supports = [float(x) for x in args.supports.split(",") if x.strip()]
    tune_rows = policy_scores(field, footprint, train_known, tune_truth, supports)
    for row in tune_rows:
        print(f"  tune s={row['support_fraction']:<6g} emitted {row['emitted_px']:>7,d} "
              f"({100 * row['emitted_fraction_of_footprint']:.2f}%)  DTI_S {row['dti_S']:.6f}")
    winner = max(tune_rows, key=lambda r: r["dti_S"])
    measure = policy_scores(field, footprint, train_known, measure_truth,
                            [winner["support_fraction"]])[0]
    print(f"  winner s={winner['support_fraction']:g} -> measure-half DTI_S {measure['dti_S']:.6f}")

    thin = skeleton_top_fraction(field, winner["support_fraction"], valid=footprint)
    submission = out_dir / args.out_name
    sha = write_submission(thin, footprint, submission, profile)
    bytes_ = submission.stat().st_size

    generated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stamp = generated_utc.replace("-", "").replace(":", "")
    unique_name = f"suture-v1-{stamp}-{sha[:8]}.tif"
    report = {
        "strategy": "SUTURE v1 - structure-tensor lineament response fused with the shipped "
                    "deep ensemble, skeletonised, policy selected under the organizers' rule",
        "generated_utc": generated_utc,
        "inputs": {
            "features": str(features), "features_sha256": sha256_file(features),
            "labels": str(labels), "labels_sha256": sha256_file(labels),
            "ensemble": str(ens) if ens else None,
            "ensemble_sha256": sha256_file(ens) if (ens and Path(ens).exists()) else None,
        },
        "field": meta,
        "split": {"frac": 0.3, "seed": args.seed, "buffer_px": 3,
                  "independent_px": args.independent_px,
                  "truth_dropped_within_independent_px": dropped,
                  "held_out_px": int((ev == 1).sum()),
                  "tune_truth_px": int(tune_truth.sum()),
                  "measure_truth_px": int(measure_truth.sum())},
        "policy_selection": {
            "population": ("organizer-faithful: footprint minus supplied-catalogue pixels; "
                           "truth = held-out segments (forum 11516 #2/#4)"),
            "tuned_on": "tune half of the held-out segments",
            "measured_on": "the other half, scored once",
            "sweep": tune_rows, "winner": winner, "measurement_on_untouched_half": measure,
        },
        "submission": {
            "path": str(submission), "sha256": sha, "bytes": bytes_,
            "support_fraction": winner["support_fraction"],
            "emitted_px": winner["emitted_px"],
            "unique_name": unique_name,
            "note": f"SUTURE v1 | structure-tensor lineaments + ensemble | "
                    f"s={winner['support_fraction']:.3f} | build {sha[:8]}",
            "format": {"crs": profile["crs"].to_string(), "dtype": "float32",
                       "nodata": "nan (outside the footprint)", "values": "[0, 1]"},
        },
        "honesty": ("Local numbers are an imitation of the board: the held-out segments are pixels "
                    "of the supplied catalogue.  Only DrivenData can score against the private "
                    "new-fault set; no claim of beating 0.3049 is made here."),
    }
    (out_dir / (Path(args.out_name).stem + "_report.json")).write_text(json.dumps(report, indent=1))
    (out_dir / (Path(args.out_name).stem + ".sha256")).write_text(f"{sha}  {submission}\n")
    print(f"wrote {submission} ({bytes_} B, sha256 {sha[:12]}...)")
    print(f"wrote {out_dir / (Path(args.out_name).stem + '_report.json')}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
