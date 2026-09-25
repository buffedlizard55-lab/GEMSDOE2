#!/usr/bin/env python3
"""Score a prediction against the proxy catalogue - the closest available stand-in for the scored
"new fault" population.

WHY THIS EXISTS
---------------
Every number this repository has published about submission *quality* was measured against
`labels.tif`, i.e. the faults the catalogue already contains.  The rules say both prize phases score
faults it does NOT contain (§1.1, §3.3; verbatim quotes in `scripts/verify_rules_quotes.py`).  A
metric on the wrong universe cannot choose between a model that generalises and one that memorises.

`scripts/fetch_proxy_faults.py` + `scripts/build_proxy_catalogue.py` build an independent fault set
(USGS SGMC, Horton et al. 2017, DOI 10.3133/ds1052) rasterised on the exact competition grid, split
into "the labels already contain it" (code 1) and "the labels do not" (code 2).  This script scores
against the code-2 pixels with the OFFICIAL metric - same R, alpha, beta, kernel, same call into
`src.metrics.GtContext` that the training loop and the submission calibration use.

WHAT THE NUMBER MEANS, AND WHAT IT DOES NOT
-------------------------------------------
* It is a *population* measurement on real mapped faults absent from the training labels.  A model
  that reproduces the catalogue scores exactly 0 on it, because code-2 pixels are >R from every
  labelled pixel (that identity is asserted here, not assumed).
* It is NOT the competition metric: the scored faults were drawn by experts from the GeoDAWN
  geophysics; SGMC faults were drawn by state-map geologists from surface mapping.  Neither
  population contains the other.  Use it to compare *policies* (models, floors, emission widths),
  never to predict a leaderboard position.
* The FP term is a global sum, so a prediction with mass far from the proxy faults is penalised
  exactly as the official metric would penalise it.  Nothing is cropped by default.

USAGE
    python scripts/eval_proxy_catalogue.py --pred submission.tif \
        --proxy data/evidence/proxy/proxy_catalogue.tif --labels data/labels.tif \
        --out data/evidence/proxy/eval_submission.json
    # policy sweep (floor x thinning x emission width) on an ensemble probability raster:
    python scripts/eval_proxy_catalogue.py --pred ensemble_mean.tif --sweep --shaping-grid 5 \
        --dilate-grid 0,1,2,3,4,6 --out data/evidence/proxy/eval_sweep.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import GtContext                                             # noqa: E402
from src.submission_optim import (floor_sharpen, optimize_submission,        # noqa: E402
                                 shaping_thresholds)

CODE_NEAR, CODE_ONLY = 1, 2
DEFAULTS = dict(R=3, alpha=0.2, beta=0.8, eps=1e-7)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_grid(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        meta = {"width": src.width, "height": src.height, "crs": str(src.crs),
                "transform": [round(v, 6) for v in list(src.transform)[:6]],
                "dtype": src.dtypes[0], "nodata": src.nodata}
    return arr, meta


def same_grid(a: dict, b: dict) -> bool:
    return (a["width"], a["height"]) == (b["width"], b["height"]) and a["crs"] == b["crs"]


def score(pred: np.ndarray, ctx: GtContext, R: int, alpha: float, beta: float, eps: float) -> dict:
    dti, (tp, fp, fn) = ctx.score(pred, alpha=alpha, beta=beta, eps=eps, return_components=True)
    return {"dti": round(float(dti), 6), "TP_w": round(float(tp), 3), "FP_w": round(float(fp), 3),
            "FN_w": round(float(fn), 3), "mass": round(float(np.nansum(pred)), 1),
            "emission_px": int(np.count_nonzero(np.nan_to_num(pred)))}


ALPHA_DEFAULT, BETA_DEFAULT = DEFAULTS["alpha"], DEFAULTS["beta"]


def project_dti(coverage: float, fp_w: float, truth_px: float,
                alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT) -> float:
    """DTI for a HYPOTHETICAL scored truth of `truth_px` pixels, from a measured policy.

    The published caveat on every number in this repository is that an absolute DTI is a monitor,
    not a result, because |G| (the hidden truth's size) is unknown.  This turns that caveat into a
    calculation.  For a policy measured on the proxy population we know

        coverage = TP_w / |G_proxy|          (the fraction of the truth the prediction reaches)
        FP_w                                 (false-positive mass, in metric units)

    and the metric is  DTI = TP_w / (alpha*(TP_w + FP_w) + beta*|G|).  Holding the coverage and the
    false-positive mass fixed and letting |G| vary gives the projected score for a scored set of any
    size:

        DTI(|G|) = coverage*|G| / (alpha*(coverage*|G| + FP_w) + beta*|G|)

    At |G| = |G_proxy| this is algebraically identical to the measured value (asserted in the tests),
    so the projection only adds one explicit assumption: that a different-sized truth is covered at
    the same fractional rate and that the wrong-mass stays where it is.  It is the assumption a
    policy decision has to make out loud, because the two error terms scale differently - FP_w is
    set by the prediction, beta*|G| by the hidden label set - and that is exactly why a policy that
    wins on a 6,166 km proxy truth need not win on a smaller scored one.
    """
    tp = coverage * float(truth_px)
    return float(tp / (alpha * (tp + fp_w) + beta * float(truth_px) + 1e-12))


def sensitivity_table(sweep_table: list[dict], n_truth: int, sizes: list[float],
                      alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                      tol: float = 0.01) -> tuple[list[dict], dict]:
    """Project every swept policy onto a range of possible scored-truth sizes."""
    rows: list[dict] = []
    for r in sweep_table:
        cov = float(r["TP_w"]) / float(n_truth)
        rows.append({"t0": r["t0"], "thin": r["thin"], "dilate": r["dilate"],
                     "coverage_fraction": round(cov, 6), "FP_w": r["FP_w"],
                     "projected_dti": {str(int(g)): round(project_dti(cov, r["FP_w"], g, alpha, beta), 6)
                                       for g in sizes}})
    per_size = []
    for g in sizes:
        key = str(int(g))
        best = max(rows, key=lambda r: r["projected_dti"][key])
        skel = max((r for r in rows if r["dilate"] == 0), key=lambda r: r["projected_dti"][key])
        gain = best["projected_dti"][key] - skel["projected_dti"][key]
        per_size.append({"truth_px": int(g), "best_t0": best["t0"], "best_thin": best["thin"],
                         "best_dilate": best["dilate"], "best_dti": best["projected_dti"][key],
                         "skeleton_dti": skel["projected_dti"][key],
                         "best_is_wider_than_skeleton": bool(best["dilate"] > 0),
                         "gain_over_skeleton": round(gain, 6),
                         "passes_acceptance": bool(best["dilate"] > 0 and gain > tol)})
    verdict = {
        "sizes_px": [int(g) for g in sizes],
        "sizes_where_a_wider_band_passes": [p["truth_px"] for p in per_size
                                            if p["passes_acceptance"]],
        "sizes_where_the_skeleton_wins": [p["truth_px"] for p in per_size
                                          if not p["best_is_wider_than_skeleton"]],
        "acceptance_rule": ("a wider band is only worth changing the default for if it beats the "
                            "skeleton by more than %.2f at a truth size that is plausible for the "
                            "scored set - the proxy truth is not that size" % tol),
    }
    return rows, {"rows": rows, "per_size": per_size, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="prediction raster (submission.tif, or a probability map)")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--labels", default=None, help="known-fault raster (needed for --truth combined)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", choices=["only", "all", "combined"], default="only",
                    help="only = proxy faults absent from the catalogue (default); all = every proxy "
                         "fault; combined = proxy + labelled faults")
    ap.add_argument("--sweep", action="store_true",
                    help="also search floor x thinning x emission width on THIS population")
    ap.add_argument("--shaping-grid", type=int, default=5,
                    help="number of log-spaced floor candidates (ignored when --shaping-values "
                         "is given)")
    ap.add_argument("--shaping-values", default=None,
                    help="EXPLICIT comma-separated floor candidates, e.g. '0,0.02,0.05,0.1,0.2'. "
                         "Why this exists: shaping_thresholds() is log-spaced, so its spacing is "
                         "set by the RANGE (1e-4..0.9), not by where the field's mass actually "
                         "crosses a floor. The field's own mass profile is printed and recorded "
                         "now (field_mass_profile), so the choice of grid is auditable.")
    ap.add_argument("--dilate-grid", default="0,1,2,3,4,6")
    ap.add_argument("--soft-band", action="store_true",
                    help="also score RAMP emission candidates: the same support as the hard band, "
                         "with values decaying linearly from 1 on the skeleton to 0 at the band "
                         "edge (src.submission_optim.soft_band). Every emitted pixel beyond R "
                         "costs 0.2 per unit and a hard band charges the far ring full price.")
    ap.add_argument("--band-gammas", default="1,2",
                    help="exponents for the ramp values, applied in addition to gamma=1")
    ap.add_argument("--band-floors", default=None,
                    help="floors at which the ramp candidates are scored (default: the reference "
                         "floor, i.e. the shipped policy's own floor)")
    ap.add_argument("--thin-off", action="store_true", help="include thin=False rows in the sweep")
    ap.add_argument("--reference-t0", type=float, default=None,
                    help="the floor the SHIPPED policy uses: added to the sweep grid (the log-spaced "
                         "grid can jump straight past the operating region) and used as the "
                         "reference the acceptance rule compares against")
    ap.add_argument("--reference-report", default=None,
                    help="blend_report.json to read the shipped floor from, when --reference-t0 is "
                         "not given")
    ap.add_argument("--R", type=int, default=DEFAULTS["R"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    ap.add_argument("--eps", type=float, default=DEFAULTS["eps"])
    ap.add_argument("--sens-sizes", default="500,1000,2500,5000,10000,20000,50000,100000,250000",
                    help="assumed sizes (px) of the HIDDEN scored truth, for the projection table")
    a = ap.parse_args()

    # The shipped floor, when it can be determined.  WHY THIS EXISTS (measured, session 10): the
    # committed proxy sweep of ensemble 1 evaluated floors [0, 1e-4, 2.08e-3, 4.33e-2, 0.9] because
    # shaping_thresholds() is log-spaced - on that field the first four rows are the SAME mask (the
    # model emits nothing between 0.043 and 0.9), so the floor dimension was effectively binary and
    # the shipped floor (0.469674) was never scored.  The sweep's acceptance rule then compared
    # candidates against the t0=0 skeleton (0.0144) rather than against the shipped policy (0.0247
    # on the same population).  Both are fixed by scoring the shipped floor explicitly.
    reference_t0, reference_src = a.reference_t0, ("--reference-t0" if a.reference_t0 is not None else None)
    if reference_t0 is None and a.reference_report:
        rep_in = json.loads(Path(a.reference_report).read_text())
        t0_in = rep_in.get("shaping", {}).get("t0")
        if t0_in is None:
            raise SystemExit(f"{a.reference_report} has no shaping.t0 (unshaped run) - "
                             "pass --reference-t0 explicitly")
        reference_t0, reference_src = float(t0_in), f"{a.reference_report}:shaping.t0"

    pred, pmeta = read_grid(Path(a.pred))
    coded, cmeta = read_grid(Path(a.proxy))
    if coded.dtype != np.uint8 and coded.max() <= 3:
        coded = coded.astype("uint8")
    if not same_grid(pmeta, cmeta):
        raise SystemExit(f"grid mismatch: {a.pred} {pmeta} vs {a.proxy} {cmeta}")

    labels = None
    if a.labels:
        labels, lmeta = read_grid(Path(a.labels))
        if not same_grid(lmeta, cmeta):
            raise SystemExit(f"grid mismatch: {a.labels} vs {a.proxy}")

    near, only = coded == CODE_NEAR, coded == CODE_ONLY
    if a.truth == "only":
        truth = only
    elif a.truth == "all":
        truth = (coded > 0)
    else:
        if labels is None:
            raise SystemExit("--truth combined needs --labels")
        truth = (coded > 0) | (labels > 0.5)
    n_truth = int(truth.sum())
    if n_truth == 0:
        raise SystemExit("empty truth set - refusing to produce a number from it")

    ctx = GtContext(truth.astype("float32"), R_pixels=a.R)
    provided = score(pred, ctx, a.R, a.alpha, a.beta, a.eps)
    res: dict = {
        # The raster handed to --pred, scored exactly as it is (no shaping applied here).
        "as_provided": provided,
        # Deprecated alias kept so evidence written before 2026-09-16 still renders.  The old name
        # called every --pred input "the submission", which is true only when --pred IS the
        # committed submission - it was NOT true in the shaping sweep, where --pred was the
        # pre-shaping ensemble map, and the site displayed that row under the heading "this
        # submission, as committed" (2026-09-16, session 7).
        "as_submitted": provided,
        "prediction_role": (
            "the raster at --pred as provided: equals the committed submission only when --pred "
            "points at one; check inputs.pred_sha256 before quoting this as a submission score"),
    }

    # ---- baselines: each one answers "could this number be earned trivially?" -------------------
    # The support for the blanket baseline must NOT depend on the prediction: `np.isfinite(pred)`
    # made the baseline's meaning change with whatever was scored (a full-grid ensemble map gave a
    # bigger blanket than a NaN-clipped submission), so two runs were not comparable.  Use the
    # competition footprint instead - the finite area of the label raster, which is the raster the
    # submission's bounds must match (rules section 3.6.2) - and record which source was used.
    # Order matters: the label raster carries the competition footprint.  The PROXY raster is
    # finite over the whole grid (measured: 12,279,160 of 12,279,160 px), so it is never a
    # footprint source; with no label raster the only legal-submission support available is the
    # prediction's own finite area, and the source string says so.
    if labels is not None and np.isfinite(labels).any():
        support, support_src = np.isfinite(labels), "labels (the competition footprint)"
    elif np.isfinite(pred).any():
        support, support_src = np.isfinite(pred), \
            "the prediction's finite area (no label raster given, so the footprint is unknown)"
    else:
        support, support_src = np.ones_like(pred, dtype=bool), "whole grid (nothing is masked)"
    baselines = {"zeros": np.zeros_like(pred),
                 "blanket_ones": support.astype("float32"),
                 # Over the WHOLE grid, i.e. also outside the data footprint, where a real
                 # submission must be NaN: included so the size of that penalty is visible.
                 "blanket_ones_whole_grid": np.ones_like(pred, dtype="float32")}
    if labels is not None:
        lab = np.where(np.isfinite(labels), labels, 0.0).astype("float32")
        baselines["catalogue_copy"] = (lab > 0.5).astype("float32")
        baselines["catalogue_plus_submission"] = np.maximum((lab > 0.5).astype("float32"),
                                                            (np.nan_to_num(pred) > 0).astype("float32"))
    baselines["submission_skeleton_dilated_6px"] = None      # filled by the sweep when requested
    res["baselines"] = {k: score(v, ctx, a.R, a.alpha, a.beta, a.eps)
                        for k, v in baselines.items() if v is not None}

    # A LEGAL submission is NaN outside the data footprint (rules section 3.6.2).  A raw ensemble
    # map is not, so scoring it as-is charges it for mass a submission could not legally carry -
    # and charges the sweep's blanket baseline for the whole grid.  Score the clipped version too
    # whenever the two differ, so the policy comparison is made between legal submissions.
    outside = support & ~np.isfinite(pred)
    mass_outside = float(np.nansum(np.where(support, 0.0, pred)))
    if mass_outside > 0:
        res["as_provided_clipped_to_footprint"] = score(
            np.where(support, pred, 0.0), ctx, a.R, a.alpha, a.beta, a.eps)
        res["footprint_note"] = (
            "the raster at --pred carries %.1f of mass outside the data footprint, which a legal "
            "submission may not; 'as_provided_clipped_to_footprint' is the legal version of the "
            "same map, and the sweep candidates are clipped before scoring for the same reason"
            % mass_outside)

    # Acceptance property, asserted rather than assumed: on the proxy-only truth a perfect
    # reproduction of the catalogue must score EXACTLY zero (every truth pixel is >R from a label).
    if "catalogue_copy" in res["baselines"]:
        cc = res["baselines"]["catalogue_copy"]["dti"]
        res["acceptance"] = {
            "catalogue_copy_dti_on_proxy_only": cc,
            "expected": 0.0,
            "passed": bool(abs(cc) < 1e-9),
            "meaning": ("a catalogue-copy submission earns nothing here, so this population is not "
                        "a restatement of the training labels"),
        }
        if a.truth == "only" and not res["acceptance"]["passed"]:
            print("FAIL: catalogue copy scored non-zero on the proxy-only truth - the proxy or the "
                  "grid alignment is broken; the number must not be used", file=sys.stderr)

    # ---- policy sweep: does any shaping beat the current one ON THIS POPULATION? ----------------
    #
    # TWO AXES, MEASURED SEPARATELY (2026-09-17, session 11).
    #
    # axis 1 - the SUPPORT: floor t0 x thinning x band width (the original sweep).
    # axis 2 - the VALUES on that support: hard (1.0) vs the distance ramp of
    #          src.submission_optim.soft_band, which has the SAME support as the hard band of the
    #          same width and differs only in what is written into it.  Scoring both against the
    #          same truth isolates the ramp from any re-ranking of the field.
    #
    # The floor axis also carries the measurement that decides whether it is worth widening the
    # grid at all: `field_mass_profile` below records, per floor, how many pixels the field has
    # ABOVE that floor BEFORE thinning.  Session 10 found the first ensemble's floor dimension was
    # effectively binary (nothing emitted between 0.043 and 0.9), which no amount of grid
    # resolution can fix - so the fact is now printed instead of being inferred from identical rows.
    sweep_table: list[dict] = []
    if a.sweep:
        dilates = tuple(sorted({int(v) for v in str(a.dilate_grid).split(",") if v.strip()}))
        if a.shaping_values:
            floors = [float(v) for v in str(a.shaping_values).split(",") if v.strip()]
            print(f"  floor grid: explicit --shaping-values ({len(floors)} candidates)")
        else:
            floors = [float(v) for v in shaping_thresholds(a.shaping_grid)]
        if reference_t0 is not None and not any(abs(f - reference_t0) < 1e-9 for f in floors):
            floors.append(float(reference_t0))
        floors = sorted(set(floors))

        # how much of the field survives each floor, before any thinning: 0 above means the floor
        # is indistinguishable from every lower floor and the grid is not the constraint.
        profile = []
        for t0 in floors:
            above = int(np.count_nonzero(floor_sharpen(pred, t0=float(t0), hard=True)))
            profile.append({"t0": float(t0), "support_px_above_floor": above,
                            "mass_above_floor": round(float(np.nansum(
                                np.where(np.nan_to_num(pred) > t0, np.nan_to_num(pred), 0.0))), 3)})
        res["field_mass_profile"] = {
            "rows": profile,
            "distinct_supports": len({p["support_px_above_floor"] for p in profile}),
            "reading": ("support_px_above_floor is the floor's effect on the emitted set BEFORE "
                        "thinning; identical values across floors mean the floor axis is binary on "
                        "this field and a finer floor grid cannot change any emission"),
        }

        def _emit(row: dict, label: str) -> None:
            q = optimize_submission(pred, R=a.R, t0=float(row["t0"]), thin=bool(row["thin"]),
                                    hard=True, gamma=float(row.get("gamma", 1.0)),
                                    dilate=int(row["dilate"]), soft=bool(row.get("soft", False)))
            q = np.where(support, q, np.nan).astype(np.float32)      # legal submission
            row.update(score(q, ctx, a.R, a.alpha, a.beta, a.eps))
            row["mean_kept_px"] = int(np.count_nonzero(q))
            sweep_table.append(row)
            print(f"  {label:<42} -> proxy DTI {row['dti']:.4f}  kept {row['mean_kept_px']} px")

        for t0 in floors:
            for thin in ((False, True) if a.thin_off else (True,)):
                for d in (dilates if thin else (0,)):
                    _emit({"t0": float(t0), "thin": bool(thin), "dilate": int(d),
                           "soft": False, "gamma": 1.0},
                          f"t0={t0:<8.5g} thin={int(thin)} band={d}px hard")

        if a.soft_band:
            # Same support as the hard band of the same width - only the written values differ.
            band_floors = [float(v) for v in str(a.band_floors).split(",") if v.strip()] \
                if a.band_floors else ([float(reference_t0)] if reference_t0 is not None else [0.0])
            gammas = [float(v) for v in str(a.band_gammas).split(",") if v.strip()]
            for t0 in sorted({f for f in band_floors} | ({float(reference_t0)}
                                                         if reference_t0 is not None else set())):
                for g in gammas:
                    for d in [x for x in dilates if x > 0]:
                        _emit({"t0": float(t0), "thin": True, "dilate": int(d),
                               "soft": True, "gamma": float(g)},
                              f"t0={t0:<8.5g} thin=1 band={d}px ramp gamma={g:g}")

        sweep_table.sort(key=lambda r: (-r["dti"], r["t0"], r["dilate"], r.get("gamma", 1.0)))
        res["shaping_sweep"] = sweep_table
        res["sweep_best"] = sweep_table[0] if sweep_table else None
        hard_rows = [r for r in sweep_table if not r.get("soft")]
        by_width = {}
        for r in hard_rows:
            if r["thin"]:
                by_width.setdefault(r["dilate"], []).append(r["dti"])
        res["best_per_emission_width"] = {str(k): round(max(v), 6) for k, v in sorted(by_width.items())}
        ramp_rows = [r for r in sweep_table if r.get("soft")]
        res["best_per_emission"] = {
            "hard": (max((r["dti"] for r in hard_rows), default=None)),
            "ramp": (max((r["dti"] for r in ramp_rows), default=None)),
            "ramp_n_candidates": len(ramp_rows),
            "note": ("hard = probability 1 inside the band; ramp = the same support with values "
                     "decaying linearly to 0 at the band edge (src/submission_optim.soft_band). "
                     "A tie means the values written into the support do not matter here."),
        }
        # The shipped-policy reference row is a HARD, un-widened row: the ramp rows describe other
        # supports' worth of emission and must not be mistaken for it.
        current = next((r for r in hard_rows
                        if reference_t0 is not None and abs(r["t0"] - reference_t0) < 1e-9
                        and r["thin"] and r["dilate"] == 0), None)
        res["sweep_verdict"] = {
            "skeleton_dti": res["best_per_emission_width"].get("0"),
            "best_dti": res["sweep_best"]["dti"] if res["sweep_best"] else None,
            # The t0=0 row is NOT the shipped policy - it emits a different mask (measured: 14,285
            # px at 0.0144 vs the shipped 21,492 px at 0.0247 on the same population).  When the
            # shipped floor is known, say so and compare like with like.
            "reference_t0": reference_t0,
            "reference_source": reference_src,
            "current_policy_dti": current["dti"] if current else None,
            "beats_current_policy_by": (round(res["sweep_best"]["dti"] - current["dti"], 6)
                                        if (current is not None and res["sweep_best"]) else None),
            "acceptance_rule": ("change the default shaping only if a candidate beats the SHIPPED "
                                "policy (the reference floor with no widening%s) by > 0.01 proxy DTI "
                                "here AND is reproduced on a second ensemble; this population is the "
                                "closest measurable stand-in for the scored one, not the scored one"
                                % ("" if reference_t0 is not None else " - none was given, so the "
                                   "t0=0 skeleton is the only available reference and is NOT the "
                                   "shipped policy")),
        }
        # The proxy truth is 6,166 km of state-map fault trace - plausibly much larger than the
        # scored set.  Since FP_w and beta*|G| scale differently, a policy that wins here need not
        # win there; project every candidate onto a range of possible scored-truth sizes instead of
        # guessing which one we are in.
        sizes = [float(v) for v in str(a.sens_sizes).split(",") if v.strip()]
        # Projection over the HARD candidates only: the printed policy label is (floor, thin,
        # band width), which does not identify a ramp candidate, and the ramp comparison is made
        # at equal support by best_per_emission above.  Including ramp rows here would let the
        # projection table print a policy that cannot be reconstructed from its own label.
        sens_rows, sens = sensitivity_table(hard_rows, n_truth, sizes,
                                            alpha=a.alpha, beta=a.beta)
        res["sensitivity"] = sens
        print("\n  projected onto a HIDDEN scored truth of a different size "
              "(coverage and false-positive mass held at the measured values):")
        print("    truth px | best policy (floor/thin/band) | best DTI | skeleton DTI | gain")
        for p in sens["per_size"]:
            print("    %8d | %-27s | %8.4f | %12.4f | %+.4f%s"
                  % (p["truth_px"], "%.4g / %d / %dpx" % (p["best_t0"], p["best_thin"],
                                                          p["best_dilate"]),
                     p["best_dti"], p["skeleton_dti"], p["gain_over_skeleton"],
                     "  <-- wider band passes" if p["passes_acceptance"] else ""))
        if current is not None:
            print("    shipped policy (floor %.6g, no widening): proxy DTI %.6f; best candidate "
                  "%+.6f vs it" % (reference_t0, current["dti"],
                                   res["sweep_verdict"]["beats_current_policy_by"]))
        v = sens["verdict"]
        print("    wider band passes at: %s" % (v["sizes_where_a_wider_band_passes"] or "no size"))
        print("    skeleton still best at: %s" % (v["sizes_where_the_skeleton_wins"] or "no size"))

    sweep_px = int(np.count_nonzero(np.nan_to_num(pred)))
    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/eval_proxy_catalogue.py",
        "purpose": ("score a prediction against faults the training labels do not contain "
                    "(docs/DISCOVERY_PLAN.md 3b) - the only local measurement aligned with the "
                    "scored 'new fault' population"),
        "inputs": {
            "pred": a.pred, "pred_sha256": sha256(Path(a.pred)), "pred_meta": pmeta,
            "proxy": a.proxy, "proxy_sha256": sha256(Path(a.proxy)), "proxy_meta": cmeta,
            "labels": a.labels, "truth_mode": a.truth,
            "reference_t0": reference_t0, "reference_source": reference_src,
            "metric": {"R_pixels": a.R, "R_meters": a.R * 100, "alpha": a.alpha, "beta": a.beta,
                       "eps": a.eps},
        },
        # 100 m pixels: km = px * 0.1.  MEASURED 2026-09-16: this said 0.01 everywhere, so every
        # committed truth length was reported 10x short (61,664 px printed as 616.6 km when the
        # proxy truth is 6,166 km).  build_proxy_catalogue.py already used px_m/1000 = 0.1 for the
        # same quantity, so the two scripts contradicted each other by 10x.
        "truth": {"mode": a.truth, "px": n_truth, "km": round(n_truth * 0.1, 3),
                  "proxy_only_px": int(only.sum()), "proxy_near_label_px": int(near.sum()),
                  "catalogue_coverage_of_proxy": round(
                      int(near.sum()) / max(int(near.sum()) + int(only.sum()), 1), 4)},
        "prediction": {"emission_px": sweep_px, "mass": round(float(np.nansum(pred)), 1)},
        "baseline_support": {"source": support_src, "px": int(support.sum()),
                             "why": ("the blanket baseline is taken over this support so it does not "
                                     "change with the raster being scored; the metric itself sums over "
                                     "the whole raster, exactly as the official one does")},
        "results": res,
        "caveats": [
            "Population, not score: SGMC faults are published surface mapping; the scored faults are "
            "expert interpretations of the GeoDAWN geophysics. Compare policies, not leaderboards.",
            "The FP term is a global sum over the whole raster, exactly as the official metric - the "
            "truth set is sparse, so a broad, diffuse prediction is punished here too.",
            "Code-2 pixels are 'absent from the training labels', which is not the same as 'absent "
            "from the literature'.",
        ],
    }

    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\ntruth: {a.truth} -> {n_truth} px ({n_truth * 0.1:.1f} km of fault trace)")
    print(f"the raster at --pred ({a.pred}), scored as provided: DTI {provided['dti']:.4f}  "
          f"(TP_w {provided['TP_w']:.0f} FP_w {provided['FP_w']:.0f} "
          f"FN_w {provided['FN_w']:.0f})")
    print(f"blanket baseline support: {support_src} ({int(support.sum()):,} px)")
    for k, v in res["baselines"].items():
        print(f"  baseline {k:<28} DTI {v['dti']:.4f}")
    if "acceptance" in res:
        print(f"  acceptance catalogue-copy == 0: {'PASS' if res['acceptance']['passed'] else 'FAIL'}")
    print(f"wrote {op}")
    return 0 if res.get("acceptance", {}).get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
