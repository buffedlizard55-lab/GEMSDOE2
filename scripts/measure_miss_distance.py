#!/usr/bin/env python3
"""Decompose a submission's error on the new-fault-like population into LOCALIZATION vs DETECTION.

THE QUESTION THIS ANSWERS
-------------------------
`data/evidence/emission_decision.json` concludes that the scored universe (expert-mapped NEW
faults, absent from `labels.tif`) is measured better by a WIDER emitted band than by the shipped
thinned skeleton, and that the gain was still rising at the widest swept width.  That leaves the
practical question the width sweep cannot answer by itself: how much of the remaining error is
*localization* error - the model emitted something near the fault but not within the metric's
R = 3-pixel tolerance, which widening converts into credit - and how much is *detection* error -
the model emitted nothing anywhere near the fault, which no amount of widening recovers?

The distinction decides where effort goes.  Widening is free to implement and bounded (it can only
recover faults within the widest band); a detection failure needs a better model (features, data,
capacity), which is expensive.

METHOD (all definitions taken from the official metric, nothing modelled)
-----------------------------------------------------------------------
1. truth = the proxy catalogue's "absent from the labels" pixels (USGS SGMC faults the training
   labels do not contain), the same population `scripts/eval_proxy_catalogue.py` scores;
2. emitted = the prediction's kept set (pred > threshold; a hard shaped submission is 0/1);
3. for every truth pixel, the Euclidean distance to the nearest emitted pixel
   (`scipy.ndimage.distance_transform_edt(~emitted)` sampled at the truth).  The metric's own
   credited band is distance <= R = 3 px, so:
      d <= R        already credited when the model's probability is 1 (nothing to gain)
      R < d <= W    recoverable by emitting a band of width W
      d > W         a detection failure at that width - no widening recovers it
4. the exact official metric DTI is measured for each width W (dilate the kept set by W px and
   score with src.metrics.GtContext), so the "recoverable" claim is not an estimate: it is the
   scored quantity itself.

WHAT IT IS NOT: the proxy population is state-geological-survey surface mapping, not the expert
interpretation of GeoDAWN geophysics that the prize scores.  The split is a planning instrument,
not a leaderboard prediction.

USAGE
    python scripts/measure_miss_distance.py --pred submission.tif \
        --out data/evidence/proxy/miss_distance.json
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
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import GtContext                                  # noqa: E402
from src.submission_optim import dilate_mask                       # noqa: E402

CODE_NEAR, CODE_ONLY = 1, 2
R_PIXELS = 3                 # the official 300 m support at 100 m pixels
ALPHA, BETA = 0.2, 0.8
PX_KM = 0.1
NEAR_MISS_BANDS = (3, 6, 12, 20, 30)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = dict(width=src.width, height=src.height, crs=str(src.crs),
                    transform=[round(v, 6) for v in list(src.transform)[:6]])
    return arr, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="a submission or probability raster on the competition grid")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--truth", choices=["only", "all"], default="only")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="a prediction pixel counts as emitted when pred > this (0.5 separates a "
                         "hard-shaped submission's 0/1 pixels)")
    ap.add_argument("--widths", default="0,1,2,3,4,6,8,10,12,16,20,30,40",
                    help="band widths in pixels to measure the exact metric at")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="optional label recorded in the evidence file")
    ap.add_argument("--oracle", action="store_true",
                    help="also measure the ORACLE CEILING of each band width: the emitted set is "
                         "the truth dilated by the same width, i.e. a perfect localizer that still "
                         "writes a band. Bounds what any policy of that width can ever score, so a "
                         "projected value above it is provably wrong.")
    a = ap.parse_args()

    pred, pmeta = read(Path(a.pred))
    coded, cmeta = read(Path(a.proxy))
    if (pmeta["width"], pmeta["height"]) != (cmeta["width"], cmeta["height"]):
        raise SystemExit(f"grid mismatch: {a.pred} {pmeta['width']}x{pmeta['height']} vs "
                         f"{a.proxy} {cmeta['width']}x{cmeta['height']}")

    truth = (coded == CODE_ONLY) if a.truth == "only" else (coded > 0)
    n_truth = int(truth.sum())
    if n_truth == 0:
        raise SystemExit("empty truth set - refusing to report a decomposition of nothing")

    emitted = np.nan_to_num(pred, nan=0.0) > float(a.threshold)
    n_emit = int(emitted.sum())
    if n_emit == 0:
        raise SystemExit("the prediction emits nothing at this threshold - every miss is a "
                         "detection failure by definition; fix the prediction first")

    # ---- miss geometry (one distance transform, sampled at the truth pixels) ----------------
    dist_to_emit = distance_transform_edt(~emitted)
    d = dist_to_emit[truth].astype(np.float32)
    ctx = GtContext(truth, R_pixels=R_PIXELS)

    def frac_within(k: float) -> float:
        return float((d <= k).mean())

    localization = {
        "credited_within_R": frac_within(R_PIXELS),
        **{f"within_{k}px": frac_within(k) for k in NEAR_MISS_BANDS},
        "percentiles_px": {p: float(np.percentile(d, p)) for p in (50, 75, 90, 95)},
        "max_px": float(d.max()),
        "km_of_trace_within_12px": round(frac_within(12) * n_truth * PX_KM, 1),
    }

    # ---- the exact metric as a function of the emitted band width ---------------------------
    widths = sorted({int(w) for w in str(a.widths).split(",") if w.strip()})
    curve = []
    for w in widths:
        m = dilate_mask(emitted.astype(np.float32), radius=w) if w else emitted.astype(np.float32)
        dti, (tp, fp, fn) = ctx.score(m, alpha=ALPHA, beta=BETA, return_components=True)
        curve.append(dict(width_px=w, dti=round(float(dti), 6), TP_w=round(float(tp), 3),
                          FP_w=round(float(fp), 3), FN_w=round(float(fn), 3),
                          emission_px=int(m.sum()), emission_km=round(float(m.sum()) * PX_KM, 1)))
    best = max(curve, key=lambda r: r["dti"])
    shipped_row = next((r for r in curve if r["width_px"] == 0), None)

    # Marginal value of each extra pixel of band: where widening stops paying.
    marginal = [dict(from_px=curve[i]["width_px"], to_px=curve[i + 1]["width_px"],
                     dti_gain=round(curve[i + 1]["dti"] - curve[i]["dti"], 6),
                     px_added=curve[i + 1]["emission_px"] - curve[i]["emission_px"])
                for i in range(len(curve) - 1)]

    # ---- the ceiling: what a PERFECT localizer scores if it still emits a band ----------------
    #
    # Arithmetic, not a submission.  Emitting the truth itself scores exactly 1.0 (TP_w = |G|,
    # FP_w = 0), so the interesting question is what a band costs even when it is centred
    # perfectly: FP_w = sum over band pixels of p*(1 - max_g k(d(x,g))) = sum d(x)/R at p = 1,
    # and TP_w = |G| with FN_w = 0.  Because both TP_w and FP_w scale with |G| for a band around a
    # self-similar truth, the ceiling 1/(1 + 0.2*FP_w/|G|) does NOT depend on the hidden truth
    # size - which is what makes it usable here at all (see the projection further down).
    oracle = None
    if a.oracle:
        rows = []
        for w in widths:
            m = dilate_mask(truth.astype(np.float32), radius=w) if w \
                else truth.astype(np.float32)
            dti, (tp, fp, fn) = ctx.score(m, alpha=ALPHA, beta=BETA, return_components=True)
            rows.append(dict(width_px=w, oracle_dti=round(float(dti), 6),
                             emission_px=int(m.sum()),
                             fp_per_truth_px=round(float(fp) / max(n_truth, 1), 4)))
        oracle = {
            "rows": rows,
            "defined_as": ("the emitted set is the truth dilated by the same width with p = 1: a "
                           "perfect localizer that still writes a band. Emission of the truth "
                           "itself (width 0) scores 1.0 by construction"),
            "truth_size_independent": True,
            "why": ("TP_w and FP_w both scale with |G| for a band around a self-similar truth, so "
                    "the ceiling depends only on the width, not on how large the hidden truth is"),
            "reading": ("a candidate whose PROJECTED value exceeds the oracle ceiling at its own "
                        "width is wrong - the projection assumes the coverage a perfect localizer "
                        "would need"),
        }
        print("\n  oracle ceiling (perfect localizer, same band width):")
        for r in rows:
            print("    width %3d px -> DTI %.4f  (emits %d px, FP_w/|G| = %.3f)"
                  % (r["width_px"], r["oracle_dti"], r["emission_px"], r["fp_per_truth_px"]))

    # ---- how much of the truth is unreachable at ANY plausible width ------------------------
    unreachable = {f"beyond_{k}px": round(1.0 - frac_within(k), 6) for k in (6, 12, 20, 30)}

    # ---- which width would the SCORED set want, as a function of its (unknown) size ---------
    # Absolute proxy DTI is not comparable to a leaderboard number, and the hidden truth's size
    # |G| is unknown, so the width cannot be read off one population.  The metric's two error
    # terms scale differently: FP_w is fixed by the prediction while beta*|G| grows with the
    # hidden truth.  Holding each width's measured coverage and FP_w fixed and letting |G| vary
    # (the projection documented in scripts/decide_emission_width.py, exact at |G| = |G_proxy|)
    # gives, for every assumed truth size, which measured width maximises the score.  That turns
    # "wider wins here" into "wider wins above this truth size".
    sizes = [500, 1000, 2500, 5000, 10000, 20000, 26042, 61664, 100000, 250000]

    def project(cov: float, fp_w: float, g: float) -> float:
        tp = cov * g
        return float(tp / (ALPHA * (tp + fp_w) + BETA * g + 1e-12))

    projection = []
    for g in sizes:
        scored = [dict(width_px=r["width_px"],
                       projected_dti=round(project(r["TP_w"] / n_truth, r["FP_w"], g), 6))
                  for r in curve]
        best_g = max(scored, key=lambda r: r["projected_dti"])
        skel_g = next(r for r in scored if r["width_px"] == 0)
        projection.append(dict(
            truth_px=g, truth_km=round(g * PX_KM, 1),
            best_width_px=best_g["width_px"], best_dti=best_g["projected_dti"],
            skeleton_dti=skel_g["projected_dti"],
            gain_over_skeleton=round(best_g["projected_dti"] - skel_g["projected_dti"], 6),
            widening_wins=bool(best_g["width_px"] > 0 and best_g["projected_dti"] > skel_g["projected_dti"]),
            per_width={str(r["width_px"]): r["projected_dti"] for r in scored}))

    # The truth size above which the measured optimum stops being the skeleton.
    widen_above = next((p["truth_px"] for p in projection
                        if p["widening_wins"] and p["best_width_px"] > 0), None)

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/measure_miss_distance.py",
        "purpose": ("split the error on the new-fault-like population into localization error "
                    "(recoverable by widening the emitted band) and detection error (not "
                    "recoverable by widening) using the official metric's own R = 3 px tolerance"),
        "label": a.label,
        "inputs": {
            "pred": {"path": a.pred, "sha256": sha256(Path(a.pred)), "threshold": a.threshold},
            "proxy": {"path": a.proxy, "sha256": sha256(Path(a.proxy))},
            "truth_mode": a.truth,
        },
        "grid": pmeta,
        "metric": {"R_pixels": R_PIXELS, "R_meters": R_PIXELS * 100, "alpha": ALPHA, "beta": BETA},
        "truth_px": n_truth,
        "truth_km": round(n_truth * PX_KM, 1),
        "emitted_px": n_emit,
        "emitted_km": round(n_emit * PX_KM, 1),
        "miss_geometry": localization,
        "unreachable_by_widening": unreachable,
        "width_curve": curve,
        "oracle_band_ceiling": oracle,
        "marginal_gain_per_width": marginal,
        "projection_over_scored_truth_size": projection,
        "widening_starts_winning_above_px": widen_above,
        "verdict": {
            "shipped_width0_dti": shipped_row["dti"] if shipped_row else None,
            "best_width_px": best["width_px"],
            "best_dti": best["dti"],
            "gain_from_widening": (round(best["dti"] - shipped_row["dti"], 6)
                                   if shipped_row else None),
            "still_improving_at_the_widest_measured_width": bool(
                curve and best["width_px"] == curve[-1]["width_px"]),
            "detection_failure_share": round(1.0 - frac_within(12), 6),
            "widening_starts_winning_above_px": widen_above,
            "widening_starts_winning_above_km": (round(widen_above * PX_KM, 1)
                                                 if widen_above else None),
            "reading": ("share of the proxy truth that widening the emitted band converts into "
                        "credit vs the share that is beyond 1.2 km of any emitted pixel and needs "
                        "detection, not width"),
        },
        "caveats": [
            "Population C is USGS state-map surface mapping, not the expert interpretation of "
            "GeoDAWN geophysics that the prize scores: policy planning only.",
            "The truth is the code-2 proxy trace only (faults absent from labels.tif), so a "
            "prediction that reproduces the catalogue scores zero here by construction.",
            "The width curve dilates the SAME emitted set; it does not re-rank the probability "
            "field, so it is the marginal value of width alone.",
            "The projection over scored-truth sizes holds each width's measured coverage and FP "
            "mass fixed while |G| varies.  A different-sized truth would not be covered at exactly "
            "the same fractional rate, so the projected row is the assumption stated out loud, not "
            "an estimate; it is exact at the proxy's own size (61,664 px).",
        ],
    }
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"truth {n_truth:,} px ({out['truth_km']:,.0f} km)   emitted {n_emit:,} px "
          f"({out['emitted_km']:,.0f} km) at threshold {a.threshold:g}")
    print("miss distance (px) percentiles: " +
          "  ".join(f"p{p}={v:.0f}" for p, v in localization["percentiles_px"].items()))
    print("fraction of truth within: " +
          "  ".join([f"R=3: {localization['credited_within_R']:.3f}"] +
                   [f"{k}px: {localization[f'within_{k}px']:.3f}" for k in NEAR_MISS_BANDS]))
    print("\nwidth curve (official metric on the same emitted set, dilated):")
    for r in curve:
        print(f"  +{r['width_px']:>2d} px band  DTI {r['dti']:.4f}  emitted {r['emission_px']:>8,} px"
              f"  ({r['emission_km']:>8,.0f} km)")
    print(f"\nbest measured width {best['width_px']} px -> DTI {best['dti']:.4f} "
          f"({out['verdict']['gain_from_widening']:+.4f} vs width 0)")
    print(f"detection failures (no emitted pixel within 12 px = 1.2 km): "
          f"{out['verdict']['detection_failure_share']:.1%} of the truth")
    print("\nwhich width the SCORED set would want, by assumed truth size "
          "(projection, exact at the proxy's own size):")
    for pr in projection:
        print(f"  |G| {pr['truth_px']:>7,} px ({pr['truth_km']:>8,.0f} km): best width "
              f"{pr['best_width_px']:>2d} px  DTI {pr['best_dti']:.4f}  skeleton {pr['skeleton_dti']:.4f}"
              f"  {pr['gain_over_skeleton']:+.4f}")
    print(f"wrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
