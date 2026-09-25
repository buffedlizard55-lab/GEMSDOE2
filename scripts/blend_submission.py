#!/usr/bin/env python3
"""Blend parallel MC-fold predictions into one shaped submission.tif.

Inputs are fold directories produced by fold jobs (``python -m src.train`` with
``--override training.mc_id=F training.mc_splits=1`` followed by
``python -m src.inference --raw``).  Each fold dir must contain:

    prob_raw.tif      full-raster raw probability map from that fold's model
    heldout_mc*.npz   that fold's best-epoch held-out crop (pred, gt) — written by src/train.py
    manifest.json     that fold's run manifest (model name, per-arch dti, config)

Algorithm (this is the whole point — every step is derived from the official metric,
see docs/METRIC_STRATEGY.md):

  1. ensemble probability = nanmean of the fold maps (equal weights; the reference
     solution's MC ensemble is an equal-weight mean too);
  2. shaping calibration = for every (t0, thin) candidate, score each fold's held-out
     crop with THAT fold's model prediction and take the MEAN DTI across folds; choose
     the argmax.  One scalar pair fitted against |folds| x |windows| held-out pixels;
  3. submission = floor(t0) -> distance-R dominating thinning -> clip to [0,1];
     NaN outside the data footprint, written on the sample submission's grid;
  4. optional informational scoring vs the PUBLIC labels (known faults) — NOT the test
     set; reported as such.

Writes submission.tif + blend_report.json (auditable: calibration table, masses,
shas, grids).  Exit code 1 if the submission fails the format self-check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import (GtContext, compute_distance_weighted_tversky,        # noqa: E402
                         score_arrays_blocked)
from src.submission_optim import optimize_submission, shaping_thresholds  # noqa: E402
from src.submission_io import clean_profile, conform_to_template, sha256_pixels, write_submission  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_folds(fold_dirs: list[str]):
    folds = []
    for d in fold_dirs:
        d = Path(d)
        prob = d / "prob_raw.tif"
        if not prob.exists():
            raise SystemExit(f"fold dir {d}: missing prob_raw.tif (run src/inference --raw there)")
        held = sorted(d.glob("heldout_mc*.npz"))
        manifest = d / "manifest.json"
        man = json.loads(manifest.read_text()) if manifest.exists() else {}
        with rasterio.open(prob) as src:
            arr = src.read(1).astype(np.float32)
            grid = dict(width=src.width, height=src.height, crs=str(src.crs),
                        transform=list(src.transform), res=[float(src.res[0]), float(src.res[1])])
        if held:
            z = np.load(held[0])
            pc, gc = z["pred"].astype(np.float32), z["gt"].astype(np.float32)
        else:
            pc = gc = None
        fdti = [m.get("dti") for m in man.get("models", [])]
        folds.append(dict(dir=str(d), prob=arr, pred_crop=pc, gt_crop=gc, grid=grid,
                          manifest=man, model=[m.get("file") for m in man.get("models", [])],
                          fold_dti=fdti,
                          mean_dti=(float(np.mean([f for f in fdti if f is not None]))
                                    if any(f is not None for f in fdti) else 0.0)))
    # grids must agree exactly — different bounds would silently misalign pixels
    for f in folds[1:]:
        if f["grid"] != folds[0]["grid"]:
            raise SystemExit(f"grid mismatch between fold dirs {folds[0]['dir']} and {f['dir']}")
    return folds


def calibrate_shaping(folds, R: int, thresholds: np.ndarray, alpha: float, beta: float, pre=None,
                      dilate_options=(0,), min_dilate: int = 0):
    """Pooled held-out search over (t0, thin, dilate). Returns (t0*, thin*, mean_dti*, table, dilate*).

    Each fold contributes DTI(shaped fold-model crop, fold gt crop); the chosen point
    maximises the MEAN over folds.  The raw (unshaped) mean is reported too so the
    gain from shaping is auditable.

    `dilate_options` grows the kept set after thinning (src.submission_optim.dilate_mask).
    MEASURED 2026-09-16 on the one window where a written submission and the official labels
    coexist (data/evidence/shift_robustness.json): the skeleton kept 801 px against 5,154
    label px and scored 0.0555; growing the line to a 6-px band scored 0.1260 -- i.e. the
    operator family was the binding constraint, not the floor.  The exchange rate is the
    metric's own: TP_w takes a max within R=3 px (a prediction up to 3 px off still earns
    full credit), FP costs 0.2 per unit while missing GT costs 0.8.  Against the scored
    *new* faults the model has never seen the trace at all, so its localisation error is
    strictly larger than on the catalogue -> the optimum band width is an empirical
    question, and this is where it gets answered.
    """
    usable = [f for f in folds if f["pred_crop"] is not None]
    if not usable:
        # five values, like the success path: callers unpack (t0, thin, mean, table, dilate)
        # and a 4-tuple here used to crash them (fixed 2026-09-16, session 6)
        return 0.3, True, float("nan"), [], 0
    dilate_options = tuple(int(d) for d in dilate_options) or (0,)
    if pre is not None:
        # the FULL map gets `pre` before shaping, so calibration must see the same
        # transform on the fold crops - otherwise the floor is fitted to a different
        # distribution than the one it will be applied to (found by A/B, 2026-09-15:
        # post-Frangi calibration changed the whole outcome).
        # Copy-on-transform, never mutate: the same fold dicts are scored again by
        # loo_aggregation, and mutating them here applied `pre` TWICE with
        # --calibrate loo (fixed 2026-09-16, session 6).
        usable = [dict(f, pred_crop=pre(f["pred_crop"])) for f in usable]
    table = []
    raw_mean = float(np.mean([_dti(f["pred_crop"], f["gt_crop"], R, alpha, beta)
                              for f in usable]))
    # The unshaped row seeds the search so the shaped candidates must EARN their place.  When a
    # minimum emission width has been decided from the new-fault-like measurement (--min-dilate),
    # the seed must not be able to smuggle the old behaviour back in: on the catalogue the raw
    # unshaped map really can beat every band, and the returned parameters would then be a
    # width-0 skeleton written by a run that was asked for a band (a silent no-op, which is
    # exactly the failure mode the option exists to prevent).  The seed is therefore the poorest
    # possible score, so the best candidate AT AN ALLOWED WIDTH wins and the report records that
    # the exclusion happened.
    # The un-thinned branch is also excluded: its only width is 0, so it is the same silent no-op
    # wearing a different hat (an un-thinned blob emits the mask, not a band).
    thin_options = (True,) if int(min_dilate) > 0 else (False, True)
    if int(min_dilate) > 0:
        best = (float("-inf"), float(thresholds[0]), True, int(min(dilate_options)), -1)
    else:
        best = (raw_mean, float(thresholds[0]), True, 0, 0)
    for t in thresholds:
        for thin in thin_options:
            # dilation only makes sense on a thinned skeleton: on the un-thinned branch the
            # floor-passing mask is already a blob, and growing it further is a pure FP cost.
            for dila in (dilate_options if thin else (0,)):
                v = float(np.mean([
                    _dti(optimize_submission(f["pred_crop"], R=R, t0=float(t), thin=thin,
                                             hard=True, gamma=1.0, dilate=dila),
                         f["gt_crop"], R, alpha, beta)
                    for f in usable]))
                if v > best[0]:
                    best = (v, float(t), thin, dila, len(table))
                table.append(dict(t0=float(t), thin=bool(thin), dilate=int(dila), mean_dti=v))
    table.insert(0, dict(t0=None, thin=None, dilate=0, mean_dti=raw_mean, raw=raw_mean,
                         excluded_from_search=bool(int(min_dilate) > 0)))
    return best[1], best[2], best[0], table, best[3]


# One GtContext per label array per radius.  The label geometry (the mask, the label-pixel
# coordinates, the distance transform that FP_w needs) does not change while a search runs over
# hundreds of candidate floors against the same held-out crop; rebuilding it per candidate is what
# made the leave-one-fold-out audit run for over 90 minutes on a runner (2026-09-16).  The cache
# holds a reference to the array so its id() cannot be recycled while an entry is alive.
_GT_CACHE: dict = {}


def _gt_ctx(gt, R):
    key = (id(gt), int(R))
    ctx = _GT_CACHE.get(key)
    if ctx is None or ctx.source is not gt:
        ctx = GtContext(gt, R)
        ctx.source = gt
        _GT_CACHE[key] = ctx
    return ctx


def _dti(pred, gt, R, alpha, beta):
    return float(_gt_ctx(gt, R).score(pred, alpha=alpha, beta=beta))


def loo_aggregation(folds, R: int, thresholds: np.ndarray, alpha: float, beta: float, pre=None,
                    dilate_options=(0,)):
    """Leave-one-fold-out audit of the aggregation step itself.

    `calibrate_shaping` picks (t0, thin) on the SAME held-out folds it reports, so its mean is
    a selection-optimistic number.  Here, for every fold i the (t0, thin) is fitted on the
    other folds only and then scored on fold i.  The mean of those six scores is an honest
    estimate of what the procedure buys on an unseen fold; the gap to the pooled number is
    the optimism.  A per-fold "best on itself" row is included as an oracle ceiling.

    LOO-fitted fold weights (softmax over the other folds' held-out DTI, the `--weights dti`
    rule) are REPORTED per row but deliberately NOT scored: the held-out crops of different
    folds cover different geographic windows (each fold's own compact window subset,
    src/train.py::heldout_maps), so averaging fold j's crop with fold k's crop and scoring it
    against fold i's ground truth mixes geographies and cannot measure the weight rule.  Until
    2026-09-16 this function did exactly that whenever the crops happened to share a shape;
    session 6 removed the comparison (see STATUS.md).  The valid weight-rule evidence is the
    full-map A/B in data/evidence/runs/local-mini-ensemble/, where both rules combine the SAME
    maps on the SAME grid.  A sound LOO weight audit needs the per-fold held-out footprints:
    intersect test windows across folds, combine the full-raster prob maps there, score there
    (queued in SUGGESTIONS.md).

    Returns None when there are fewer than 3 usable folds (nothing to leave out).
    """
    usable = [f for f in folds if f["pred_crop"] is not None]
    if len(usable) < 3:
        return None
    if pre is not None:
        # same transform the full map gets; copy, never mutate (see calibrate_shaping)
        usable = [dict(f, pred_crop=pre(f["pred_crop"])) for f in usable]

    rows = []
    for i, f in enumerate(usable):
        rest = usable[:i] + usable[i + 1:]
        t_loo, thin_loo, _, _, dila_loo = calibrate_shaping(rest, R, thresholds, alpha, beta,
                                                            pre=None, dilate_options=dilate_options)
        d_loo = _dti(optimize_submission(f["pred_crop"], R=R, t0=float(t_loo), thin=bool(thin_loo),
                                         hard=True, gamma=1.0, dilate=int(dila_loo)),
                     f["gt_crop"], R, alpha, beta)
        d_unshaped = _dti(f["pred_crop"], f["gt_crop"], R, alpha, beta)
        best_self, t_self, thin_self, dila_self = d_unshaped, None, None, None
        for t in thresholds:
            for thin in (False, True):
                for dila in (dilate_options if thin else (0,)):
                    v = _dti(optimize_submission(f["pred_crop"], R=R, t0=float(t), thin=bool(thin),
                                                 hard=True, gamma=1.0, dilate=int(dila)),
                             f["gt_crop"], R, alpha, beta)
                    if v > best_self:
                        best_self, t_self, thin_self, dila_self = v, float(t), bool(thin), int(dila)
        row = dict(fold=Path(f["dir"]).name, t0_loo=float(t_loo), thin_loo=bool(thin_loo),
                   dilate_loo=int(dila_loo), dti_loo=d_loo, dti_unshaped=d_unshaped,
                   t0_self_best=t_self, thin_self_best=thin_self, dilate_self_best=dila_self,
                   dti_self_best=best_self)

        # LOO-fitted fold weights: reported, NOT scored (see docstring for why scoring
        # them here mixed geographies).  The arithmetic is the --weights dti rule exactly.
        other = [o["mean_dti"] for o in rest]
        if all(v is not None and v > 0 for v in other):
            ws = np.array(other, dtype=np.float64)
            ws = np.exp((ws - ws.max()) * 8.0)
            ws = ws / ws.sum()
            row.update(weights_loo=[round(float(w), 4) for w in ws],
                       weights_loo_note=("softmax over the other folds' held-out DTI; reported, "
                                         "not scored — folds' held-out crops cover different "
                                         "regions, so no shared-footprint comparison exists"))
        else:
            row.update(weights_loo=None,
                       weights_loo_note="some folds report no positive held-out DTI")
        rows.append(row)

    mean = lambda k: (float(np.mean([r[k] for r in rows if r.get(k) is not None]))
                      if any(r.get(k) is not None for r in rows) else None)
    dti_loo, dti_unshaped = mean("dti_loo"), mean("dti_unshaped")
    summary = dict(
        n_folds=len(rows),
        mean_dti_loo=dti_loo, mean_dti_unshaped=dti_unshaped,
        mean_dti_self_best_oracle=mean("dti_self_best"),
        gain_loo_over_unshaped=(None if dti_loo is None or dti_unshaped is None
                                else dti_loo - dti_unshaped),
        # No LOO score for the fold-weight rule exists in this report (see docstring): the
        # keys stay present with None so older report readers do not KeyError, but they are
        # not measurements of anything.
        mean_dti_loo_weighted=None,
        mean_dti_loo_equal_weighted_mean=None,
        weight_rule_gain=None,
        weight_rule_note=("the fold-weight rule is fitted per row (weights_loo) but not scored: "
                          "folds' held-out crops cover different regions. Compare weight rules on "
                          "full-raster blends scored on the same grid instead "
                          "(data/evidence/runs/local-mini-ensemble/)."),
        acceptance=("aggregation calibration is worth keeping when mean_dti_loo beats "
                    "mean_dti_unshaped by > 0.01 on held-out folds"),
        note=("(t0, thin) is re-fitted on the folds that are NOT being scored, so this mean is "
              "not selection-optimistic; the pooled number reported elsewhere is."),
    )
    return dict(rows=rows, summary=summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", required=True, help="fold directories (see module docstring)")
    ap.add_argument("--config", default="configs/config_ci_ensemble.yaml")
    ap.add_argument("--sample", default=None, help="sample_submission.tif — authoritative grid for the write")
    ap.add_argument("--labels", default=None, help="known-fault raster for the informational score")
    ap.add_argument("--out", default="submission.tif")
    ap.add_argument("--save-ensemble", default=None,
                    help="write the pre-shaping ensemble mean (probability raster) here, so policy "
                         "sweeps and proxy-catalogue scoring act on the map shaping consumed")
    ap.add_argument("--report", default=None, help="defaults to <out stem>_report.json")
    ap.add_argument("--frangi", action="store_true",
                    help="A/B knob: vesselness (Frangi) line enhancement on the blended mean map "
                         "BEFORE shaping (src/postprocess.frangi_enhance). Off by default until "
                         "measured to help on held-out calibration; see SUGGESTIONS.md.")
    ap.add_argument("--shaping-grid", type=int, default=None, help="threshold count (default: config)")
    ap.add_argument("--calibrate", choices=["pooled", "loo"], default="pooled",
                    help="'pooled' (default) fits (t0, thin) on the same held-out folds it reports - "
                         "selection-optimistic. 'loo' ADDS a leave-one-fold-out audit: the floor is "
                         "re-fitted on the folds that are not being scored, so the reported mean is "
                         "honest; LOO fold weights are reported per row but not scored (folds' "
                         "held-out crops cover different regions). The submission itself is unchanged "
                         "(a single global floor cannot be fitted per fold); only the report grows.")
    ap.add_argument("--dilate-grid", default="0,1,2,3,4,6",
                    help="emission width candidates in pixels: the kept set is the skeleton grown by "
                         "k px. '0' = pure skeleton (previous behaviour). See "
                         "src.submission_optim.dilate_mask for why the scored (new-fault) universe "
                         "may prefer k > 0, and data/evidence/shift_robustness.json for the "
                         "measurement that motivated the search.")
    ap.add_argument("--min-dilate", type=int, default=0,
                    help="drop every candidate narrower than this many pixels from the pooled "
                         "search. The pooled search maximises IN-DOMAIN held-out DTI (the faults "
                         "the model trained on), which always prefers the narrowest band; the "
                         "scored population is the opposite regime. MEASURED 2026-09-17 "
                         "(data/evidence/proxy/miss_distance-ensemble1.json): on faults absent from "
                         "the labels the shipped skeleton has a 22 px MEDIAN miss distance, and "
                         "widening to 8 px / 16 px projects better at every plausible scored-truth "
                         "size above ~2,000 km (8 px is optimal at the GeoDAWN-blocks anchor, 16 px "
                         "at the proxy's own). This option is how a measured width is shipped; it "
                         "changes the search SPACE, never the reported calibration. Default 0 = "
                         "unchanged behaviour until the width decision's condition 3 is met.")
    ap.add_argument("--shaping-t0", type=float, default=None,
                    help="ADOPT this floor instead of the in-domain calibrated one.  The pooled "
                         "held-out calibration (the faults the model trained on) always prefers the "
                         "narrow/high floor corner, and data/evidence/emission_decision.json records "
                         "a floor measured on the new-fault-like population instead.  Requires "
                         "--shaping-dilate so that the shipped policy is exactly a MEASURED policy "
                         "(a floor from one measurement combined with a width from another is not "
                         "measured at all).  The in-domain calibration still runs and is reported as "
                         "the control; it does not choose what is written.")
    ap.add_argument("--shaping-dilate", type=int, default=None,
                    help="emission width (px) shipped together with --shaping-t0; the adopted policy "
                         "is always thinned (thin=True), which is what every swept candidate used.")
    ap.add_argument("--shaping-source", default=None,
                    help="evidence file the adopted policy was measured in; recorded verbatim in the "
                         "blend report so the shipped shaping is traceable to a measurement")
    ap.add_argument("--loo-grid", type=int, default=9,
                    help="threshold count for the leave-one-fold-out search (cheaper than the "
                         "submission grid; the LOO search runs n_folds times)")
    ap.add_argument("--weights", choices=["equal", "dti"], default="equal",
                    help="fold averaging weights. 'dti' = softmax over each fold's best HELD-OUT "
                         "shaped DTI. MEASURED 2026-09-15 on the real 512x512 fixture window: with "
                         "2 folds a weak fold drowned the strong one and dti-weights helped "
                         "(0.0143 -> 0.0984); with 6 folds per-fold DTI differences are crop-noise "
                         "and dti-weights HURT (0.1033 -> 0.0656). Equal averaging is therefore the "
                         "default for the 6-fold workflow; consider 'dti' only to exclude a known-"
                         "catastrophic fold. Evidence: data/evidence/runs/local-mini-ensemble/.")
    args = ap.parse_args()

    # An adopted policy is a JOINT (floor, thinning, width) measurement.  Shipping a floor from one
    # measurement and a width from another would produce a policy that nothing measured - the exact
    # defect --min-dilate was introduced to prevent, one level up.  Refuse at parse time.
    if (args.shaping_t0 is None) != (args.shaping_dilate is None):
        ap.error("--shaping-t0 and --shaping-dilate must be given together: the measured candidate "
                 "is a joint (floor, width) policy, and shipping a floor from one measurement with "
                 "a width from another is not a measured policy")
    if args.shaping_source and args.shaping_t0 is None:
        ap.error("--shaping-source describes an adopted policy, so it needs --shaping-t0/--shaping-dilate")

    cfg = yaml.safe_load(Path(args.config).read_text())
    R = int(cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"])
    alpha = float(cfg["training"]["alpha"])
    beta = float(cfg["training"]["beta"])
    t0_cfg = time.time()

    folds = load_folds(args.folds)
    print(f"loaded {len(folds)} fold(s): {[f['dir'] for f in folds]}")

    # 1. ensemble mean (equal, or softmax over fold held-out DTI when --weights dti).
    # NaN-safe and footprint-aware: pixels where a fold is outside the data footprint
    # average over the folds that ARE present (weighted by their weights), exactly like
    # np.nanmean does for the equal case.
    stack = np.stack([f["prob"] for f in folds])
    ws = None
    if args.weights == "dti" and all(f["mean_dti"] > 0 for f in folds):
        ws = np.array([f["mean_dti"] for f in folds], dtype=np.float64)
        ws = np.exp((ws - ws.max()) * 8.0)
        ws = ws / ws.sum()
        print(f"fold weights (softmax over held-out DTI): {[round(w, 3) for w in ws]}")
    elif args.weights == "dti":
        # previously this fell back to equal weights SILENTLY; a requested weighting that does
        # not happen must say so (fixed 2026-09-16, session 6)
        print("WARNING --weights dti requested but some fold reports no positive held-out DTI; "
              "using equal weights")
    valid = ~np.isnan(stack)
    allnan = ~valid.any(axis=0)
    vals = np.where(valid, stack, 0.0)
    if ws is None:
        num = vals.sum(axis=0)
        present = valid.sum(axis=0).astype(np.float64)
    else:
        num = np.tensordot(ws, vals, axes=1)              # sum_i w_i p_i over present folds
        present = np.tensordot(ws, valid.astype(np.float64), axes=1)
    mean = num / np.maximum(present, 1e-9)
    mean[allnan] = np.nan
    mean = np.clip(np.nan_to_num(mean, nan=0.0), 0.0, 1.0)   # 0 outside footprint for now
    pre_mass = float(mean[~allnan].sum())

    enh = None
    if args.frangi:
        from src.postprocess import frangi_enhance
        enh = lambda m: frangi_enhance(m, scale_range=(1, 6), scale_step=2, weight=0.35)
        mean = enh(mean)
        print("frangi line-enhancement applied to the blended map (A/B, consistent calibration)")

    # 2. pooled held-out shaping calibration ---------------------------------------
    n_grid = int(args.shaping_grid or cfg["training"].get("shaping_grid", 11))
    # shaping_thresholds() is log-spaced and always contains 0.0.  This script used to
    # build its own linear 0.02-0.9 grid, which (a) omitted the no-floor reference
    # point and (b) could not reach a low-contrast optimum -- the same defect PR #9 fixed
    # in src/train.py but not here, even though the BINDING calibration for the 6-fold
    # workflow happens in this script.  See src/submission_optim.shaping_thresholds.
    thr = shaping_thresholds(n_grid)
    dil = tuple(sorted({int(v) for v in str(args.dilate_grid).split(",") if v.strip()}))
    if int(args.min_dilate) > 0:
        kept = tuple(d for d in dil if d >= int(args.min_dilate))
        print(f"--min-dilate {args.min_dilate}: pooled search restricted to widths {kept} "
              f"(the in-domain calibration would otherwise always return the narrowest band)")
        dil = kept or (int(args.min_dilate),)
    t0b, thinb, mean_dti, table, dilb = calibrate_shaping(folds, R, thr, alpha=alpha, beta=beta,
                                                          pre=enh, dilate_options=dil,
                                                          min_dilate=int(args.min_dilate))
    if int(args.min_dilate) > 0 and args.shaping_t0 is None:
        assert int(dilb) >= int(args.min_dilate), (
            f"the pooled search returned dilate={dilb} although --min-dilate "
            f"{args.min_dilate} was requested and the search space was restricted to {dil}; "
            "a requested band that is not shipped is the defect this option exists to prevent")
    print(f"pooled shaping: t0={t0b:.3f} thin={thinb} dilate={dilb}px -> mean held-out DTI "
          f"{mean_dti:.4f} (unshaped {table[0]['mean_dti']:.4f}; "
          f"skeleton r=0 {next((r['mean_dti'] for r in table if r.get('dilate') == 0 and r.get('thin')), float('nan')):.4f})")

    # 2a. the SHIPPED policy: the in-domain calibration above, or an adopted measured policy ------
    # The in-domain calibration maximises held-out DTI against the faults the model trained on, so
    # it is a plumbing check, not a decision (the record says so: that same widening that costs
    # 0.19 -> 0.09 in-domain is worth +0.09 on the new-fault-like population).  When a policy has
    # been measured on the new-fault-like population, this is how it is shipped.
    adopted = args.shaping_t0 is not None
    if adopted:
        ship_t0, ship_thin, ship_dilate = float(args.shaping_t0), True, int(args.shaping_dilate)
        print(f"ADOPTED measured policy: t0={ship_t0:g} thin=True dilate={ship_dilate}px "
              f"(in-domain calibration would have shipped t0={t0b:.3f} thin={thinb} "
              f"dilate={dilb}px, DTI {mean_dti:.4f})"
              + (f"  source: {args.shaping_source}" if args.shaping_source else ""))
    else:
        ship_t0, ship_thin, ship_dilate = float(t0b), bool(thinb), int(dilb)
    calibrated_control = dict(t0=float(t0b), thin=bool(thinb), dilate=int(dilb),
                              mean_heldout_dti=float(mean_dti))

    # 2b. optional leave-one-fold-out audit of the aggregation step ----------------
    loo = None
    if args.calibrate == "loo":
        t_loo = time.time()
        loo = loo_aggregation(folds, R, shaping_thresholds(int(args.loo_grid)), alpha=alpha,
                              beta=beta, pre=enh, dilate_options=dil)
        if loo is None:
            print("LOO audit skipped: fewer than 3 folds carry held-out crops")
        else:
            sm = loo["summary"]
            print("leave-one-fold-out aggregation audit "
                  f"({sm['n_folds']} folds, {time.time() - t_loo:.0f}s):")
            print(f"  mean DTI, floor re-fitted without the scored fold : {sm['mean_dti_loo']:.4f}"
                  f"   (unshaped {sm['mean_dti_unshaped']:.4f},"
                  f" gain {sm['gain_loo_over_unshaped']:+.4f})")
            print(f"  mean DTI, oracle floor fitted on the scored fold : "
                  f"{sm['mean_dti_self_best_oracle']:.4f}")
            print(f"  fold weights are fitted per row but not scored "
                  f"(held-out crops cover different regions; see report note)")
            print(f"  pooled (selection-optimistic) reference : {mean_dti:.4f}")
            for r in loo["rows"]:
                print(f"    {r['fold']:>12s}  t0_loo {r['t0_loo']:.4f} thin {str(r['thin_loo']):5s}"
                      f"  dti_loo {r['dti_loo']:.4f}  unshaped {r['dti_unshaped']:.4f}"
                      f"  self-best {r['dti_self_best']:.4f}")

    # 3. shape on the full ensemble map --------------------------------------------
    q = optimize_submission(mean, R=R, t0=ship_t0, thin=ship_thin, hard=True, gamma=1.0,
                            dilate=ship_dilate)
    q = np.clip(q, 0.0, 1.0).astype(np.float32)
    post_mass = float(q[~allnan].sum())
    if post_mass <= 0.0:
        # Observed failure mode (fixture A/B, 2026-09-15): a pre-transform that flattens
        # the map can drive EVERY pixel under the calibrated floor -> an all-zero
        # submission (DTI 0).  Never write that silently; fail the step loudly instead.
        raise SystemExit("BLEND ABORTED: shaping collapsed the map to all zeros "
                         f"(t0={ship_t0:.3f}, thin={ship_thin}, dilate={ship_dilate}, "
                         f"mean mass {pre_mass:.0f}). "
                         "A pre-transform likely changed the distribution outside calibration.")
    q[allnan] = np.nan                                        # spec: outside bounds null/nan

    # write on the sample grid when given, else on the fold grid (identical by spec;
    # scripts/validate_submission.py re-checks against both the sample and the features)
    #
    # NOTE (2026-09-16): the profile MUST NOT inherit block geometry from the sample file.
    # A striped GeoTIFF reports blockxsize == width (3292 for the competition template), and
    # requesting TILED=YES with that block size makes GDAL refuse to write at all:
    #   "RasterBlockError: The height and width of TIFF dataset blocks must be multiples of 16"
    # (this is exactly how Actions run 35042805806 lost its submission).  clean_profile()
    # drops stale block keys and pins legal 256-px tiles.
    grid = folds[0]["grid"]
    if args.sample and Path(args.sample).exists():
        with rasterio.open(args.sample) as src:
            sgrid = dict(width=src.width, height=src.height, crs=str(src.crs),
                         transform=list(src.transform), res=[float(src.res[0]), float(src.res[1])])
            sref = src.read(1)
            # nodata must survive: clean_profile's default would drop the template's "nan"
            profile = clean_profile(src.profile.copy(), dtype="float32", nodata=src.nodata)
        if sgrid["width"] != grid["width"] or sgrid["height"] != grid["height"]:
            print(f"WARNING sample grid {sgrid['width']}x{sgrid['height']} != fold grid "
                  f"{grid['width']}x{grid['height']} — writing on the SAMPLE grid (crop/pad)")
            buf = np.full((sgrid["height"], sgrid["width"]), np.nan, np.float32)
            hh, ww = min(sgrid["height"], q.shape[0]), min(sgrid["width"], q.shape[1])
            buf[:hh, :ww] = q[:hh, :ww]
            q = buf
        # Template conformance (added 2026-09-25 after a real platform rejection):
        # NaN holes inside the sample's valid region are read by the platform as values
        # outside [0, 1] ("Predicted values must be in range [0, 1]"), and finite px
        # outside it violate "data outside the bounds is null or nan".  conform_to_template
        # fixes both and reports the counts; scoring is unaffected (np.nan_to_num semantics).
        q, conf = conform_to_template(q, sref)
        if conf["filled_inside"] or conf["masked_outside"] or conf["clipped"]:
            print(f"template conformance: filled {conf['filled_inside']} NaN px inside the "
                  f"sample's valid region, masked {conf['masked_outside']} px outside it, "
                  f"clipped {conf['clipped']} px to [0,1]")
    else:
        profile = clean_profile(None, dtype="float32", crs=grid["crs"],
                                transform=rasterio.Affine(*grid["transform"]),
                                height=q.shape[0], width=q.shape[1],
                                nodata=float("nan"))

    # Optional: persist the pre-shaping ensemble mean.  Policy questions (floor, emission width,
    # thinning) can only be re-asked on the map shaping actually consumed; re-deriving it from the
    # shaped submission would bake the current policy into the comparison.  Written on the same grid
    # and with the same profile rules as the submission.
    if args.save_ensemble:
        ens = np.asarray(mean, np.float32)
        if ens.shape != q.shape:
            buf = np.full(q.shape, np.nan, np.float32)
            hh, ww = min(q.shape[0], ens.shape[0]), min(q.shape[1], ens.shape[1])
            buf[:hh, :ww] = ens[:hh, :ww]
            ens = buf
        eprof = clean_profile(dict(profile), dtype="float32")
        eprof["nodata"] = float("nan")
        ip = Path(args.save_ensemble)
        ip.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(ip, "w", **eprof) as dst:
            dst.write(ens.astype(np.float32), 1)
            dst.set_band_description(1, "MC-ensemble mean probability, pre-shaping")
        print(f"wrote ensemble mean {ip} ({ip.stat().st_size} B) - the exact input to shaping, "
              "kept so policy sweeps do not have to re-derive it")

    out = Path(args.out)
    # write_submission() reopens the bytes on disk and raises unless it reads back as a
    # single-band float32 raster on this grid with non-zero mass -> a crashed or empty
    # submission can no longer be reported as success.
    written = write_submission(
        out, q, profile,
        band_description="fault-presence probability (GEMSDOE MC ensemble, shaped)",
        tags=dict(source="GEMSDOE train-ensemble workflow", n_models=str(len(folds)),
                  models=";".join(str(m) for f in folds for m in f["model"]),
                  shaping_t0=str(ship_t0), shaping_thin=str(bool(ship_thin)),
                  shaping_dilate=str(int(ship_dilate)),
                  shaping_source=("adopted_measured_policy" if adopted
                                  else "pooled_in_domain_calibration"),
                  shaping_evidence=str(args.shaping_source or "")))
    print(f"wrote {out}  {written['bytes']} B sha256={written['sha256'][:16]}  "
          f"nonzero={written['nonzero_px']} finite={written['finite_px']}/{written['total_px']} "
          f"mass {pre_mass:.0f} -> {post_mass:.0f}  tiled={written['tiled']} "
          f"blocks={written['block_shapes']}")

    # 4. informational score vs known (public) faults -------------------------------
    local = None
    discovery = None
    if args.labels and Path(args.labels).exists():
        with rasterio.open(args.labels) as src:
            lab = src.read(1)
            if src.nodata is not None:
                lab[lab == src.nodata] = 0
        gt = (np.asarray(lab, np.float32) > 0.5).astype(np.float32)
        if gt.shape != q.shape:
            gt = gt[: q.shape[0], : q.shape[1]]
        d, (tp, fp, fn) = score_arrays_blocked(np.nan_to_num(q), gt, R_pixels=R, alpha=alpha, beta=beta)
        n_g = float(gt.sum())
        closed = tp / ((1 - beta) * tp + alpha * fp + beta * n_g + cfg["metric"].get("epsilon", 1e-7))
        ones = np.ones_like(gt)
        d_ones, (tp1, fp1, fn1) = score_arrays_blocked(ones, gt, R_pixels=R, alpha=alpha, beta=beta)
        local = dict(dti_known_faults=float(d), TP_w=float(tp), FP_w=float(fp), FN_w=float(fn),
                     closed_form=float(closed), closed_form_matches=bool(abs(closed - d) < 1e-6),
                     blanket_ones_dti=float(d_ones),
                     note="scored against the PUBLIC known-fault raster — NOT the competition test "
                          "set (private expert-labelled NEW faults). Optimistic AND wrong-universe: "
                          "reported for pipeline monitoring only.")
        print(f"informational DTI vs known faults = {d:.4f} (blanket-ones floor {d_ones:.4f})")
        # The competition target is the NEW fault dataset (rules §1.1) — disjoint from the
        # catalog we can see.  Measure how much of the submission is a *discovery*.
        from src.discovery import discovery_report
        thr_diag = [t for t in (0.1, 0.5, 0.9)]
        discovery = discovery_report(np.nan_to_num(q), gt, R=R, alpha=alpha, beta=beta,
                                     thresholds=thr_diag)
        print("discovery vs known catalog: novel_fraction={:.3f}  candidate_new(>0.5)={} px "
              "in {} components".format(
                  discovery["novel_mass"]["novel_fraction"] or 0.0,
                  discovery["per_threshold"]["0.5"]["novel_px"],
                  discovery["per_threshold"]["0.5"]["n_novel_components"]))

    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/blend_submission.py",
        minutes=round((time.time() - t0_cfg) / 60.0, 2),
        n_folds=len(folds),
        folds=[dict(dir=f["dir"], models=f["model"], fold_best_dti=f["fold_dti"],
                    grid=f["grid"]) for f in folds],
        metric=dict(R_pixels=R, alpha=alpha, beta=beta),
        aggregation_calibration=loo,
        shaping=dict(t0=ship_t0, thin=bool(ship_thin), dilate=int(ship_dilate), dilate_grid=list(dil),
                     min_dilate=int(args.min_dilate),
                     mean_heldout_dti=float(mean_dti),
                     unshaped_mean_heldout_dti=float(table[0]["mean_dti"]),
                     calibration_table=table,
                     # Where the SHIPPED policy came from, and what the in-domain calibration would
                     # have shipped instead.  Both are needed to audit a policy change: the second
                     # is the counterfactual, and without it "adopted" is not distinguishable from
                     # "the calibration happened to agree".
                     source=("adopted_measured_policy" if adopted
                             else "pooled_in_domain_calibration"),
                     adopted=dict(t0=ship_t0, thin=bool(ship_thin), dilate=int(ship_dilate),
                                  evidence=args.shaping_source) if adopted else None,
                     in_domain_calibration_control=calibrated_control),
        probability_mass=dict(pre_shaping=pre_mass, post_shaping=post_mass,
                              collapse_factor=(pre_mass / post_mass if post_mass > 0 else None)),
        fold_weights=("equal" if ws is None else [round(float(w), 4) for w in ws]),
        submission=dict(path=str(out), bytes=out.stat().st_size, sha256=sha256(out),
                        sha256_pixels=sha256_pixels(out),
                        grid=grid, finite_px=int(np.isfinite(q).sum()), total_px=int(q.size),
                        nonzero_px=int(np.count_nonzero(np.nan_to_num(q))),
                        min=float(np.nanmin(q)), max=float(np.nanmax(q)),
                        mean=float(np.nanmean(q)),
                        readback=written),
        local_score_vs_known=local,
        discovery_vs_known_catalog=discovery,
    )
    rp = Path(args.report) if args.report else out.with_name(out.stem + "_report.json")
    rp.write_text(json.dumps(report, indent=1))
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
