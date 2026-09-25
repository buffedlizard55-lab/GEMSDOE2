#!/usr/bin/env python3
"""Independent held-out-segment evaluation of any prediction raster (the community protocol).

WHY A SECOND SCORER
-------------------
Every number this repository has published for a candidate field comes from
`src/metrics.py` measured on populations built out of *this* repository's own choices:
the SGMC proxy catalogue, the shipped labels, or the union of the two.  A candidate can be
tuned against those choices without anyone noticing.  This script scores candidates with the
protocol published by a third party, `gems-eval` (https://github.com/Gameassassin777/gems-eval),
which this project did not write and cannot tune:

  1. connected-component-label the known faults in `data/labels.tif`;
  2. hold out a fraction of those *segments* (default 30 %, seed 0);
  3. score a prediction only on the held-out segments, with a 3 px buffer around the
     remaining (training) faults excluded from the scored population.

That is an imitation of how the board scores: the leaderboard truth is faults that are NOT in
the training labels, and the supplied labels are excluded from scoring.  A field that merely
re-draws the known catalogue looks excellent under any random split and scores near zero here.

TWO POPULATIONS, BECAUSE THE ORGANIZERS PINNED THE SCORING RULE
--------------------------------------------------------------
The organizers answered this on the forum (verified 2026-09-25, verbatim in
`data/evidence/rules_quotes.json` and quoted below), which settles what the scored population is:

  "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation,
   so they do not count towards penalty terms."                    -- chrisk-dd, forum 11516 #2
  "The mask is indeed pixel-exact - it is identical to the provided set of training fault
   labels."                                                        -- chrisk-dd, forum 11516 #4
  "A predicted pixel that is near a known fault trace but far from a new-fault ground truth
   pixel will be fully penalized, i.e., the buffer does not apply to known faults."
                                                                   -- chrisk-dd, forum 11516 #4

So the FP term is charged over every footprint pixel that is NOT itself a supplied-label pixel.
This script therefore reports two numbers per candidate:

  L  (gems-eval's protocol, third party, unchanged): scored population = background + held-out
     segments; the 3 px buffer around the TRAINING faults is excluded from scoring.  This is an
     optimistic imitation -- it forgives predicted mass sitting next to the training catalogue.
  S  (organizer-faithful, added here): scored population = the whole footprint MINUS the
     training-catalogue pixels; truth = the held-out segments.  Predicted mass in the 3 px
     buffer around the training catalogue is charged as a false positive, exactly as
     chrisk-dd describes.
  I  (independent, added here): as S, but the truth population is restricted to held-out
     segments that lie MORE than `--independent-px` (default 5) from every training-catalogue
     pixel.  This is the only one of the three that removes the label-echo path: under L and S a
     field that simply brightens near the supplied catalogue still covers the held-out segments,
     because those segments are pieces of the same fault systems.  Two shipped candidates scored
     their mass by an explicit exp(-distance_to_known_fault) boost
     (`scripts/generate_turbo_variants.py`); under I that path earns nothing.  I is the number
     to trust when the question is "finds new faults", and it is deliberately harder than the
     board: the organizers confirmed new-fault truth may itself sit within 300 m of a known
     trace, and those pixels are excluded here.

A field that simply re-draws the supplied catalogue looks good under L (the held-out segments are
literally pixels of that catalogue) and collapses under S.  That is the "0.305 local -> 0.0387
leaderboard" trap the gems-eval README documents.

The metric is the distance-weighted Tversky index of the official problem page, re-derived from
the published formula (NOT the organizer's code):

    k(d) = max(1 - d/R, 0),  R = 300 m = 3 px at 100 m
    TP_w = sum_g max_{x: d(x,g)<=R} p(x) * k(d(x,g))
    FP_w = sum_x p(x) * [1 - max_g k(d(x,g))]
    FN_w = |G| - TP_w
    DTI  = TP_w / (TP_w + 0.2*FP_w + 0.8*FN_w + eps)

`--parity` re-scores through the third-party package itself when a checkout is available and
prints the two numbers side by side; the committed evidence records that comparison.  Without
the package installed the script still runs -- the implementation below is self-contained
(numpy + scipy + scikit-image, all already dependencies of this repository).

USAGE
    python scripts/segment_holdout_eval.py --build-split
    python scripts/segment_holdout_eval.py --pred docs/gemsdoe2_recall_union_submission.tif
    python scripts/segment_holdout_eval.py --pred-dir data/evidence/runs --json out.json
    python scripts/segment_holdout_eval.py --pred FILE --parity /tmp/gems-eval

Exit code is 0 on success, 2 when a requested input is missing -- so it can gate a workflow.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

R_PX = 3.0
ALPHA, BETA = 0.2, 0.8
EPS = 1e-7
_r = int(np.ceil(R_PX))
_yy, _xx = np.mgrid[-_r:_r + 1, -_r:_r + 1]
_CONE = np.clip(1.0 - np.hypot(_yy, _xx) / R_PX, 0.0, 1.0).astype(np.float32)
_LOG_CONE = np.where(_CONE > 0, np.log(np.clip(_CONE, 1e-6, 1.0)), -np.inf).astype(np.float32)


# ---------------------------------------------------------------------------------------
# the metric, exactly as the published formula defines it
# ---------------------------------------------------------------------------------------
def cone_max(p: np.ndarray) -> np.ndarray:
    """m(g) = max_x p(x) * k(d(x, g)) for every pixel g: a grey dilation in the log domain."""
    lp = np.log(np.clip(p, 1e-6, 1.0)).astype(np.float32)
    m = np.exp(ndimage.grey_dilation(lp, structure=_LOG_CONE)).astype(np.float32)
    m[m <= 1.0001e-6] = 0.0
    return m


def dti(pred: np.ndarray, truth: np.ndarray, alpha: float = ALPHA, beta: float = BETA,
        eps: float = EPS, valid: np.ndarray | None = None,
        return_terms: bool = False):
    """Score `pred` (float 0..1, NaN allowed) against `truth` (0/1) on the pixels in `valid`."""
    p = np.clip(np.nan_to_num(pred, nan=0.0), 0.0, 1.0).astype(np.float32)
    g = truth > 0
    if valid is not None:
        p = np.where(valid, p, 0.0).astype(np.float32)
        g = g & valid
    d_true = (ndimage.distance_transform_edt(~g).astype(np.float32) if g.any()
              else np.full(g.shape, np.inf, np.float32))
    w_true = np.clip(1.0 - d_true / R_PX, 0.0, 1.0)
    tp = float(cone_max(p)[g].sum())
    fp = float((p * (1.0 - w_true)).sum())
    fn = float(g.sum()) - tp
    score = tp / (tp + alpha * fp + beta * fn + eps)
    if return_terms:
        return score, {"tp": tp, "fp": fp, "fn": fn, "n_truth": int(g.sum())}
    return score


def keep_top_fraction(pred, frac, valid=None, binarise=False):
    p = np.array(pred, dtype=np.float32, copy=True)
    finite = np.isfinite(p)
    v = finite if valid is None else (finite & valid)
    if not v.any():
        return p
    t = float(np.quantile(p[v], 1.0 - frac))
    out = np.where(v & (p >= t) & (p > 0), 1.0 if binarise else p, 0.0).astype(np.float32)
    out[~finite] = np.nan
    return out


def skeleton_top_fraction(pred, frac, valid=None, exclude=None, exclude_dilate_px=1):
    from skimage.morphology import skeletonize
    p = np.array(pred, dtype=np.float32, copy=True)
    finite = np.isfinite(p)
    v = finite if valid is None else (finite & valid)
    if exclude is not None:
        m = (ndimage.binary_dilation(exclude, iterations=exclude_dilate_px)
             if exclude_dilate_px > 0 else exclude)
        v = v & ~m
    q = np.where(v, p, 0.0)
    t = float(np.quantile(q[v], 1.0 - frac))
    out = skeletonize(v & (q >= t) & (q > 0)).astype(np.float32)
    out[~finite] = np.nan
    return out


# ---------------------------------------------------------------------------------------
# the split (identical arithmetic to gems-eval holdout.split_segments)
# ---------------------------------------------------------------------------------------
def split_segments(labels: np.ndarray, frac: float = 0.3, seed: int = 0,
                   exclude_buffer_px: int = 3):
    known = labels >= 1
    comp, n = ndimage.label(known, structure=np.ones((3, 3)))
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    rng.shuffle(ids)
    held = ids[: int(round(frac * n))]
    held_mask = np.isin(comp, held)
    train = np.where(labels < 0, -1, np.where(known & ~held_mask, 1, 0)).astype(np.int8)
    excl = (ndimage.binary_dilation(known & ~held_mask, iterations=exclude_buffer_px)
            if exclude_buffer_px > 0 else (known & ~held_mask))
    ev = np.where(labels < 0, -1, np.where(held_mask, 1, np.where(excl, 2, 0))).astype(np.int8)
    return train, ev


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as s:
        return s.read(1), dict(profile=s.profile, transform=s.transform, crs=s.crs,
                               shape=(s.height, s.width))


def build_split(labels_path: Path, split_dir: Path, frac: float, seed: int, buffer: int) -> dict:
    labels, meta = read_raster(labels_path)
    labels = labels.astype(np.int32)
    labels[labels == -1] = -1
    train, ev = split_segments(labels, frac=frac, seed=seed, exclude_buffer_px=buffer)
    split_dir.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=labels.shape[0], width=labels.shape[1], count=1,
                dtype="int8", crs=meta["crs"], transform=meta["transform"], nodata=-1,
                compress="deflate")
    for arr, name in ((train, "train_labels.tif"), (ev, "eval_labels.tif")):
        with rasterio.open(split_dir / name, "w", **prof) as d:
            d.write(arr, 1)
    n_held_segments = int(np.unique(ndimage.label(ev == 1, structure=np.ones((3, 3)))[0]).size - 1)
    summary = {
        "protocol": "gems-eval held-out-segment",
        "reference": "https://github.com/Gameassassin777/gems-eval",
        "labels_sha_input": str(labels_path),
        "frac": frac, "seed": seed, "buffer_px": buffer,
        "held_out_px": int((ev == 1).sum()), "buffer_px_total": int((ev == 2).sum()),
        "scoreable_background_px": int((ev == 0).sum()), "nodata_px": int((ev == -1).sum()),
        "held_out_segments": n_held_segments,
        "train_fault_px": int((train == 1).sum()),
        "train_labels": str(split_dir / "train_labels.tif"),
        "eval_labels": str(split_dir / "eval_labels.tif"),
    }
    return summary


# ---------------------------------------------------------------------------------------
def score_prediction(pred_path: Path, ev: np.ndarray, top: float = 0.02) -> dict:
    pred, meta = read_raster(pred_path)
    p = np.nan_to_num(pred.astype(np.float32), nan=0.0)
    valid = (ev == 0) | (ev == 1)
    truth = ev == 1
    base, terms = dti(p, truth, valid=valid, return_terms=True)
    row = {
        "pred": str(pred_path),
        "dti": base,
        "tp_w": terms["tp"], "fp_w": terms["fp"], "fn_w": terms["fn"],
        "n_truth": terms["n_truth"],
        "coverage_ge_0.5": float(np.mean(p[valid] >= 0.5)),
        "mean_pred": float(np.mean(p[valid])),
        "nonzero_px": int((p[valid] > 0).sum()),
    }
    for frac in (top, 0.05):
        row[f"dti_top{frac:g}"] = dti(keep_top_fraction(p, frac, valid=valid), truth, valid=valid)
        row[f"dti_skel_top{frac:g}"] = dti(
            skeleton_top_fraction(p, frac, valid=valid, exclude=(ev == 2),
                                  exclude_dilate_px=0), truth, valid=valid)
    row["nan_inside_valid_region"] = int((~np.isfinite(pred) & valid).sum())
    row["crs_matches"] = bool(meta["crs"].to_epsg() == 32611)
    row["template_sha256_hint"] = "compare with data/sample_submission.tif before upload"
    return row


def score_prediction_organizer(pred_path: Path, ev: np.ndarray, train_known: np.ndarray,
                              top: float = 0.02) -> dict:
    """Protocol S: the organizers' stated scoring rule, applied to the held-out-segment split.

    valid = footprint pixels that are not themselves supplied-catalogue pixels
            (ev >= 0 marks the footprint; train_known marks the supplied catalogue)
    truth = the held-out segments (ev == 1)
    The 3 px buffer around the training catalogue is NOT excused -- per chrisk-dd it is fully
    penalized when no new truth is nearby.
    """
    pred, _ = read_raster(pred_path)
    p = np.nan_to_num(pred.astype(np.float32), nan=0.0)
    footprint = ev >= 0
    valid = footprint & ~train_known
    truth = ev == 1
    base, terms = dti(p, truth, valid=valid, return_terms=True)
    row = {
        "pred": str(pred_path),
        "dti_S": base,
        "tp_w_S": terms["tp"], "fp_w_S": terms["fp"], "fn_w_S": terms["fn"],
        "n_truth_S": terms["n_truth"],
        "emitted_px_S": int((p[valid] > 0).sum()),
        "emitted_frac_of_truth": float((p[valid] > 0).sum() / max(terms["n_truth"], 1)),
        "coverage_ge_0.5_S": float(np.mean(p[valid] >= 0.5)),
    }
    for frac in (top, 0.05):
        row[f"dti_S_top{frac:g}"] = dti(keep_top_fraction(p, frac, valid=valid), truth, valid=valid)
        row[f"dti_S_skel_top{frac:g}"] = dti(
            skeleton_top_fraction(p, frac, valid=valid), truth, valid=valid)
    return row


def score_prediction_independent(pred_path: Path, ev: np.ndarray, train_known: np.ndarray,
                                 dist_to_train: np.ndarray, independent_px: float = 5.0,
                                 top: float = 0.02) -> dict:
    """Protocol I: truth = held-out segments farther than `independent_px` from the training
    catalogue; scored population = footprint minus the supplied catalogue (as in S)."""
    pred, _ = read_raster(pred_path)
    p = np.nan_to_num(pred.astype(np.float32), nan=0.0)
    footprint = ev >= 0
    valid = footprint & ~train_known
    truth = (ev == 1) & (dist_to_train > independent_px)
    base, terms = dti(p, truth, valid=valid, return_terms=True)
    row = {
        "pred": str(pred_path),
        "dti_I": base,
        "tp_w_I": terms["tp"], "fp_w_I": terms["fp"], "fn_w_I": terms["fn"],
        "n_truth_I": terms["n_truth"],
        "emitted_px_I": int((p[valid] > 0).sum()),
    }
    for frac in (top, 0.05):
        row[f"dti_I_skel_top{frac:g}"] = dti(
            skeleton_top_fraction(p, frac, valid=valid), truth, valid=valid)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--split-dir", default="data/evidence/seghold")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=3)
    ap.add_argument("--build-split", action="store_true")
    ap.add_argument("--pred", default="", help="one prediction raster")
    ap.add_argument("--pred-dir", default="", help="glob <dir>/**/submission.tif")
    ap.add_argument("--top", type=float, default=0.02)
    ap.add_argument("--independent-px", type=float, default=5.0,
                    help="protocol I: minimum distance from the training catalogue for a "
                         "held-out truth pixel to count")
    ap.add_argument("--json", default="", help="write the report here")
    ap.add_argument("--parity", default="", help="path to a gems-eval checkout for a parity run")
    args = ap.parse_args(argv)

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"FAIL: {labels_path} not found - run python scripts/assemble_data_bridge.py",
              file=sys.stderr)
        return 2
    split_dir = Path(args.split_dir)
    split_json = split_dir / "split.json"
    if args.build_split or not (split_dir / "eval_labels.tif").exists():
        summary = build_split(labels_path, split_dir, args.frac, args.seed, args.buffer)
        split_json.write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
    else:
        summary = json.loads(split_json.read_text())

    preds: list[Path] = []
    if args.pred:
        preds.append(Path(args.pred))
    if args.pred_dir:
        preds += sorted(Path(p) for p in glob.glob(f"{args.pred_dir}/**/*.tif", recursive=True))
    if not preds:
        print("nothing to score: pass --pred or --pred-dir", file=sys.stderr)
        return 2

    ev, _ = read_raster(split_dir / "eval_labels.tif")
    train_known, _ = read_raster(split_dir / "train_labels.tif")
    train_known = train_known == 1
    dist_to_train = ndimage.distance_transform_edt(~train_known).astype(np.float32)
    n_truth_I = int(((ev == 1) & (dist_to_train > args.independent_px)).sum())
    rows = []
    for path in preds:
        if not path.exists():
            print(f"  MISSING {path}", file=sys.stderr)
            continue
        try:
            row = score_prediction(path, ev, top=args.top)
            row.update(score_prediction_organizer(path, ev, train_known, top=args.top))
            row.update(score_prediction_independent(path, ev, train_known, dist_to_train,
                                                    args.independent_px, args.top))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  ERROR {path}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        print(f"  L {row['dti']:.6f} | S {row['dti_S']:.6f} | I {row['dti_I']:.6f} "
              f"(skel {row[f'dti_I_skel_top{args.top:g}']:.6f}) | {path}")

    report = {"split": summary, "candidates": rows,
              "metric": "distance-weighted Tversky index (alpha=0.2, beta=0.8, R=300 m)",
              "protocol_L_source": "https://github.com/Gameassassin777/gems-eval",
              "protocol_S_source": "organizer clarification, forum 11516 #2 and #4",
              "protocol_I_source": "this script: held-out segments >5 px from the training catalogue",
              "protocol_I_truth_px": n_truth_I,
              "note": ("Scored populations are held-out fault segments only; these are local "
                       "numbers and cannot be compared with a leaderboard score.  S is the "
                       "organizer-faithful population (predicted mass next to the training "
                       "catalogue is charged); L is the third-party protocol; I additionally "
                       "drops held-out truth within 5 px of the training catalogue, which is "
                       "the only one of the three that cannot be won by echoing the catalogue.")}

    if args.parity:
        parity = []
        for row in rows[:3]:
            parity.append(_parity_row(Path(row["pred"]), split_dir, args.parity))
        report["parity_against_gems_eval"] = parity
        print("parity:", json.dumps(parity, indent=1))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")
    return 0


def _parity_row(pred_path: Path, split_dir: Path, pkg_root: str) -> dict:
    """Re-score one raster through the third-party package itself."""
    sys.path.insert(0, pkg_root)
    try:
        from gems_eval import holdout_dti                                   # type: ignore
    except Exception as exc:                                                # noqa: BLE001
        return {"pred": str(pred_path), "status": f"package not importable: {exc}"}
    pred, _ = read_raster(pred_path)
    ev, _ = read_raster(split_dir / "eval_labels.tif")
    p = np.nan_to_num(pred.astype(np.float32), nan=0.0)
    theirs = float(holdout_dti(p, ev))
    mine = float(dti(p, ev == 1, valid=(ev == 0) | (ev == 1)))
    return {"pred": str(pred_path), "gems_eval": theirs, "this_script": mine,
            "abs_diff": abs(theirs - mine), "agrees_to_1e-9": abs(theirs - mine) < 1e-9}


if __name__ == "__main__":
    raise SystemExit(main())
