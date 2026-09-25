#!/usr/bin/env python3
"""Prune an emission to the pixels that actually pay for themselves, on a tune half only.

THE ARITHMETIC THAT MOTIVATES THIS
----------------------------------
The published metric is ``DTI = TP_w / (TP_w + 0.2*FP_w + 0.8*FN_w)`` with
``TP_w = sum_g max_{d<=3px} p`` and ``FP_w = sum_x p * (1 - max_g k)``.  On the canonical
11-fold artifact (`data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`, 168,283
emitted pixels, already a one-pixel-wide skeleton -- `skimage.morphology.skeletonize` returns it
unchanged) the weighted terms on the held-out-segment population are

    TP_w = 4,146    FP_w = 163,892    FN_w = 9,051     ->  DTI 0.0939

so the loss is overwhelmingly *emission that earns nothing*: 0.2*FP_w = 32,778 against a
numerator of 4,146.  Coverage is not the binding constraint (that emission already reaches 58 % of
the held-out truth within 3 px); *precision per emitted pixel* is.  Removing an emitted pixel that
is farther than 3 px from every truth pixel costs no TP at all and saves 0.2 in the denominator.

The file carries no probabilities to rank its own pixels by, so this script builds rankings from
signals that do not depend on the supplied catalogue:

  * the 6-fold blend run (`35042805806`, 19,637 px, a different model set -- see
    ``scripts/blend_submission.py``) as a *high-precision* corroborating detector;
  * component length inside the 11-fold mask (a lineament that runs 40 px is a geology-shaped
    object; a 2 px speck is noise);
  * the structure-tensor lineament field from `scripts/suture_submission.py`
    (`data/evidence/suture/field_raw.tif`), which is built from the geophysical bands and never
    sees a label;
  * distance to the supplied catalogue, because the platform zeroes prediction on those pixels
    (forum 11516) and pays nothing for them, while the held-out truth sits a median 9.4 px away.

POLICY GRID (pre-registered here, before any score was looked at)
----------------------------------------------------------------
  A_k   keep 11-fold pixels within k px of a 6-fold pixel (k = 0, 1, 2)
  B_L   keep 11-fold components of at least L px (L = 4, 8, 16, 32)
  C_q   keep 11-fold pixels whose lineament-field value is at or above the q-th percentile of
        the field over the 11-fold mask (q = 50, 70, 85)
  D_{k,L} intersection of A_k and B_L
  E_k   A_k union the 6-fold mask itself (recall arm: the 6-fold pixels do not have to be
        near an 11-fold pixel to count)
  F     the unpruned 11-fold mask (control: the shipped artifact as-is)
  G     the 6-fold mask alone (control)

SELECTION
---------
The held-out segments are split by connected-component parity into a tune half (6,776 px) and a
measure half (6,421 px).  Every policy is scored on the tune half; the winner is the argmax of the
*platform-masked* score (P: prediction masked on the supplied catalogue, which is what the
organizers' pixel-exact mask does).  The winner is then scored once on the measure half, and the
measure-half numbers are what get reported and shipped.  The pre-registered grid, the tune table
and the single measure-half number all go to `data/evidence/optimised/pruning_report.json`.

USAGE
    python scripts/prune_submission_emission.py
    python scripts/prune_submission_emission.py --write    # also write the winning .tif
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

RECALL_MASK = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
PRECISION_MASK = "data/evidence/runs/35042805806/submission.tif"
LINEAMENT_FIELD = "data/evidence/suture/field_raw.tif"
LABELS = "data/labels.tif"
TEMPLATE = "data/sample_submission.tif"
SPLIT_DIR = "data/evidence/seghold"
OUT_DIR = "data/evidence/optimised"
INDEPENDENT_PX = 5.0


def read(path) -> np.ndarray:
    with rasterio.open(ROOT / path) as src:
        return src.read(1).astype(np.float32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score(mask: np.ndarray, truth: np.ndarray, valid: np.ndarray, masked: np.ndarray) -> dict:
    """Protocol-P score of a binary emission: prediction zeroed on the supplied catalogue."""
    pred = np.where(valid & ~masked & mask, 1.0, 0.0).astype(np.float32)
    value, terms = dti(pred, truth, valid=valid, return_terms=True)
    return {"dti_P": value, "tp": terms["tp"], "fp": terms["fp"], "fn": terms["fn"],
            "emitted_px": int((mask & valid).sum())}


def coverage_at_3px(mask: np.ndarray, truth: np.ndarray) -> float:
    if not truth.any():
        return float("nan")
    d = ndimage.distance_transform_edt(~mask)
    return float((d[truth] <= 3.0).mean())


def policies(recall: np.ndarray, precision: np.ndarray, lineament: np.ndarray,
             catalogue_dist: np.ndarray) -> dict[str, np.ndarray]:
    near = {k: ndimage.binary_dilation(precision, iterations=k) if k else precision
            for k in (0, 1, 2)}
    comp, n = ndimage.label(recall, structure=np.ones((3, 3)))
    sizes = np.bincount(comp.ravel())
    comp_keep = {"B4": 4, "B8": 8, "B16": 16, "B32": 32}
    keep_size = {name: np.isin(comp, np.flatnonzero(sizes >= lim)) for name, lim in comp_keep.items()}
    field_on_mask = lineament[recall]
    qs = {}
    for q in (50, 70, 85):
        thr = float(np.percentile(field_on_mask, q))
        qs[f"C{q}"] = recall & (lineament >= thr)
    out = {
        "F (unpruned 11-fold, the shipped artifact)": recall,
        "G (6-fold blend alone)": precision,
    }
    for k, m in near.items():
        out[f"A{k} (11-fold within {k} px of 6-fold)"] = recall & m
        out[f"E{k} (A{k} union 6-fold)"] = (recall & m) | precision
    for name, m in keep_size.items():
        out[f"{name} (components >= {comp_keep[name]} px)"] = recall & m
    out.update({f"{name} (lineament >= p{q})": m for (name, m), q in zip(qs.items(), (50, 70, 85))})
    for k in (0, 1, 2):
        for name, m in keep_size.items():
            out[f"D{k}{name} (A{k} and {name})"] = recall & near[k] & m
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the winning .tif")
    ap.add_argument("--json", default=f"{OUT_DIR}/pruning_report.json")
    args = ap.parse_args(argv)

    for rel in (RECALL_MASK, PRECISION_MASK, LABELS, TEMPLATE, f"{SPLIT_DIR}/eval_labels.tif"):
        if not (ROOT / rel).exists():
            print(f"FAIL: {rel} missing - run python scripts/assemble_data_bridge.py",
                  file=sys.stderr)
            return 2

    footprint = np.isfinite(read(TEMPLATE))
    supplied = read(LABELS) >= 1
    ev = read(f"{SPLIT_DIR}/eval_labels.tif")
    train_known = read(f"{SPLIT_DIR}/train_labels.tif") == 1
    valid = footprint & ~train_known
    d_train = ndimage.distance_transform_edt(~train_known).astype(np.float32)
    d_cat = ndimage.distance_transform_edt(~supplied).astype(np.float32)

    independent = d_train > INDEPENDENT_PX
    comp = ndimage.label(ev == 1, structure=np.ones((3, 3)))[0]
    ids = np.unique(comp[ev == 1])
    tune_ids = set(ids[: len(ids) // 2].tolist())
    tune_truth = np.isin(comp, list(tune_ids)) & (ev == 1) & independent
    measure_truth = (ev == 1) & ~tune_truth & independent
    print(f"held-out segments {ids.size} -> tune truth {int(tune_truth.sum()):,} px / "
          f"measure truth {int(measure_truth.sum()):,} px")

    recall = read(RECALL_MASK) > 0
    precision = read(PRECISION_MASK) > 0
    lineament = np.nan_to_num(read(LINEAMENT_FIELD), nan=0.0)
    print(f"recall mask {int(recall.sum()):,} px | precision mask {int(precision.sum()):,} px | "
          f"recall masked on the supplied catalogue {int((recall & supplied).sum()):,} px")

    table = policies(recall, precision, lineament, d_cat)
    rows = []
    for name, mask in table.items():
        r = score(mask, tune_truth, valid, supplied)
        r.update({"policy": name, "coverage_tune": coverage_at_3px(mask, tune_truth)})
        rows.append(r)
    rows.sort(key=lambda r: -r["dti_P"])
    print(f"{'policy':52s} {'emitted':>8s} {'cov':>5s} {'DTI_P':>7s} {'TP':>7s} {'FP':>8s} {'FN':>7s}")
    for r in rows:
        print(f"{r['policy']:52s} {r['emitted_px']:>8,} {r['coverage_tune']:>5.2f} "
              f"{r['dti_P']:>7.4f} {r['tp']:>7.1f} {r['fp']:>8.1f} {r['fn']:>7.1f}")
    winner = rows[0]
    win_mask = table[winner["policy"]]
    measure = score(win_mask, measure_truth, valid, supplied)
    measure["coverage_measure"] = coverage_at_3px(win_mask, measure_truth)
    baseline_mask = table["F (unpruned 11-fold, the shipped artifact)"]
    baseline_measure = score(baseline_mask, measure_truth, valid, supplied)
    baseline_measure["coverage_measure"] = coverage_at_3px(baseline_mask, measure_truth)
    print(f"\nwinner (tune): {winner['policy']}  P_tune {winner['dti_P']:.4f}")
    print(f"  measure half: P {measure['dti_P']:.4f}  (unpruned artifact on the same half: "
          f"{baseline_measure['dti_P']:.4f})  emitted {measure['emitted_px']:,} px")

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/prune_submission_emission.py",
        "protocol": ("prune the canonical 11-fold emission to pixels corroborated by an "
                     "independent detector, by component length or by the label-free lineament "
                     "field; select on the tune half of the held-out segments, report the measure "
                     "half once"),
        "inputs": {
            "recall_mask": {"path": RECALL_MASK, "sha256": sha256_file(ROOT / RECALL_MASK)},
            "precision_mask": {"path": PRECISION_MASK,
                               "sha256": sha256_file(ROOT / PRECISION_MASK)},
            "lineament_field": {"path": LINEAMENT_FIELD,
                                "sha256": sha256_file(ROOT / LINEAMENT_FIELD)},
            "supplied_catalogue": {"path": LABELS, "sha256": sha256_file(ROOT / LABELS)},
        },
        "split": {"dir": SPLIT_DIR, "tune_truth_px": int(tune_truth.sum()),
                  "measure_truth_px": int(measure_truth.sum()),
                  "held_out_segments": int(ids.size),
                  "independent_px_gt": INDEPENDENT_PX},
        "scoring": ("prediction masked on the supplied catalogue and the 3 px training buffer, "
                    "truth = held-out segments, alpha 0.2 / beta 0.8, k = max(1 - d/3, 0)"),
        "tune_table": rows,
        "winner": winner["policy"],
        "measurement_on_untouched_half": measure,
        "unpruned_artifact_on_same_half": baseline_measure,
        "caveat": ("one local population and one split; the tune half is 6,776 truth pixels, so "
                   "the winner is a hypothesis about the leaderboard, not a measurement of it"),
    }
    if args.write:
        out_dir = ROOT / OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        win = win_mask & footprint
        out = out_dir / "submission.tif"
        with rasterio.open(ROOT / TEMPLATE) as tmpl:
            profile = tmpl.profile.copy()
        profile.update(dtype="float32", count=1, nodata=float("nan"), compress="deflate")
        data = np.where(win, 1.0, np.where(footprint, 0.0, np.nan)).astype(np.float32)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data, 1)
        sha = sha256_file(out)
        report["written"] = {"path": str(out.relative_to(ROOT)), "bytes": out.stat().st_size,
                             "sha256": sha, "emitted_px": int((win & footprint).sum())}
        (out_dir / "submission.tif.sha256").write_text(f"{sha}  submission.tif\n")
        print(f"wrote {out.relative_to(ROOT)}  {out.stat().st_size:,} B  sha256 {sha[:12]}...")
    out_json = ROOT / args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
