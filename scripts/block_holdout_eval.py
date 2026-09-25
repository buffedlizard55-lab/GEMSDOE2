#!/usr/bin/env python3
"""Block-stratified evaluation of an emission policy: where does the score come from, and is
the advantage uniform across geography or carried by a few blocks?

WHY THIS EXISTS
---------------
Every number the shipped emission policy rests on is a SINGLE global figure on one population:

    proxy DTI 0.1365 (ensemble 1) / 0.0999 (shipping field) / 0.0850 (3-ensemble field)
    contrast  +0.0954 / +0.0695 / +0.0581 against that field's own reference policy
    (data/evidence/proxy/eval_sweep*.json, data/evidence/emission_decision.json)

A global DTI over 61,664 proxy-only pixels spread across 2,083 components in a 51,857 km^2
survey carries no published error bar.  Two different fields can produce the same global number
for opposite reasons: a small uniform gain, or a large gain in one basin that the mean hides.
Selecting a policy on a scalar with unknown variance is how a project ships something that does
not travel - and this competition scores geography nobody has seen (rules §1.1/§3.3, verified
verbatim in data/evidence/rules_quotes.json).

This script adds the missing instrument using only data already committed to the repository:

1. **Exact block decomposition.**  The grid is cut into `--block-px` blocks (512 px = 51.2 km at
   the official 100 m resolution) and the metric's three sums are aggregated per block WITH
   GLOBAL CONTEXT, so the parts sum exactly to the published global score.  `score_within_mask`
   (src/metrics.py) is the reference implementation of one block's terms and is pinned against a
   from-the-definition brute force in tests/test_block_decomposition.py; the vectorised
   aggregation used here for all 56 blocks is pinned against it in
   tests/test_block_holdout_eval.py.  Cropping a block instead would under-credit truth near the
   boundary and over-penalise predictions whose support sits just outside it - exactly at the
   boundaries a spatial hold-out is meant to measure.
2. **Both populations on the same blocks.**  Catalogue labels (`labels.tif`) and the
   new-fault-like proxy-only pixels (`data/evidence/proxy/proxy_catalogue.tif`, code 2) are
   scored block by block.  That converts the in-domain/proxy sign conflict recorded globally in
   docs/METRIC_STRATEGY.md into a per-block count: in how many 51 km blocks does widening help
   the proxy population while hurting the catalogue population?
3. **Block bootstrap.**  Because the three sums are additive, resampling BLOCKS with replacement
   and recomposing TP/(TP + a*FP + b*FN + eps) is an exact bootstrap of the published index.
   Blocks, not pixels, are the resampling unit: fault traces run for kilometres, so neighbouring
   pixels are not independent draws and a pixel bootstrap would understate the interval.

WHAT IT CANNOT DO (stated so the output is never over-read)
-----------------------------------------------------------
* It scores the SHIPPED raster, so only candidates recoverable from it are swept: floors at or
  above the floor already applied (0.1) and hard bands widened from the shipped support.  Lower
  floors, un-thinned variants and re-shaped raw ensemble fields need the runner
  (.github/workflows/proxy-eval.yml, which re-blends the saved fold artifacts).
* The shipped model trained on ALL catalogue labels, so per-block catalogue DTI is an in-domain
  number, not a hold-out.  Per-block PROXY-only DTI is honest (those pixels are absent from the
  training labels by construction).  The true spatial hold-out - retraining with whole blocks
  excluded - is src/blocks.py + configs/config_block_holdout.yaml +
  .github/workflows/block-holdout.yml.

USAGE
    python scripts/block_holdout_eval.py \
        --pred data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif \
        --labels data/labels.tif \
        --proxy data/evidence/proxy/proxy_catalogue.tif \
        --out data/evidence/block_holdout/block_stratified.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (DEFAULT_BLOCK_PX, DEFAULT_N_FOLDS, assign_folds, block_id_map,  # noqa: E402
                        block_table, describe_partition)
from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS, EPS,  # noqa: E402
                         GtContext, block_aggregate, bootstrap_from_blocks, score_within_mask)

# The per-block aggregation and the block bootstrap live in src/metrics.py (shared with
# scripts/measure_cross_catalogue_transfer.py and unit-tested there); these aliases keep the
# names this script's tests and reports use.
block_components = block_aggregate

KM_PER_PIXEL = 0.1
PROXY_CODE_ONLY = 2          # data/evidence/proxy/proxy_stats.json: 2 = no training label within R
RESOLUTION_M = 100.0


# --------------------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_band(path: Path) -> tuple:
    with rasterio.open(path) as src:
        arr = src.read(1)
        grid = dict(width=src.width, height=src.height, crs=str(src.crs),
                    transform=[float(v) for v in src.transform][:6],
                    nodata=(None if src.nodata is None else float(src.nodata)),
                    dtype=str(arr.dtype), sha256=sha256(path),
                    bytes=int(Path(path).stat().st_size))
    return arr, grid


def same_grid(a: dict, b: dict) -> bool:
    return (a["width"], a["height"]) == (b["width"], b["height"]) and \
        np.allclose(a["transform"], b["transform"], atol=1e-6)


def finite_mask(pred_raw: np.ndarray, template_raw: Optional[np.ndarray]) -> np.ndarray:
    """The data footprint: where a legal submission may carry finite values.

    The problem page requires "data outside bounds is null or NaN" and the shipped raster is NaN
    over 57.93 % of the grid, so the footprint comes from the submission template when one is
    supplied, and from the prediction's own finite values otherwise.
    """
    src = template_raw if template_raw is not None else pred_raw
    return np.isfinite(np.asarray(src, dtype=np.float64))


# --------------------------------------------------------------------------------------
# candidates (generated lazily: a full-grid float64 emission is ~98 MB, so 50 of them at once
# would not fit in the 3.9 GB development sandbox that has to be able to run this)
# --------------------------------------------------------------------------------------
def candidate_emissions(pred: np.ndarray, footprint: np.ndarray, floors: Sequence[float],
                        widths: Sequence[int], reference: Optional[dict] = None) -> Iterator[dict]:
    """Hard-band candidates recoverable from an already-shaped submission, ONE AT A TIME.

    For a floor f the support is `pred >= f` inside the footprint; for a width w the emission is
    the support grown to w px and written as a HARD band of 1.0 - the same family the runner
    sweep measures (rows with `hard: true` in data/evidence/proxy/eval_sweep*.json), so the two
    are comparable in kind.  One Euclidean distance transform per floor serves every width,
    which is exactly equivalent to iterated dilation by a disk and far cheaper.

    Laziness is a memory requirement, not a style choice: a full-grid float64 emission is
    ~98 MB on the 3292x3730 competition raster, so the 50-candidate default grid would need
    ~4.9 GB held at once and would OOM the 3.9 GB development sandbox that must be able to run
    this script.  Each candidate is scored as it is produced and then dropped.
    """
    p = np.clip(np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0), 0.0, 1.0)
    labels = []
    for f in floors:
        support = (p >= float(f)) & footprint
        edt = distance_transform_edt(~support) if support.any() else None
        for w in widths:
            if w == 0:
                emis = support
            elif edt is None:
                emis = np.zeros(p.shape, bool)
            else:
                emis = support | (edt <= float(w))
            label = f"floor{float(f):g}_w{int(w)}px"
            labels.append(label)
            yield dict(floor=float(f), width=int(w), thin=True, hard=True, label=label,
                       emission=emis.astype(np.float64), support_px=int(np.count_nonzero(emis)),
                       # a cheap content key, so two candidates that produce the SAME emission can
                       # be reported as one measurement instead of being counted twice
                       emission_key=hashlib.sha1(np.ascontiguousarray(emis).view(np.uint8)).hexdigest()[:16])
        del edt
    if reference is not None:
        ref = f"floor{reference['floor']:g}_w{reference['width']}px"
        if ref not in labels:
            raise ValueError(f"reference candidate {ref} is not in the swept grid {labels}")


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------
def score_candidates(pred: np.ndarray, footprint: np.ndarray, truths: Dict[str, np.ndarray],
                     floors: Sequence[float], widths: Sequence[int], blocks: np.ndarray,
                     n_blocks: int, R: int, alpha: float, beta: float, eps: float,
                     crosscheck_blocks: int = 2) -> Dict[str, dict]:
    """Score every recoverable candidate on every population, block by block.

    Each candidate is built, scored on all populations and dropped, so peak memory is one
    emission plus the per-population geometry (see `candidate_emissions`).
    """
    ctxs = {name: GtContext(t, R_pixels=R) for name, t in truths.items()}
    fpw = {name: ctxs[name].fp_weight() for name in truths}
    rows: Dict[str, List[dict]] = {name: [] for name in truths}
    for c in candidate_emissions(pred, footprint, floors, widths):
        emis = c["emission"]
        for name, ctx in ctxs.items():
            dti_glob, (TP_g, FP_g, FN_g) = ctx.score(emis, alpha=alpha, beta=beta, eps=eps,
                                                     return_components=True)
            comp = block_components(ctx, emis, blocks, n_blocks, fpw[name])
            # cross-check the vectorised aggregation against the reference per-block
            # implementation on the first `crosscheck_blocks` scoreable blocks
            checked = 0
            for b in range(n_blocks):
                if checked >= crosscheck_blocks:
                    break
                if comp["n_gt"][b] == 0:
                    continue
                ref = score_within_mask(emis, ctx, blocks == b, alpha=alpha, beta=beta, eps=eps,
                                        credit=comp["credit"])
                for key, got in (("TP_w", comp["TP"][b]), ("FP_w", comp["FP"][b]),
                                 ("FN_w", comp["FN"][b])):
                    if not np.isclose(got, ref[key], rtol=1e-9, atol=1e-9):
                        raise AssertionError(f"{name} block {b} {key}: vectorised {got} != "
                                             f"reference {ref[key]}")
                checked += 1
            per_block = []
            for b in range(n_blocks):
                ngt = int(comp["n_gt"][b])
                denom = comp["TP"][b] + alpha * comp["FP"][b] + beta * comp["FN"][b] + eps
                per_block.append(dict(
                    block=int(b), dti=(round(float(comp["TP"][b] / denom), 6) if ngt else None),
                    TP_w=float(comp["TP"][b]), FP_w=float(comp["FP"][b]),
                    FN_w=float(comp["FN"][b]), n_gt=ngt,
                    pred_px=int(comp["pred_px"][b]), scoreable=bool(ngt > 0)))
            row = dict(label=c["label"], floor=c["floor"], width=c["width"],
                       support_px=int(c["support_px"]), emission_key=c.get("emission_key"),
                       global_dti=(round(float(dti_glob), 6) if dti_glob is not None else None),
                       TP_w=float(TP_g), FP_w=float(FP_g), FN_w=float(FN_g),
                       n_gt=int(ctx.n_gt), pred_px=int(np.count_nonzero(emis > 0)),
                       blocks=per_block)
            tp = sum(b["TP_w"] for b in per_block)
            fp = sum(b["FP_w"] for b in per_block)
            fn = sum(b["FN_w"] for b in per_block)
            ngt = sum(b["n_gt"] for b in per_block)
            if not (np.isclose(tp, row["TP_w"], rtol=1e-9, atol=1e-8) and
                    np.isclose(fp, row["FP_w"], rtol=1e-9, atol=1e-8) and
                    np.isclose(fn, row["FN_w"], rtol=1e-9, atol=1e-8) and ngt == row["n_gt"]):
                raise AssertionError(
                    f"{name}/{row['label']}: block components do not sum to the global score "
                    f"(TP {tp} vs {row['TP_w']}, FP {fp} vs {row['FP_w']}, FN {fn} vs {row['FN_w']}, "
                    f"n_gt {ngt} vs {row['n_gt']})")
            rows[name].append(row)
        del emis
    return {name: dict(n_gt=int(ctxs[name].n_gt),
                       truth_km=round(int(ctxs[name].n_gt) * KM_PER_PIXEL, 3),
                       alpha=alpha, beta=beta, R_pixels=R, eps=eps, candidates=rows[name])
            for name in truths}


def block_bootstrap(rows: List[dict], alpha: float, beta: float, eps: float,
                    n_boot: int = 2000, seed: int = 0) -> Dict[str, dict]:
    """Percentile bootstrap over blocks (src.metrics.bootstrap_from_blocks).

    The first row is the reference candidate, so `prob_beats_reference` is the paired bootstrap
    probability that a candidate beats the reference on the SAME resampled set of blocks.
    """
    return bootstrap_from_blocks(rows, alpha=alpha, beta=beta, eps=eps, n_boot=n_boot, seed=seed)


def per_block_sign_agreement(labels_rows: List[dict], proxy_rows: List[dict],
                             reference_label: str) -> dict:
    """Do the two populations prefer the same candidate in the same 51 km blocks?

    For every candidate the per-block contrast against the reference is computed on BOTH
    populations and the signs compared.  `blocks_proxy_gain` counts blocks where the candidate
    beats the reference on the new-fault-like population - the quantity the widening decision
    turns on.  The conflict counters require BOTH signs to be known before they count anything:
    `blocks_conflict_labels_lose_proxy_gains` is the number of blocks where the catalogue
    population loses AND the new-fault-like population gains for the same candidate, which is the
    conflict docs/METRIC_STRATEGY.md could previously only record globally.  (An earlier draft
    counted the labels side alone, which overstated the conflict - the counters are pinned by
    tests/test_block_holdout_eval.py::test_conflict_counters_require_both_signs.)
    """
    L = {r["label"]: r for r in labels_rows}
    P = {r["label"]: r for r in proxy_rows}
    if reference_label not in L or reference_label not in P:
        raise ValueError(f"reference candidate {reference_label!r} was not scored on both populations")
    lref = {b["block"]: b for b in L[reference_label]["blocks"]}
    pref = {b["block"]: b for b in P[reference_label]["blocks"]}
    out = []
    for label in L:
        lb = {b["block"]: b for b in L[label]["blocks"]}
        both = agree = proxy_up = labels_down = scoreable = 0
        conflict_labels_lose = conflict_proxy_lose = 0
        worst_proxy = None
        for blk, b in sorted({b["block"]: b for b in P[label]["blocks"]}.items()):
            if not b["scoreable"] or not pref[blk]["scoreable"]:
                continue
            cp = b["dti"] - pref[blk]["dti"]
            scoreable += 1
            if cp > 0:
                proxy_up += 1
            if worst_proxy is None or cp < worst_proxy:
                worst_proxy = cp
            lbb, lrefb = lb.get(blk), lref.get(blk)
            if lbb is None or lrefb is None or not lbb["scoreable"] or not lrefb["scoreable"]:
                continue
            cl = lbb["dti"] - lrefb["dti"]
            both += 1
            if cl < 0:
                labels_down += 1
            if cl < 0 < cp:
                conflict_labels_lose += 1
            elif cp < 0 < cl:
                conflict_proxy_lose += 1
            if (cp > 0) == (cl > 0):
                agree += 1
        out.append(dict(label=label, blocks_scoreable_proxy=scoreable,
                        blocks_proxy_gain=int(proxy_up),
                        blocks_both_populations=both,
                        blocks_agree_in_sign=int(agree),
                        blocks_labels_lose=int(labels_down),
                        blocks_conflict_labels_lose_proxy_gains=int(conflict_labels_lose),
                        blocks_conflict_proxy_lose_labels_gain=int(conflict_proxy_lose),
                        agreement_fraction=(round(agree / both, 4) if both else None),
                        worst_block_contrast_proxy=(round(float(worst_proxy), 6)
                                                    if worst_proxy is not None else None)))
    return dict(reference=reference_label, per_candidate=out)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def combined_truth_provenance(labels_path: Path, proxy_path: Path, n_labels: int,
                              n_proxy_only: int, n_combined: int) -> dict:
    """What the `combined` population IS, and what it is only a surrogate for.

    Phase 2 of the prize scores the SAME submission against "all fault labels in the updated
    set" (official rules 3.6, verbatim in data/evidence/rules_quotes.json): the existing mapped
    faults PLUS the faults the expert panel adds after reviewing the predictions.  Neither
    committed population is that set on its own - `labels` is only the existing mapped faults and
    `proxy_only` is only the new-fault-like part - so a decision that trades one against the other
    (the SGMC pseudo-label experiment: proxy DTI up, catalogue DTI down) cannot be read from
    either arm alone.  Their union is the closest thing this repository can build from bytes it
    already has, and it is a SURROGATE: the real updated set is written by experts after seeing
    the predictions, so it is neither SGMC nor QFaults.
    """
    return dict(
        definition="labels.tif fault pixels UNION proxy_catalogue.tif code-2 pixels, inside the footprint",
        inputs=dict(labels=str(labels_path), proxy=str(proxy_path), proxy_code=PROXY_CODE_ONLY),
        px=dict(labels=int(n_labels), proxy_only=int(n_proxy_only), combined=int(n_combined)),
        disjoint=bool(n_labels + n_proxy_only == n_combined),
        surrogacy=("surrogate for the Phase-2 'complete updated test set' (rules 3.6: the same "
                   "submission is scored again against all fault labels in the set the expert "
                   "panel updates after the competition closes).  It is NOT that set: the updated "
                   "set is written by experts, partly FROM the predictions, so it can contain "
                   "faults no free catalogue maps and can omit SGMC traces the panel rejects."),
        why_not_decomposable=("FP_w is a sum over PREDICTION pixels of 1 - max_g k(d(x,g)), so the "
                              "combined FP_w cannot be recomposed from the two component "
                              "populations' FP_w (TP_w and FN_w can: both are sums over truth "
                              "pixels, and the two truth sets are disjoint).  The combined "
                              "population is therefore scored from the rasters, never inferred "
                              "from the two committed reports."),
    )


def build_report(pred_path: Path, labels_path: Path, proxy_path: Path,
                 template_path: Optional[Path], block_px: int, n_folds: int,
                 floors: Sequence[float], widths: Sequence[int], reference: dict,
                 n_boot: int, seed: int, R: int, alpha: float, beta: float, eps: float,
                 score_fold: Optional[int] = None, complement: bool = False,
                 combined_population: bool = False) -> dict:
    pred_raw, pred_grid = read_band(pred_path)
    labels_raw, labels_grid = read_band(labels_path)
    proxy_raw, proxy_grid = read_band(proxy_path)
    template_raw, template_grid = read_band(template_path) if template_path else (None, None)
    for name, g in (("labels", labels_grid), ("proxy", proxy_grid), ("template", template_grid)):
        if g is not None and not same_grid(pred_grid, g):
            raise SystemExit(f"{name} raster is not on the prediction grid ({g['width']}x{g['height']} "
                             f"vs {pred_grid['width']}x{pred_grid['height']}) - every input must share "
                             "the competition grid (problem page #submission-format)")
    footprint = finite_mask(pred_raw, template_raw)
    pred = np.clip(np.nan_to_num(np.asarray(pred_raw, dtype=np.float64), nan=0.0), 0.0, 1.0)
    labels = (np.nan_to_num(np.asarray(labels_raw, dtype=np.float64), nan=0.0) > 0.5) & footprint
    proxy_codes = np.nan_to_num(np.asarray(proxy_raw, dtype=np.float64), nan=0.0).astype(np.int16)
    proxy_only = (proxy_codes == PROXY_CODE_ONLY) & footprint
    # The optional THIRD population: existing mapped faults + new-fault-like trace, i.e. the
    # closest local surrogate for the Phase-2 "complete updated test set" (rules 3.6).  It is
    # off by default so every committed report keeps the schema its tests pin.
    combined = (labels | proxy_only) if combined_population else None

    if not labels.any():
        raise SystemExit("labels contain no fault pixels inside the footprint - nothing to score")
    if not proxy_only.any():
        raise SystemExit(f"proxy raster has no code-{PROXY_CODE_ONLY} pixels inside the footprint; "
                         "the new-fault-like population is empty (data/evidence/proxy/proxy_stats.json)")

    # ---- block partition: the SAME geometry training holds out (src/blocks.py) ---------
    shape = labels.shape
    tab = block_table(shape, block_px, valid=footprint, labels=labels)
    fold_of = assign_folds(tab, n_folds, seed=0, mode="balanced")
    blocks = block_id_map(shape, block_px)                    # every pixel belongs to a block
    n_blocks = len(tab)
    partition = describe_partition(shape, block_px, n_folds, 0, tab, fold_of,
                                   labels=labels, valid=footprint, mode="balanced")
    # keep only the blocks that can be scored (footprint present); the rest carry no truth and
    # no legal submission may emit there, but they stay in the array so the sums stay exact
    scored_ids = [int(t["block_id"]) for t in tab if int(t["valid_px"]) > 0]

    # ---- optional restriction to ONE fold's blocks ------------------------------------
    # This is what turns the report from "how does reshaping the shipped raster score?" into
    # "how does a model that never saw these 51 km blocks score on them?".  It is only a
    # generalisation reading when the prediction was produced by a model trained with
    # training.holdout=spatial_blocks and training.block_fold=score_fold (configs/
    # config_block_holdout.yaml); scoring the SHIPPED ensemble this way measures nothing about
    # generalisation, because that ensemble saw every block.  The restriction is recorded in the
    # report either way, so a number can never be quoted without its scope.
    restriction = dict(mode="all_blocks", score_fold=None, complement=False,
                       blocks_included=len(scored_ids), blocks_scoreable=len(scored_ids),
                       blocks_excluded=0,
                       px_included=int(footprint.sum()), px_excluded=0,
                       note=("every scoreable block; this is a RESHAPING measurement on whatever "
                             "model produced --pred, not a generalisation measurement"))
    if score_fold is not None:
        if not 0 <= int(score_fold) < n_folds:
            raise SystemExit(f"--score-fold {score_fold} is outside [0,{n_folds}) - the partition "
                             "has no such fold")
        fold_blocks = sorted(b for b, f in fold_of.items() if int(f) == int(score_fold))
        if complement:
            fold_blocks = sorted(set(scored_ids) - set(fold_blocks))
        if not fold_blocks:
            raise SystemExit(f"--score-fold {score_fold}{' --complement' if complement else ''} "
                             "selected no blocks")
        keep = np.isin(blocks, np.asarray(fold_blocks, dtype=blocks.dtype))
        before_px = int(footprint.sum())
        footprint = footprint & keep
        labels = labels & footprint
        proxy_only = proxy_only & footprint
        if combined_population:
            combined = labels | proxy_only      # both operands are already restricted
        if not labels.any() or not proxy_only.any():
            raise SystemExit(f"the fold-{score_fold} restriction left no truth pixels "
                             f"(labels {int(labels.sum())}, proxy-only {int(proxy_only.sum())}) - "
                             "nothing to score in these blocks")
        tab = [t for t in tab if int(t["block_id"]) in set(fold_blocks)]
        scored_ids = [int(t["block_id"]) for t in tab if int(t["valid_px"]) > 0]
        restriction = dict(
            mode=("fold_complement" if complement else "fold_heldout"),
            score_fold=int(score_fold), complement=bool(complement),
            blocks_included=len(fold_blocks),
            blocks_excluded=int(n_blocks - len(fold_blocks)),
            blocks_scoreable=len(scored_ids),
            px_included=int(footprint.sum()), px_excluded=int(before_px - footprint.sum()),
            blocks=fold_blocks,
            note=(f"only the blocks of fold {score_fold}"
                  + (" (the blocks the model TRAINED on)" if complement else
                     " (the blocks held out from that fold's training)")
                  + "; this is a generalisation reading ONLY if --pred came from a model trained "
                    "with training.holdout=spatial_blocks and training.block_fold="
                    f"{score_fold} on the same partition (block_px={block_px}, folds={n_folds}, "
                    "seed 0, mode balanced)"),
        )

    finite_pred = pred[footprint]
    uniq = np.unique(finite_pred)
    pred_profile = dict(
        px=int(finite_pred.size),
        minv=round(float(finite_pred.min()), 6) if finite_pred.size else None,
        maxv=round(float(finite_pred.max()), 6) if finite_pred.size else None,
        nonzero_px=int(np.count_nonzero(finite_pred)),
        distinct_values=int(uniq.size),
        frac_at_max=round(float((finite_pred >= finite_pred.max() - 1e-12).mean()), 6)
        if finite_pred.size else None,
        hard_band=bool(uniq.size <= 2),
        note=("A shaped submission written with hard=True carries two values (0 and the band "
              "value), so raising the floor above the applied floor selects the SAME support: "
              "the floor axis is degenerate for this input and only the width axis is measured. "
              "Sweeping floors needs the pre-shaping ensemble field, which only the runner has "
              "(.github/workflows/proxy-eval.yml re-blends the saved fold artifacts).")
        if uniq.size <= 2 else
        ("The input carries a continuous probability field, so both the floor and the width axis "
         "are measured."),
    )

    # ---- candidates -------------------------------------------------------------------
    # The reference candidate must be FIRST in the swept order, because block_bootstrap treats
    # row 0 as the reference every contrast is paired against.  Sweeping in (floor, width) order
    # and asserting the reference's position keeps that a property of the code, not of the CLI.
    ref_label = f"floor{reference['floor']:g}_w{reference['width']}px"
    floors = sorted(set(float(f) for f in floors))
    widths = sorted(set(int(w) for w in widths))
    if float(reference["floor"]) not in floors or int(reference["width"]) not in widths:
        raise ValueError(f"reference candidate {ref_label} is outside the swept grid "
                         f"(floors {floors}, widths {widths})")
    truths = {"labels": labels.astype(np.float64),
              "proxy_only": proxy_only.astype(np.float64)}
    if combined_population:
        truths["combined"] = combined.astype(np.float64)
    pops = score_candidates(pred, footprint, truths,
                            floors, widths, blocks, n_blocks, R, alpha, beta, eps)
    for name, pop in pops.items():
        pop["candidates"].sort(key=lambda c: (c["label"] != ref_label, c["floor"], c["width"]))
        if pop["candidates"][0]["label"] != ref_label:
            raise AssertionError(f"{name}: reference candidate is not first "
                                 f"({pop['candidates'][0]['label']} != {ref_label})")
    pop_labels, pop_proxy = pops["labels"], pops["proxy_only"]
    pop_combined = pops.get("combined")
    boot_proxy = block_bootstrap(pop_proxy["candidates"], alpha, beta, eps, n_boot=n_boot, seed=seed)
    boot_labels = block_bootstrap(pop_labels["candidates"], alpha, beta, eps, n_boot=n_boot, seed=seed)
    boot_combined = (block_bootstrap(pop_combined["candidates"], alpha, beta, eps,
                                     n_boot=n_boot, seed=seed) if pop_combined else None)
    agreement = per_block_sign_agreement(pop_labels["candidates"], pop_proxy["candidates"], ref_label)

    proxy_by = {c["label"]: c for c in pop_proxy["candidates"]}
    labels_by = {c["label"]: c for c in pop_labels["candidates"]}
    combined_by = ({c["label"]: c for c in pop_combined["candidates"]} if pop_combined else {})
    ref_proxy = proxy_by[ref_label]["global_dti"]
    best_proxy = max((c for c in pop_proxy["candidates"] if c["global_dti"] is not None),
                     key=lambda c: c["global_dti"])
    agree_ref = next(c for c in agreement["per_candidate"] if c["label"] == ref_label)
    boot_ref = boot_proxy[ref_label]
    keys = [c.get("emission_key") for c in pop_proxy["candidates"]]
    distinct = len({k for k in keys if k})
    first_of = {}
    for c in pop_proxy["candidates"]:
        first_of.setdefault(c.get("emission_key"), c["label"])
    for pop in tuple(p for p in (pop_proxy, pop_labels, pop_combined) if p is not None):
        for c in pop["candidates"]:
            c["duplicate_of"] = (None if first_of.get(c.get("emission_key")) == c["label"]
                                 else first_of.get(c.get("emission_key")))
    alternatives = [c for c in pop_proxy["candidates"]
                    if c["label"] != ref_label and c["duplicate_of"] is None
                    and c["global_dti"] is not None]
    best_alt = max(alternatives, key=lambda c: c["global_dti"]) if alternatives else None
    best_alt_agree = next((c for c in agreement["per_candidate"]
                           if best_alt and c["label"] == best_alt["label"]), None)
    verdict = dict(
        reference_candidate=ref_label,
        n_candidates_swept=len(pop_proxy["candidates"]),
        n_distinct_emissions=distinct,
        floor_axis_degenerate=bool(pred_profile["hard_band"]),
        best_alternative_candidate=(best_alt["label"] if best_alt else None),
        best_alternative_proxy_dti=(best_alt["global_dti"] if best_alt else None),
        best_alternative_contrast=(round(best_alt["global_dti"] - ref_proxy, 6) if best_alt else None),
        best_alternative_prob_beats_reference=(
            boot_proxy[best_alt["label"]]["prob_beats_reference"] if best_alt else None),
        best_alternative_worst_block_contrast=(
            best_alt_agree["worst_block_contrast_proxy"] if best_alt_agree else None),
        reference_proxy_dti=ref_proxy,
        reference_proxy_ci95=boot_ref["dti_ci95"],
        reference_labels_dti=labels_by[ref_label]["global_dti"],
        reference_combined_dti=(combined_by[ref_label]["global_dti"] if combined_by else None),
        best_swept_candidate=best_proxy["label"],
        best_swept_proxy_dti=best_proxy["global_dti"],
        best_swept_contrast=round(best_proxy["global_dti"] - ref_proxy, 6),
        reference_is_best_of_sweep=bool(best_proxy["label"] == ref_label),
        worst_block_contrast_proxy=agree_ref["worst_block_contrast_proxy"],
        blocks_where_reference_gains=agree_ref["blocks_proxy_gain"],
        blocks_scoreable_proxy=agree_ref["blocks_scoreable_proxy"],
        note=("Candidates are derived from the SHIPPED raster, so floors below the applied floor "
              "and un-thinned variants are not recoverable here; the runner sweep "
              "(.github/workflows/proxy-eval.yml) covers those.  Per-block catalogue DTI is "
              "in-domain (the shipped model trained on every label); per-block proxy-only DTI is "
              "not, because those pixels are absent from the training labels by construction."),
    )
    combined_section = None
    if combined_by:
        best_comb = max((c for c in pop_combined["candidates"] if c["global_dti"] is not None),
                        key=lambda c: c["global_dti"])
        n_lab, n_pro, n_com = int(labels.sum()), int(proxy_only.sum()), int(combined.sum())
        for nm, got, want in (("labels", n_lab, pop_labels["n_gt"]),
                              ("proxy_only", n_pro, pop_proxy["n_gt"]),
                              ("combined", n_com, pop_combined["n_gt"])):
            if got != want:
                raise AssertionError(f"combined population: {nm} mask has {got} px but the scored "
                                     f"population reports {want} - the mask and the score drifted")
        ref_comb = combined_by[ref_label]
        combined_section = dict(
            provenance=combined_truth_provenance(labels_path, proxy_path, n_lab, n_pro, n_com),
            reference=dict(label=ref_label, dti=ref_comb["global_dti"],
                           ci95=boot_combined[ref_label]["dti_ci95"],
                           support_px=ref_comb["support_px"], TP_w=ref_comb["TP_w"],
                           FP_w=ref_comb["FP_w"], FN_w=ref_comb["FN_w"]),
            best_swept=dict(label=best_comb["label"], dti=best_comb["global_dti"],
                            support_px=best_comb["support_px"],
                            contrast_vs_reference=round(best_comb["global_dti"]
                                                        - ref_comb["global_dti"], 6),
                            prob_beats_reference=boot_combined[best_comb["label"]][
                                "prob_beats_reference"]),
            component_dti=dict(labels=labels_by[ref_label]["global_dti"],
                               proxy_only=proxy_by[ref_label]["global_dti"],
                               combined=ref_comb["global_dti"]),
            caveat=("best_swept is a maximum over the swept grid ON THE JUDGING POPULATION: "
                    "upward biased, never a shipping recommendation (the same caveat "
                    "docs/FIELD_SELECTION_RULE.md records).  The combined population is a "
                    "surrogate, not the Phase-2 set - see provenance.surrogacy."),
        )
    return dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/block_holdout_eval.py",
        purpose=("block-stratified error bars on the emission policy, on the catalogue and on the "
                 "new-fault-like proxy population, from an exact additive decomposition of the "
                 "official metric plus a block bootstrap"),
        inputs=dict(pred=str(pred_path), labels=str(labels_path), proxy=str(proxy_path),
                    template=(str(template_path) if template_path else None), pred_grid=pred_grid),
        metric=dict(R_pixels=R, R_meters=R * RESOLUTION_M, alpha=alpha, beta=beta, eps=eps,
                    source="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric"),
        blocks=dict(block_px=block_px, block_km=round(block_px * KM_PER_PIXEL, 3),
                    n_blocks=n_blocks, n_blocks_scoreable=len(scored_ids), block_ids=scored_ids,
                    partition=partition),
        footprint=dict(px=int(footprint.sum()), frac_of_grid=round(float(footprint.mean()), 6)),
        prediction=pred_profile,
        restriction=restriction,
        populations=dict(labels=pop_labels, proxy_only=pop_proxy,
                         **({"combined": pop_combined} if pop_combined else {})),
        bootstrap=dict(n=n_boot, seed=seed, unit="spatial block, resampled with replacement, "
                                                 "recomposed from the additive metric components",
                       reliability=bootstrap_reliability(len(scored_ids), n_blocks, n_boot,
                                                         restriction),
                       proxy_only=boot_proxy, labels=boot_labels,
                       **({"combined": boot_combined} if boot_combined else {})),
        **({"combined_population": combined_section} if combined_section else {}),
        population_agreement=agreement,
        interpretation=per_block_caveat_from(verdict, agreement, pop_proxy),
        verdict=verdict,
    )


MIN_BLOCKS_FOR_A_READABLE_CI = 12


def bootstrap_reliability(n_scoreable: int, n_blocks: int, n_boot: int, restriction: dict) -> dict:
    """How much a block-bootstrap interval from THIS many blocks is worth.

    Measured 2026-09-18 on the real grid, and the reason this function exists: restricting the
    measurement to one fold's blocks leaves 8 scoreable blocks, and in those 8 blocks widening to
    8 px looked BETTER than the reference (0.0852 vs 0.0785, P(beats reference) = 0.66) while the
    same artifact over all 34 scoreable blocks says widening is harmful (0.0878 at 1 px,
    P = 0.008).  A bootstrap that resamples 8 units cannot separate a regional preference from
    sampling noise, and quoting its interval next to a 34-block interval invites exactly that
    mistake.  So the report states the resampling-unit count, whether the interval is readable, and
    what a sign flip at this sample size means.
    """
    readable = n_scoreable >= MIN_BLOCKS_FOR_A_READABLE_CI
    out = dict(
        resampling_units=n_scoreable,
        blocks_in_partition=n_blocks,
        n_boot=n_boot,
        min_units_for_a_readable_ci=MIN_BLOCKS_FOR_A_READABLE_CI,
        interval_readable=readable,
        scope=(restriction or {}).get("mode", "all_blocks"),
    )
    if readable:
        out["note"] = (f"{n_scoreable} independent resampling units: the CI95 is readable and "
                       "comparable with other runs of this script that score a similar number of "
                       "blocks. It quantifies spatial sampling noise in this footprint, not "
                       "uncertainty over truth definitions or over the hidden set.")
    else:
        out["note"] = (
            f"ONLY {n_scoreable} independent resampling units (below "
            f"{MIN_BLOCKS_FOR_A_READABLE_CI}), so the interval is COARSE: a bootstrap that redraws "
            f"{n_scoreable} blocks cannot separate a regional preference from sampling noise. A "
            "candidate whose contrast flips sign against the full-grid measurement is expected "
            "sampling variation at this sample size, NOT evidence that the policy should change; "
            "the global full-grid number remains the selection statistic. This happens whenever the "
            "measurement is restricted (--score-fold) or the footprint is small.")
        out["warning"] = ("sign flips against the full-grid run are sampling noise at "
                          f"{n_scoreable} resampling units")
    return out


def prune_duplicate_block_rows(rep: dict) -> dict:
    """Drop the per-block rows of candidates whose emission is byte-identical to another's.

    A hard-band submission makes the floor axis degenerate: measured 2026-09-18, 50 swept
    candidates produced only 10 DISTINCT emissions, so 40 of them carried per-block rows that are
    an exact copy of their twin's.  Everything that consumes those rows (the bootstrap, the sign
    agreement table, the verdict, the reproduction check) has already run by the time this is
    called, so pruning loses no measurement - it only stops the committed evidence from being 4x
    bigger than the information it contains.  Each pruned candidate keeps its global numbers and
    names the twin whose rows stand in for its own.
    """
    pruned = 0
    for pop in rep["populations"].values():
        for cand in pop["candidates"]:
            twin = cand.get("duplicate_of")
            if twin and cand.get("blocks"):
                cand["blocks_note"] = (f"emission identical to {twin}; {len(cand['blocks'])} "
                                       "per-block rows omitted because they are an exact copy of "
                                       "that candidate's rows (see verdict.floor_axis_degenerate)")
                cand["blocks"] = []
                pruned += 1
    rep["pruning"] = dict(candidates_pruned=pruned,
                          rule="per-block rows dropped only for emissions that duplicate another "
                               "candidate's emission; global numbers and the twin reference kept",
                          applied_after="bootstrap, sign agreement, verdict and the reproduction "
                                        "check (none of which read a duplicate's rows)")
    return rep


def load_config(path: Path) -> tuple:
    """Read a YAML config -> (config, block_holdout_eval section, conflict, note).

    The partition cross-check is the point of this function: `training.block_px` /
    `training.block_folds` describe the split the MODEL was trained against, while
    `block_holdout_eval.block_px` / `.folds` describe the split this SCORE is computed on.  If
    they differ the report would claim a block-holdout number for a partition the training never
    held out, which is exactly the overstatement this script exists to prevent.
    """
    if not path.exists():
        raise SystemExit(f"--config {path} does not exist")
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is required for --config (pip install pyyaml)")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"--config {path} must be a YAML mapping at the top level")
    bh = dict(cfg.get("block_holdout_eval") or {})
    tr = dict(cfg.get("training") or {})
    conflict = None
    for eval_key, train_key in (("block_px", "block_px"), ("folds", "block_folds")):
        ev, tv = bh.get(eval_key), tr.get(train_key)
        if ev is not None and tv is not None and int(ev) != int(tv):
            conflict = (f"configs disagree about the partition: block_holdout_eval.{eval_key}={ev} "
                        f"but training.{train_key}={tv} in {path} - the score would be computed on "
                        "a partition the model was not trained against, which is not a holdout "
                        "measurement.  Make them equal (or drop --config and pass the flags).")
            break
    # A note, not an error: scoring a model that trained on every block is legitimate (it is what
    # the shipped submission is), but the report must not be read as a generalisation measurement.
    note = None
    if "holdout" in tr and str(tr.get("holdout")) != "spatial_blocks":
        note = (f"{path} sets training.holdout={tr.get('holdout')!r}: the model it trains saw every "
                "block, so per-block numbers here measure RESHAPING, not generalisation to unseen "
                "blocks.  Set training.holdout: spatial_blocks for the holdout reading.")
    return cfg, bh, conflict, note


def crosscheck_against_committed_sweep(rep: dict, sweep_path: Path, reference: dict) -> dict:
    """Assert the local measurement reproduces a committed runner sweep, digit for digit.

    The runner sweep (.github/workflows/proxy-eval.yml -> scripts/eval_proxy_catalogue.py) shaped
    the pre-shaping ensemble mean field on a GitHub-hosted runner; this script scores the SHIPPED
    raster in the development sandbox with an independently written per-block decomposition.  If
    the two agree on DTI and on all three metric components, then (a) the shipped bytes really are
    the policy the sweep measured, and (b) the sandbox can reproduce runner evidence offline.
    A missing file or a missing row is recorded as `not_available` - never as a pass.
    """
    out = dict(sweep_file=str(sweep_path), reference_candidate=rep["verdict"]["reference_candidate"])
    if not sweep_path.exists():
        out.update(status="not_available", reason=f"{sweep_path} does not exist")
        return out
    try:
        sweep = json.loads(sweep_path.read_text())
    except Exception as exc:                                          # noqa: BLE001
        out.update(status="not_available", reason=f"unreadable: {type(exc).__name__}: {exc}")
        return out
    rows = ((sweep.get("results") or {}).get("shaping_sweep")) or []
    match = [r for r in rows
             if abs(float(r.get("t0", -1)) - float(reference["floor"])) < 1e-9
             and int(r.get("dilate", -1)) == int(reference["width"])
             and bool(r.get("thin", False)) and not bool(r.get("soft", False))]
    if not match:
        out.update(status="not_available",
                   reason=f"no row with t0={reference['floor']}, dilate={reference['width']}, "
                          f"thin=True, soft=False among {len(rows)} sweep rows")
        return out
    row = match[0]
    local = next(c for c in rep["populations"]["proxy_only"]["candidates"]
                 if c["label"] == rep["verdict"]["reference_candidate"])
    diffs = dict(dti=abs(float(row["dti"]) - float(local["global_dti"])),
                 TP_w=abs(float(row["TP_w"]) - float(local["TP_w"])),
                 FP_w=abs(float(row["FP_w"]) - float(local["FP_w"])),
                 FN_w=abs(float(row["FN_w"]) - float(local["FN_w"])),
                 emission_px=abs(int(row.get("emission_px", -1)) - int(local["pred_px"])))
    tol = dict(dti=1e-6, TP_w=1e-3, FP_w=1e-3, FN_w=1e-3, emission_px=0)
    ok = all(diffs[k] <= tol[k] for k in tol)
    out.update(status=("reproduced" if ok else "MISMATCH"),
               runner_row=dict(t0=row["t0"], thin=row["thin"], dilate=row["dilate"],
                               soft=row.get("soft", False), dti=row["dti"], TP_w=row["TP_w"],
                               FP_w=row["FP_w"], FN_w=row["FN_w"],
                               emission_px=row.get("emission_px")),
               sandbox_row=dict(dti=local["global_dti"], TP_w=round(local["TP_w"], 6),
                                FP_w=round(local["FP_w"], 6), FN_w=round(local["FN_w"], 6),
                                emission_px=local["pred_px"]),
               abs_difference={k: round(v, 9) for k, v in diffs.items()},
               tolerance=tol,
               runner_sha256=(sweep.get("inputs") or {}).get("pred_sha256"),
               sandbox_input_sha256=rep["inputs"]["pred_grid"]["sha256"],
               meaning=("the shipped raster scored in the sandbox equals the policy the runner "
                        "sweep measured, so the committed evidence and the local numbers describe "
                        "the same bytes"))
    return out


def per_block_caveat(rep: dict) -> dict:
    """Is the per-block majority preference the same as the global preference?

    Measured 2026-09-18 on the shipped raster: widening to 2 px LOSES globally on the proxy
    population (0.0999 -> 0.0823) yet WINS in 32 of 33 scoreable blocks.  DTI is not decomposable
    - TP_w is a sum over truth pixels while the FP penalty is a global mass term - so a per-block
    argmax is NOT a valid policy selector.  Blocks are for variance (the bootstrap) and for
    locating where a score comes from; the global figure remains the selection statistic.  This
    function derives that statement from the numbers instead of asserting it in prose.
    """
    agree = rep["population_agreement"]["per_candidate"]
    ref = rep["verdict"]["reference_candidate"]
    conflicts = [c for c in agree if c["label"] != ref and c["blocks_proxy_gain"] and
                 c["blocks_proxy_gain"] > c["blocks_scoreable_proxy"] / 2.0]
    globally_worse = []
    proxy_by = {c["label"]: c for c in rep["populations"]["proxy_only"]["candidates"]}
    ref_dti = proxy_by[ref]["global_dti"]
    for c in conflicts:
        row = proxy_by[c["label"]]
        if row["duplicate_of"] is None and row["global_dti"] is not None and row["global_dti"] < ref_dti:
            globally_worse.append(dict(label=c["label"], global_dti=row["global_dti"],
                                       reference_global_dti=ref_dti,
                                       blocks_gaining=c["blocks_proxy_gain"],
                                       blocks_scoreable=c["blocks_scoreable_proxy"]))
    return dict(
        per_block_majority_conflicts_with_global=globally_worse,
        n_conflicts=len(globally_worse),
        interpretation=("DTI is not decomposable over blocks: TP_w sums over truth pixels while "
                        "the FP penalty is a global mass term, so a candidate can win in most "
                        "blocks and still lose globally.  Per-block numbers are for VARIANCE "
                        "(the block bootstrap) and for locating where a score comes from; the "
                        "global DTI stays the selection statistic.  "
                        + ("This run measured that conflict: "
                           + ", ".join(f"{g['label']} wins {g['blocks_gaining']}/{g['blocks_scoreable']} "
                                       f"blocks but loses globally ({g['global_dti']:.4f} < "
                                       f"{g['reference_global_dti']:.4f})" for g in globally_worse)
                           if globally_worse else
                           "No candidate won a majority of blocks while losing globally in this run.")),
    )


def per_block_caveat_from(verdict: dict, agreement: dict, pop_proxy: dict) -> dict:
    """`per_block_caveat` for a report that is still being assembled."""
    return per_block_caveat(dict(population_agreement=agreement, verdict=verdict,
                                 populations=dict(proxy_only=pop_proxy)))


def print_report(rep: dict) -> None:
    v, b = rep["verdict"], rep["blocks"]
    print(f"\nblocks: {b['block_px']} px = {b['block_km']} km; {b['n_blocks_scoreable']} of "
          f"{b['n_blocks']} scoreable ({b['partition']['empty_blocks_excluded']} footprint-empty)")
    print(f"footprint: {rep['footprint']['px']:,} px ({rep['footprint']['frac_of_grid'] * 100:.2f} % of the grid)")
    print(f"populations: catalogue {rep['populations']['labels']['n_gt']:,} px "
          f"({rep['populations']['labels']['truth_km']:,.1f} km) | proxy-only "
          f"{rep['populations']['proxy_only']['n_gt']:,} px "
          f"({rep['populations']['proxy_only']['truth_km']:,.1f} km)"
          + (f" | combined {rep['populations']['combined']['n_gt']:,} px "
             f"({rep['populations']['combined']['truth_km']:,.1f} km)"
             if "combined" in rep["populations"] else ""))
    print(f"\nreference candidate {v['reference_candidate']}:")
    print(f"  proxy-only DTI {v['reference_proxy_dti']:.4f}  block-bootstrap CI95 "
          f"[{v['reference_proxy_ci95'][0]:.4f}, {v['reference_proxy_ci95'][1]:.4f}]")
    print(f"  catalogue  DTI {v['reference_labels_dti']:.4f}  (in-domain - see the note in the JSON)")
    cs = rep.get("combined_population")
    if cs:
        cd = cs["component_dti"]
        print(f"  combined   DTI {cd['combined']:.4f}  block-bootstrap CI95 "
              f"[{cs['reference']['ci95'][0]:.4f}, {cs['reference']['ci95'][1]:.4f}]  "
              f"(Phase-2-like surrogate: labels {cd['labels']:.4f} + proxy-only {cd['proxy_only']:.4f}; "
              f"{cs['provenance']['px']['combined']:,} truth px, disjoint="
              f"{cs['provenance']['disjoint']})")
    print(f"  worst single-block contrast {v['worst_block_contrast_proxy']:+.4f}; "
          f"{v['blocks_where_reference_gains']} of {v['blocks_scoreable_proxy']} scoreable blocks gain")
    print(f"  best of the recoverable sweep: {v['best_swept_candidate']} "
          f"{v['best_swept_proxy_dti']:.4f} ({v['best_swept_contrast']:+.4f}); "
          f"reference is best: {v['reference_is_best_of_sweep']}")
    print(f"  candidates swept {v['n_candidates_swept']}, DISTINCT emissions "
          f"{v['n_distinct_emissions']} (floor axis degenerate: {v['floor_axis_degenerate']})")
    if v["best_alternative_candidate"]:
        print(f"  best genuinely different alternative: {v['best_alternative_candidate']} "
              f"{v['best_alternative_proxy_dti']:.4f} ({v['best_alternative_contrast']:+.4f}), "
              f"P(beats reference over block resamples) = {v['best_alternative_prob_beats_reference']}, "
              f"worst single block {v['best_alternative_worst_block_contrast']:+.4f}")
    rc = rep.get("reproduction")
    if rc:
        print(f"  reproduction of the committed runner sweep ({Path(rc['sweep_file']).name}): "
              f"{rc['status']}")
        if rc["status"] == "reproduced":
            print(f"    runner  dti {rc['runner_row']['dti']} TP {rc['runner_row']['TP_w']} "
                  f"FP {rc['runner_row']['FP_w']} FN {rc['runner_row']['FN_w']} "
                  f"px {rc['runner_row']['emission_px']}")
            print(f"    sandbox dti {rc['sandbox_row']['dti']} TP {rc['sandbox_row']['TP_w']} "
                  f"FP {rc['sandbox_row']['FP_w']} FN {rc['sandbox_row']['FN_w']} "
                  f"px {rc['sandbox_row']['emission_px']}")
        else:
            print(f"    {rc.get('reason', '')}")
    print(f"  prediction field: {rep['prediction']['distinct_values']} distinct values, "
          f"{rep['prediction']['nonzero_px']:,} emitted px, hard band {rep['prediction']['hard_band']}")
    print("\n  candidate                  proxy DTI   CI95 (block bootstrap)     P(>ref)  catalogue DTI")
    boot = rep["bootstrap"]["proxy_only"]
    for c in rep["populations"]["proxy_only"]["candidates"][:16]:
        bi = boot[c["label"]]
        lab = next(x for x in rep["populations"]["labels"]["candidates"] if x["label"] == c["label"])
        p = bi["prob_beats_reference"]
        print(f"  {c['label']:<26} {c['global_dti']:.4f}      "
              f"[{bi['dti_ci95'][0]:.4f}, {bi['dti_ci95'][1]:.4f}]     "
              f"{('  ref ' if p is None else f'{p:5.2f}')}   "
              f"{(lab['global_dti'] if lab['global_dti'] is not None else float('nan')):.4f}")
    agree = [c["agreement_fraction"] for c in rep["population_agreement"]["per_candidate"]
             if c["agreement_fraction"] is not None]
    interp = rep.get("interpretation") or {}
    if interp.get("n_conflicts"):
        print(f"\n  CAVEAT (derived): {interp['n_conflicts']} candidate(s) win a majority of blocks "
              "yet lose globally - a per-block argmax is not a valid policy selector because DTI "
              "is not decomposable.  Blocks give variance; the global DTI selects.")
        for g in interp["per_block_majority_conflicts_with_global"]:
            print(f"    {g['label']}: wins {g['blocks_gaining']}/{g['blocks_scoreable']} blocks, "
                  f"global {g['global_dti']:.4f} < reference {g['reference_global_dti']:.4f}")
    if agree:
        print(f"\n  mean per-block sign agreement between the two populations: {np.mean(agree):.3f} "
              "(1.00 = they prefer the same candidate in the same block every time)")
        conflict = [c for c in rep["population_agreement"]["per_candidate"]
                    if c["blocks_conflict_labels_lose_proxy_gains"]]
        if conflict:
            worst = max(conflict, key=lambda c: c["blocks_conflict_labels_lose_proxy_gains"])
            print(f"  largest population conflict: {worst['label']} gains on the new-fault-like "
                  f"proxy population in {worst['blocks_conflict_labels_lose_proxy_gains']} of "
                  f"{worst['blocks_both_populations']} blocks where both populations are scoreable "
                  "while the catalogue population loses in the same block")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--template", default="data/sample_submission.tif",
                    help="footprint source (NaN outside the survey area); '' uses the prediction's own finite mask")
    ap.add_argument("--config", default="",
                    help="YAML whose `block_holdout_eval:` section supplies defaults (see "
                         "configs/config_block_holdout.yaml); an explicit CLI flag still wins.  "
                         "The partition parameters are cross-checked against the file's "
                         "`training:` section - a score computed on a partition the model was not "
                         "trained against is not a holdout score")
    # Measurement parameters default to None so `--config` can supply them; BUILTIN fills the rest.
    BUILTIN = dict(block_px=DEFAULT_BLOCK_PX, folds=DEFAULT_N_FOLDS,
                   floors="0.1,0.15,0.2,0.3,0.5", widths="0,1,2,3,4,6,8,12,16,20",
                   reference_floor=0.1, reference_width=0, bootstraps=2000, seed=0,
                   R=DEFAULT_R_PIXELS, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, eps=EPS,
                   crosscheck_sweep="data/evidence/proxy/eval_sweep-mean12.json",
                   combined_population=False,
                   out="data/evidence/block_holdout/block_stratified.json")
    ap.add_argument("--block-px", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None,
                    help="training folds the partition is reported against (the measurement is per block)")
    ap.add_argument("--floors", default=None)
    ap.add_argument("--widths", default=None)
    ap.add_argument("--reference-floor", type=float, default=None)
    ap.add_argument("--reference-width", type=int, default=None)
    ap.add_argument("--bootstraps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--crosscheck-sweep", default=None,
                    help="committed runner sweep whose row for the reference policy must be "
                         "reproduced digit-for-digit ('' to skip)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-fold", type=int, default=None,
                    help="restrict the measurement to the blocks of ONE fold of the partition "
                         "(the blocks that fold held out from training).  Only a generalisation "
                         "reading if --pred came from a model trained with "
                         "training.holdout=spatial_blocks and training.block_fold=this fold")
    ap.add_argument("--complement", action="store_true",
                    help="with --score-fold K: score the OTHER blocks (the ones fold K trained "
                         "on), so the held-out/complement gap can be measured")
    ap.add_argument("--combined-population", action="store_true", default=None,
                    help="also score the UNION of the catalogue labels and the new-fault-like "
                         "proxy pixels as a third population - the closest local surrogate for "
                         "the Phase-2 'complete updated test set' (rules 3.6), and the only "
                         "population on which a change that trades catalogue DTI for proxy DTI "
                         "can be read as a gain or a loss.  Off by default so committed reports "
                         "keep the schema their tests pin")
    ap.add_argument("--keep-duplicate-rows", action="store_true",
                    help="write the per-block rows of duplicate emissions too (4x larger file, "
                         "no extra information on a hard-band submission)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    cfg, bh, conflict, note = load_config(Path(a.config)) if a.config else ({}, {}, None, None)
    if conflict:
        raise SystemExit(conflict)
    for key, builtin in BUILTIN.items():          # explicit CLI flag > config > builtin default
        if getattr(a, key) is None:
            setattr(a, key, bh.get(key, builtin))
    if cfg:
        print(f"config {a.config}: block_holdout_eval={json.dumps(bh, sort_keys=True)}")
        if note:
            print(f"::note:: {note}")

    floors = [float(x) for x in str(a.floors).split(",") if x.strip() != ""]
    widths = [int(x) for x in str(a.widths).split(",") if x.strip() != ""]
    if a.reference_floor not in floors or a.reference_width not in widths:
        raise SystemExit(f"the reference candidate (floor {a.reference_floor}, width {a.reference_width}) "
                         "must be inside the swept grid, otherwise every contrast is against a "
                         "candidate this run did not measure")
    rep = build_report(Path(a.pred), Path(a.labels), Path(a.proxy),
                       Path(a.template) if a.template else None,
                       a.block_px, a.folds, floors, widths,
                       dict(floor=a.reference_floor, width=a.reference_width),
                       a.bootstraps, a.seed, a.R, a.alpha, a.beta, a.eps,
                       a.score_fold, a.complement, bool(a.combined_population))
    if a.crosscheck_sweep:
        rep["reproduction"] = crosscheck_against_committed_sweep(
            rep, Path(a.crosscheck_sweep), dict(floor=a.reference_floor, width=a.reference_width))
        if rep["reproduction"]["status"] == "MISMATCH":
            print("\nREPRODUCTION MISMATCH - the sandbox score does not equal the committed "
                  "runner sweep row:", json.dumps(rep["reproduction"], indent=1))
            raise SystemExit(2)
    if not a.keep_duplicate_rows:
        rep = prune_duplicate_block_rows(rep)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    if not a.quiet:
        print_report(rep)
        print(f"\nwrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
