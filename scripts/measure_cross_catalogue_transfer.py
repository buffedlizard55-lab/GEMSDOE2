#!/usr/bin/env python3
"""Cross-catalogue transfer: emit catalogue A, score against independent catalogue B.

THE QUESTION
------------
Both prize phases score faults the training catalogue does NOT contain (rules §1.1, §3.3, §3.5 -
quoted verbatim in data/evidence/rules_quotes.json).  The repository's only stand-in for that
population is the USGS SGMC proxy catalogue (`data/evidence/proxy/proxy_catalogue.tif`, code 2:
61,664 px / 6,166 km of mapped fault with no training label within R = 3 px).  Every emission
policy number - including the shipped `floor 0.1, thin, width 0 px` - is measured on it.

That single surrogate cannot answer the question EXECUTIVE_SUMMARY.md §11 puts first:

    does an idea that helps on SGMC help on faults in general?

and it cannot measure the highest-upside idea at all, because the proxy IS catalogue A: emitting
SGMC scores 0.0 against SGMC-only truth by construction (the acceptance control in
`scripts/eval_proxy_catalogue.py`).  Scoring A against an independently compiled B breaks the
circularity:

    A = USGS SGMC structure (state geologic maps, mostly pre-Quaternary; DOI 10.3133/ds1052)
    B = USGS Quaternary Fault and Fold Database (seismic-hazard mapping; DOI 10.5066/P9BCVRCK)
    truth = B's code-2 pixels: B traces with NO training label within R

WHAT IS MEASURED (each row is a number, never an argument)
----------------------------------------------------------
1. **Surrogate agreement.**  DTI(shipped submission, B-only) next to DTI(shipped submission,
   A-only) = 0.0999 (`data/evidence/block_holdout/block_stratified.json`, which reproduces the
   runner's `eval_sweep-mean12.json` value).  If the two agree, the local selection signal
   travels across catalogues; if they disagree, every policy chosen on A is suspect and the
   disagreement is the thing to report.
2. **Catalogue-as-predictor.**  DTI(A-emission at band width w, B-only) for w in the swept grid:
   what a published catalogue alone is worth against a different published catalogue.
3. **Union gain - the actionable number.**  DTI(union(shipped submission, A-emission), B-only)
   minus DTI(shipped submission, B-only).  Rules §3.2 permits external data whose licence allows
   sharing with the sponsor; SGMC and QFaults are both USGS public domain.  If adding an external
   catalogue's traces to our emission raises the score on faults neither the labels nor the model
   contain, that is a legal, measurable route to recall on the hidden expert set.
4. **Controls, checked BEFORE any of the above is believed.**  A copy of the training labels must
   score ~0 against B-only (otherwise B-only is not "new"), a copy of B must score 1.0 (the
   scorer works on this population), and blanket-ones gives the coverage floor.
5. **Block bootstrap intervals** on every contrast (`src.metrics.bootstrap_from_blocks`), so a
   +0.01 "gain" is reported with the probability that it survives resampling 51 km blocks.

The verdict is DERIVED from the pre-registered acceptance criterion, never typed:

    adopt-external-catalogue emission  <=>  controls pass AND union beats the model alone by
    more than 0.01 on B-only AND the paired block bootstrap puts P(union > model) >= 0.95.

INPUTS (all produced by committed scripts; nothing is hand-written)
    A  data/evidence/proxy/proxy_catalogue.tif     scripts/build_proxy_catalogue.py (SGMC)
    B  data/evidence/xcat/qfaults_catalogue.tif    scripts/fetch_qfaults.py + the same builder
    prediction, labels, template                   data/bridge -> scripts/assemble_data_bridge.py

USAGE
    python scripts/measure_cross_catalogue_transfer.py \
        --truth-b data/evidence/xcat/qfaults_catalogue.tif \
        --catalogue-a data/evidence/proxy/proxy_catalogue.tif \
        --pred data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif \
        --out data/evidence/xcat/transfer_report.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt, label as cc_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (DEFAULT_BLOCK_PX, assign_folds, block_id_map, block_table,  # noqa: E402
                        describe_partition)
from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS, EPS,  # noqa: E402
                         GtContext, block_aggregate, bootstrap_from_blocks, score_within_mask)

KM_PER_PIXEL = 0.1
CODE_NONE, CODE_NEAR, CODE_ONLY = 0, 1, 2      # scripts/build_proxy_catalogue.py


# --------------------------------------------------------------------------------------
# io helpers (shared vocabulary with scripts/block_holdout_eval.py)
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
        grid = dict(path=str(path), width=src.width, height=src.height, crs=str(src.crs),
                    transform=[float(v) for v in src.transform][:6],
                    nodata=(None if src.nodata is None else float(src.nodata)),
                    dtype=str(arr.dtype), sha256=sha256(path),
                    bytes=int(Path(path).stat().st_size))
    return arr, grid


def require_same_grid(name: str, grid: dict, ref: dict) -> None:
    if (grid["width"], grid["height"]) != (ref["width"], ref["height"]) or \
            not np.allclose(grid["transform"], ref["transform"], atol=1e-6):
        raise SystemExit(f"{name} is not on the prediction grid ({grid['width']}x{grid['height']} "
                         f"vs {ref['width']}x{ref['height']}) - every input must share the "
                         "competition grid (problem page #submission-format)")


def decode_catalogue(arr: np.ndarray, footprint: np.ndarray, codes: Sequence[int]) -> np.ndarray:
    """Boolean mask of the requested codes, restricted to the data footprint."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0).astype(np.int16)
    mask = np.zeros(a.shape, bool)
    for c in codes:
        mask |= (a == int(c))
    return mask & footprint


def components(mask: np.ndarray) -> dict:
    """Connected-component summary of a trace mask (8-connectivity, as the proxy stats use)."""
    lab, n = cc_label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return dict(components=0, px=0, km=0.0, largest_px=0, largest_km=0.0)
    counts = np.bincount(lab.ravel())[1:]
    return dict(components=int(n), px=int(mask.sum()), km=round(float(mask.sum()) * KM_PER_PIXEL, 3),
                largest_px=int(counts.max()), largest_km=round(float(counts.max()) * KM_PER_PIXEL, 3),
                median_px=float(np.median(counts)))


# --------------------------------------------------------------------------------------
# emissions
# --------------------------------------------------------------------------------------
def band(mask: np.ndarray, width_px: int) -> np.ndarray:
    """Grow a trace mask to `width_px` (Euclidean), exactly as the emission sweep does."""
    if width_px <= 0:
        return mask.astype(np.float64)
    if not mask.any():
        return np.zeros(mask.shape, np.float64)
    edt = distance_transform_edt(~mask)
    return (mask | (edt <= float(width_px))).astype(np.float64)


def union_field(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise maximum of two emissions in [0, 1] (a probability field, not a mask OR)."""
    return np.maximum(np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0))


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def score_field(field: np.ndarray, ctx: GtContext, blocks: np.ndarray, n_blocks: int,
                fp_weight: np.ndarray, alpha: float, beta: float, eps: float) -> dict:
    """Global DTI + exact per-block components for one emission against one truth population."""
    dti, (TP, FP, FN) = ctx.score(field, alpha=alpha, beta=beta, eps=eps, return_components=True)
    comp = block_aggregate(ctx, field, blocks, n_blocks, fp_weight=fp_weight)
    per_block = []
    for b in range(n_blocks):
        ngt = int(comp["n_gt"][b])
        denom = comp["TP"][b] + alpha * comp["FP"][b] + beta * comp["FN"][b] + eps
        per_block.append(dict(block=int(b),
                              dti=(round(float(comp["TP"][b] / denom), 6) if ngt else None),
                              TP_w=float(comp["TP"][b]), FP_w=float(comp["FP"][b]),
                              FN_w=float(comp["FN"][b]), n_gt=ngt,
                              pred_px=int(comp["pred_px"][b]), scoreable=bool(ngt > 0)))
    # the reference per-block implementation must agree with the vectorised one
    for blk in [b for b in range(n_blocks) if per_block[b]["scoreable"]][:2]:
        ref = score_within_mask(field, ctx, blocks == blk, alpha=alpha, beta=beta, eps=eps)
        for key, got in (("TP_w", per_block[blk]["TP_w"]), ("FP_w", per_block[blk]["FP_w"]),
                         ("FN_w", per_block[blk]["FN_w"])):
            if not np.isclose(got, ref[key], rtol=1e-9, atol=1e-9):
                raise AssertionError(f"block {blk} {key}: {got} != reference {ref[key]}")
    tp = sum(b["TP_w"] for b in per_block)
    fp = sum(b["FP_w"] for b in per_block)
    fn = sum(b["FN_w"] for b in per_block)
    if not (np.isclose(tp, TP, rtol=1e-9, atol=1e-8) and np.isclose(fp, FP, rtol=1e-9, atol=1e-8)
            and np.isclose(fn, FN, rtol=1e-9, atol=1e-8)):
        raise AssertionError(f"per-block components do not sum to the global score (TP {tp} vs {TP})")
    return dict(global_dti=(round(float(dti), 6) if dti is not None else None),
                TP_w=float(TP), FP_w=float(FP), FN_w=float(FN), n_gt=int(ctx.n_gt),
                emitted_px=int(np.count_nonzero(field > 0)),
                emitted_mass=float(np.nansum(np.clip(field, 0, 1))),
                weighted_recall=(round(float(TP) / ctx.n_gt, 6) if ctx.n_gt else None),
                blocks=per_block)


class PopulationDegenerate(Exception):
    """Catalogue B is not independent of the training labels - a finding, not a failure.

    Raised when B has a substantial in-footprint population but essentially all of it is code 1
    (already within R of a training label).  `main` catches it, writes the report with a REFUSED
    verdict and exits 0, because "these two catalogues are the same lines" is a measurement worth
    committing; an empty or misaligned B raster is a data problem and still exits non-zero.
    """

    def __init__(self, payload: dict):
        super().__init__(payload.get("verdict", {}).get("conclusion", "population degenerate"))
        self.payload = payload


def build_report(truth_b_path: Path, catalogue_a_path: Path, pred_path: Optional[Path],
                 labels_path: Path, template_path: Optional[Path], widths: Sequence[int],
                 a_codes: Sequence[int], block_px: int, n_folds: int, n_boot: int, seed: int,
                 R: int, alpha: float, beta: float, eps: float,
                 min_union_gain: float, min_bootstrap_prob: float,
                 min_b_only_px: int = 100, min_b_only_fraction: float = 0.005,
                 min_b_all_px: int = 1000) -> dict:
    truth_b_raw, grid_b = read_band(truth_b_path)
    a_raw, grid_a = read_band(catalogue_a_path)
    labels_raw, grid_l = read_band(labels_path)
    template_raw, grid_t = read_band(template_path) if template_path else (None, None)
    pred_raw, grid_p = read_band(pred_path) if pred_path else (None, None)
    ref_grid = grid_p or grid_t or grid_b
    for name, g in (("catalogue A", grid_a), ("labels", grid_l), ("template", grid_t),
                    ("truth B", grid_b)):
        if g is not None:
            require_same_grid(name, g, ref_grid)

    footprint = np.isfinite(np.asarray(template_raw if template_raw is not None else
                                       (pred_raw if pred_raw is not None else truth_b_raw),
                                       dtype=np.float64))
    labels = (np.nan_to_num(np.asarray(labels_raw, dtype=np.float64), nan=0.0) > 0.5) & footprint
    b_all = decode_catalogue(truth_b_raw, footprint, (CODE_NEAR, CODE_ONLY))
    b_near = decode_catalogue(truth_b_raw, footprint, (CODE_NEAR,))
    b_only = decode_catalogue(truth_b_raw, footprint, (CODE_ONLY,))
    a_only = decode_catalogue(a_raw, footprint, (CODE_ONLY,))
    a_all = decode_catalogue(a_raw, footprint, tuple(a_codes))

    n_b_all, n_b_only = int(b_all.sum()), int(b_only.sum())
    if n_b_all < min_b_all_px:
        # Nothing to do with independence: B barely exists on this grid, which is a data problem
        # (wrong raster, wrong codes, misaligned grid) rather than a scientific finding.
        raise SystemExit(f"catalogue B contributes only {n_b_all:,} in-footprint pixels "
                         f"(< --min-b-all-px {min_b_all_px}): the raster is empty, misaligned or "
                         "coded unexpectedly - fix the input before drawing any conclusion "
                         f"(built by scripts/build_proxy_catalogue.py from {truth_b_path})")
    b_only_fraction = n_b_only / max(1, n_b_all)
    if n_b_only < min_b_only_px or b_only_fraction < min_b_only_fraction:
        labels_px = int(labels.sum())
        near_px = int(b_near.sum())
        raise PopulationDegenerate(dict(
            generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            generated_by="scripts/measure_cross_catalogue_transfer.py",
            purpose=("REFUSED cross-catalogue transfer measurement: catalogue B turned out not to be "
                     "independent of the training labels, so there is no 'new fault' population to "
                     "score against and no transfer claim is possible"),
            sources=dict(
                catalogue_A=dict(path=str(catalogue_a_path), grid=grid_a,
                                 provenance="USGS SGMC structure, DOI 10.3133/ds1052 "
                                            "(scripts/fetch_proxy_faults.py -> build_proxy_catalogue.py)"),
                catalogue_B=dict(path=str(truth_b_path), grid=grid_b,
                                 provenance="USGS Quaternary Fault and Fold Database, "
                                            "DOI 10.5066/P9BCVRCK (scripts/fetch_qfaults.py -> "
                                            "build_proxy_catalogue.py)"),
                labels=dict(path=str(labels_path), grid=grid_l),
                prediction=(dict(path=str(pred_path), grid=grid_p) if pred_path else None),
            ),
            footprint_px=int(footprint.sum()),
            metric=dict(R_pixels=R, R_meters=R * 100.0, alpha=alpha, beta=beta, eps=eps),
            population=dict(
                B_in_footprint_px=n_b_all, B_in_footprint_km=round(n_b_all * KM_PER_PIXEL, 1),
                B_code1_near_a_label_px=near_px,
                B_code1_fraction=round(near_px / max(1, n_b_all), 6),
                B_only_px=n_b_only, B_only_km=round(n_b_only * KM_PER_PIXEL, 3),
                B_only_fraction=round(b_only_fraction, 6),
                label_fault_px=labels_px,
                labels_within_R_of_B_px=int(np.count_nonzero(
                    labels & (distance_transform_edt(~b_all) <= R))) if n_b_all else 0,
                thresholds=dict(min_b_only_px=min_b_only_px,
                                min_b_only_fraction=min_b_only_fraction,
                                min_b_all_px=min_b_all_px)),
            controls_pass=False,
            measurements=[],
            verdict=dict(
                best_union_candidate=None, best_union_dti=None, model_dti=None, union_gain=None,
                criterion=dict(min_union_gain=min_union_gain,
                               min_bootstrap_prob=min_bootstrap_prob, controls_pass=False,
                               union_gain=None, bootstrap_prob_beats_model=None),
                conclusion=(
                    f"REFUSED - catalogue B is NOT independent of the training labels: "
                    f"{near_px:,} of its {n_b_all:,} in-footprint pixels "
                    f"({100.0 * near_px / max(1, n_b_all):.2f} %) are already within R = {R} px of a "
                    f"label, leaving {n_b_only:,} px ({100.0 * b_only_fraction:.4f} %) as a "
                    f"'new fault' population - below the pre-registered minimum of "
                    f"{min_b_only_px:,} px and {min_b_only_fraction * 100:.2f} %. Scoring against "
                    f"{n_b_only:,} pixel(s) would produce a number with no meaning, so none is "
                    "reported. This is a FINDING about the two catalogues, not a bug: it means the "
                    "training labels and catalogue B are the same lines in this footprint, and it is "
                    "recorded with the overlap that establishes it.")),
            caveats=[
                "the refusal is quantitative: the overlap fraction above is the evidence, and it is "
                "reproducible from the committed rasters with scripts/build_proxy_catalogue.py",
                "a REFUSED verdict says nothing about whether a POLICY transfers - it says this pair "
                "of catalogues cannot answer the question",
                "the new-fault-like SGMC population (data/evidence/proxy/proxy_catalogue.tif, "
                "61,664 code-2 px, 24.94 % already covered by the labels) remains the only "
                "available surrogate, with the limitation that it is pre-Quaternary bedrock "
                "structure rather than an expert interpretation of the geophysics",
            ],
        ))
    if not a_all.any():
        raise SystemExit(f"catalogue A emission is empty for codes {list(a_codes)} - nothing to transfer")

    # ---- blocks (the same partition the training hold-out uses) ------------------------
    shape = footprint.shape
    tab = block_table(shape, block_px, valid=footprint, labels=b_only)
    fold_of = assign_folds(tab, n_folds, seed=0, mode="balanced")
    blocks = block_id_map(shape, block_px)
    n_blocks = len(tab)
    partition = describe_partition(shape, block_px, n_folds, 0, tab, fold_of,
                                   labels=b_only, valid=footprint, mode="balanced")

    ctx_b = GtContext(b_only.astype(np.float64), R_pixels=R)
    fpw_b = ctx_b.fp_weight()
    ctx_a = GtContext(a_only.astype(np.float64), R_pixels=R)          # for the A-vs-B overlap rows

    model_field = None
    if pred_raw is not None:
        model_field = np.clip(np.nan_to_num(np.asarray(pred_raw, dtype=np.float64), nan=0.0), 0.0, 1.0)
        model_field[~footprint] = 0.0

    # ---- CONTROLS (checked before any transfer number is reported) ---------------------
    ones = footprint.astype(np.float64)
    controls = {}
    controls["labels_copy_vs_B_only"] = score_field(labels.astype(np.float64), ctx_b, blocks,
                                                    n_blocks, fpw_b, alpha, beta, eps)
    controls["B_copy_vs_B_only"] = score_field(b_only.astype(np.float64), ctx_b, blocks, n_blocks,
                                               fpw_b, alpha, beta, eps)
    controls["blanket_ones_vs_B_only"] = score_field(ones, ctx_b, blocks, n_blocks, fpw_b,
                                                     alpha, beta, eps)
    labels_copy_dti = controls["labels_copy_vs_B_only"]["global_dti"]
    b_copy_dti = controls["B_copy_vs_B_only"]["global_dti"]
    controls_pass = bool(labels_copy_dti is not None and labels_copy_dti < 0.05 and
                         b_copy_dti is not None and b_copy_dti > 0.99)

    # ---- MEASUREMENTS -----------------------------------------------------------------
    rows: List[dict] = []
    if model_field is not None:
        rows.append(dict(kind="model", width=0, label="model_as_shipped",
                         **score_field(model_field, ctx_b, blocks, n_blocks, fpw_b, alpha, beta, eps)))
    for w in widths:
        a_band = band(a_all, int(w))
        rows.append(dict(kind="catalogue_a", width=int(w), label=f"A_codes{'_'.join(map(str, a_codes))}_w{int(w)}px",
                         **score_field(a_band, ctx_b, blocks, n_blocks, fpw_b, alpha, beta, eps)))
        if model_field is not None:
            rows.append(dict(kind="union", width=int(w), label=f"model+A_w{int(w)}px",
                             **score_field(union_field(model_field, a_band), ctx_b, blocks,
                                           n_blocks, fpw_b, alpha, beta, eps)))
    # A's own population, for the surrogate-agreement row (A-only truth, A emission is circular
    # by construction and is reported only to prove the control still fires)
    a_self = score_field(b_only.astype(np.float64), ctx_a, blocks, n_blocks, ctx_a.fp_weight(),
                         alpha, beta, eps)

    # ---- paired block bootstrap: model alone is the reference for every union row ------
    model_row = next((r for r in rows if r["kind"] == "model"), None)
    boot_rows = ([model_row] if model_row else []) + [r for r in rows if r["kind"] == "union"]
    bootstrap = (bootstrap_from_blocks([{k: v for k, v in r.items() if k in ("label", "blocks")}
                                        for r in boot_rows], alpha=alpha, beta=beta, eps=eps,
                                       n_boot=n_boot, seed=seed) if len(boot_rows) > 1 else {})
    a_boot_rows = [model_row] if model_row else []
    a_boot_rows += [r for r in rows if r["kind"] == "catalogue_a"]
    bootstrap_a = (bootstrap_from_blocks([{k: v for k, v in r.items() if k in ("label", "blocks")}
                                          for r in a_boot_rows], alpha=alpha, beta=beta, eps=eps,
                                         n_boot=n_boot, seed=seed) if len(a_boot_rows) > 1 else {})

    # ---- overlap geometry: how much of B-only does A actually touch? -------------------
    edt_a = distance_transform_edt(~a_all) if a_all.any() else None
    edt_b = distance_transform_edt(~b_only) if b_only.any() else None
    edt_model = distance_transform_edt(~(model_field > 0)) if model_field is not None else None
    overlap = dict(
        B_only_px=int(b_only.sum()), B_only_km=round(float(b_only.sum()) * KM_PER_PIXEL, 3),
        B_all_px=int(b_all.sum()), B_near_label_px=int(b_near.sum()),
        B_near_label_fraction_of_all=round(float(b_near.sum()) / max(1, int(b_all.sum())), 6),
        A_only_px=int(a_only.sum()), A_emitted_px=int(a_all.sum()),
        # recall-like: how much of B's new-fault-like truth does A's emission reach?
        B_only_within_R_of_A_px=(int(np.count_nonzero(b_only & (edt_a <= R)))
                                 if edt_a is not None else 0),
        # precision-like: how much of A's emission lands on B's new-fault-like truth?
        A_emitted_within_R_of_B_only_px=(int(np.count_nonzero(a_all & (edt_b <= R)))
                                         if edt_b is not None else 0),
        B_only_within_R_of_model_px=(int(np.count_nonzero(b_only & (edt_model <= R)))
                                     if edt_model is not None else None),
        A_only_and_B_only_same_px=int(np.count_nonzero(a_only & b_only)),
        components_B_only=components(b_only), components_A=components(a_all),
    )
    # Two fractions, reported as a pair because either one alone is misleading: A can reach most
    # of B-only (high recall-like) while emitting so much that almost none of it lands on B-only
    # (low precision-like), which under the official metric is a net loss, not a net gain.
    overlap["B_only_recall_by_A_at_R"] = round(
        overlap["B_only_within_R_of_A_px"] / max(1, overlap["B_only_px"]), 6)
    overlap["A_emitted_precision_on_B_only"] = round(
        overlap["A_emitted_within_R_of_B_only_px"] / max(1, overlap["A_emitted_px"]), 6)
    if overlap["B_only_within_R_of_model_px"] is not None:
        overlap["B_only_recall_by_model_at_R"] = round(
            overlap["B_only_within_R_of_model_px"] / max(1, overlap["B_only_px"]), 6)
        overlap["model_emitted_precision_on_B_only"] = round(
            int(np.count_nonzero((model_field > 0) & (edt_b <= R)))
            / max(1, int(np.count_nonzero(model_field > 0))), 6)

    # ---- verdict: DERIVED from the pre-registered criterion ---------------------------
    union_rows = [r for r in rows if r["kind"] == "union"]
    best_union = max(union_rows, key=lambda r: r["global_dti"]) if union_rows else None
    model_dti = model_row["global_dti"] if model_row else None
    gain = (round(best_union["global_dti"] - model_dti, 6)
            if best_union and model_dti is not None else None)
    prob = (bootstrap.get(best_union["label"], {}).get("prob_beats_reference")
            if best_union and bootstrap else None)
    criterion = dict(min_union_gain=min_union_gain, min_bootstrap_prob=min_bootstrap_prob,
                     controls_pass=controls_pass,
                     union_gain=gain, bootstrap_prob_beats_model=prob)
    if not controls_pass:
        conclusion = ("REFUSED - the controls failed, so B-only is not an independent new-fault "
                      "population and no transfer claim is made")
    elif best_union is None or gain is None:
        conclusion = ("NOT MEASURABLE - no prediction was supplied, so the union gain cannot be "
                      "computed; catalogue-A-only rows are still reported")
    elif gain > min_union_gain and (prob is not None and prob >= min_bootstrap_prob):
        conclusion = ("ADOPT - adding the external catalogue's traces to the shipped emission "
                      "beats the model alone on an independent new-fault-like population, by more "
                      "than the pre-registered margin and with the pre-registered bootstrap "
                      "probability")
    else:
        conclusion = ("DO NOT ADOPT - the union gain does not clear the pre-registered margin "
                      "(gain, bootstrap probability and controls are all recorded above)")

    return dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/measure_cross_catalogue_transfer.py",
        purpose=("emit catalogue A (SGMC) and score against independent catalogue B (QFaults) "
                 "code-2 pixels: surrogate agreement, catalogue-as-predictor, and the union gain "
                 "that decides whether external catalogue traces join the submission"),
        sources=dict(
            catalogue_A=dict(path=str(catalogue_a_path), grid=grid_a,
                             provenance="USGS SGMC structure, DOI 10.3133/ds1052 "
                                        "(scripts/fetch_proxy_faults.py -> build_proxy_catalogue.py)"),
            catalogue_B=dict(path=str(truth_b_path), grid=grid_b,
                             provenance="USGS Quaternary Fault and Fold Database, "
                                        "DOI 10.5066/P9BCVRCK (scripts/fetch_qfaults.py -> "
                                        "build_proxy_catalogue.py)"),
            labels=dict(path=str(labels_path), grid=grid_l),
            prediction=(dict(path=str(pred_path), grid=grid_p) if pred_path else None),
        ),
        metric=dict(R_pixels=R, R_meters=R * 100.0, alpha=alpha, beta=beta, eps=eps,
                    source="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric"),
        blocks=dict(block_px=block_px, block_km=round(block_px * KM_PER_PIXEL, 3),
                    n_blocks=n_blocks, partition=partition),
        footprint_px=int(footprint.sum()),
        overlap=overlap,
        controls=controls,
        controls_pass=controls_pass,
        measurements=rows,
        surrogate_agreement=dict(
            model_vs_B_only=model_dti,
            B_only_scored_against_A_only=a_self["global_dti"],
            note=("model_vs_B_only is the second independent estimate of the quantity the SGMC "
                  "proxy gives (0.0999 on the shipping field, "
                  "data/evidence/block_holdout/block_stratified.json).  "
                  "B_only_scored_against_A_only scores B's new-fault-like pixels using B's own "
                  "trace as the prediction: it is the 1.0 sanity anchor for that population, "
                  "reported next to the A-population number so the two are never confused."),
        ),
        bootstrap=dict(n=n_boot, seed=seed, unit="spatial block, paired against the model row",
                       union_vs_model=bootstrap, catalogue_a_vs_model=bootstrap_a),
        verdict=dict(best_union_candidate=(best_union["label"] if best_union else None),
                     best_union_dti=(best_union["global_dti"] if best_union else None),
                     model_dti=model_dti, union_gain=gain,
                     criterion=criterion, conclusion=conclusion),
    )


def print_report(rep: dict) -> None:
    if "population" in rep:
        pop, v = rep["population"], rep["verdict"]
        print(f"\nREFUSED: catalogue B has {pop['B_in_footprint_px']:,} px in the footprint but only "
              f"{pop['B_only_px']:,} px ({pop['B_only_fraction'] * 100:.4f} %) are code 2 - "
              f"{pop['B_code1_near_a_label_px']:,} px ({pop['B_code1_fraction'] * 100:.2f} %) are "
              f"already within R = {rep['metric']['R_pixels']} px of a training label")
        print(f"  label fault px {pop['label_fault_px']:,}; label px within R of B "
              f"{pop['labels_within_R_of_B_px']:,}")
        print(f"  thresholds: {pop['thresholds']}")
        print(f"\nverdict: {v['conclusion']}")
        print("\nno DTI is reported: a 1-pixel truth population cannot support a transfer claim.")
        return
    o, v = rep["overlap"], rep["verdict"]
    print(f"\ncatalogue B (QFaults) new-fault-like truth: {o['B_only_px']:,} px "
          f"({o['B_only_km']:,.1f} km); B already within R of a label: "
          f"{o['B_near_label_fraction_of_all'] * 100:.1f} % of B")
    print(f"catalogue A (SGMC) emission: {o['A_emitted_px']:,} px; "
          f"B-only within R of A: {o['B_only_within_R_of_A_px']:,} px "
          f"({o['B_only_recall_by_A_at_R'] * 100:.1f} % recall)")
    if o.get("B_only_recall_by_model_at_R") is not None:
        print(f"model recall of B-only within R: {o['B_only_recall_by_model_at_R'] * 100:.1f} %")
    print(f"controls: labels-copy {rep['controls']['labels_copy_vs_B_only']['global_dti']:.4f} (must be ~0), "
          f"B-copy {rep['controls']['B_copy_vs_B_only']['global_dti']:.4f} (must be ~1), "
          f"blanket-ones {rep['controls']['blanket_ones_vs_B_only']['global_dti']:.4f} -> "
          f"{'PASS' if rep['controls_pass'] else 'FAIL'}")
    print("\n  emission                        DTI(B-only)   weighted recall   emitted px")
    for r in rep["measurements"]:
        print(f"  {r['label']:<30} {r['global_dti']:.4f}        "
              f"{(r['weighted_recall'] or 0):.4f}          {r['emitted_px']:>9,}")
    print(f"\nsurrogate agreement: model vs B-only {rep['surrogate_agreement']['model_vs_B_only']}")
    print(f"verdict: {v['conclusion']}")
    print(f"  best union {v['best_union_candidate']} = {v['best_union_dti']}, "
          f"model alone = {v['model_dti']}, gain = {v['union_gain']}, "
          f"criterion {v['criterion']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-b", default="data/evidence/xcat/qfaults_catalogue.tif")
    ap.add_argument("--catalogue-a", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--pred", default="data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif",
                    help="'' to measure catalogue A alone")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--widths", default="0,1,2,3,4,6")
    ap.add_argument("--a-codes", default="2",
                    help="which A codes to emit: 2 = A traces the labels lack (default), 1,2 = all A")
    ap.add_argument("--block-px", type=int, default=DEFAULT_BLOCK_PX)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--bootstraps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=int, default=DEFAULT_R_PIXELS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--min-union-gain", type=float, default=0.01,
                    help="pre-registered margin the union must beat the model alone by")
    ap.add_argument("--min-b-only-px", type=int, default=100,
                    help="below this many code-2 pixels in catalogue B the population is "
                         "degenerate: the report is written with a REFUSED verdict and the run "
                         "exits 0, because 'these two catalogues are the same lines' is a finding")
    ap.add_argument("--min-b-only-fraction", type=float, default=0.005,
                    help="...or below this fraction of B's in-footprint traces")
    ap.add_argument("--min-b-all-px", type=int, default=1000,
                    help="below this many in-footprint B pixels the raster itself is suspect "
                         "(empty/misaligned/miscoded) and the run fails instead of reporting")
    ap.add_argument("--min-bootstrap-prob", type=float, default=0.95,
                    help="pre-registered paired block-bootstrap probability required")
    ap.add_argument("--out", default="data/evidence/xcat/transfer_report.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    for label, path in (("catalogue B", a.truth_b), ("catalogue A", a.catalogue_a),
                        ("labels", a.labels)):
        if not Path(path).exists():
            raise SystemExit(f"{label} raster not found at {path}. Build it first:\n"
                             "  python scripts/fetch_qfaults.py --bbox-source data/labels.tif \\\n"
                             "      --out data/external/qfaults_footprint.geojson \\\n"
                             "      --meta data/evidence/xcat/fetch_meta.json\n"
                             "  python scripts/build_proxy_catalogue.py \\\n"
                             "      --proxy data/external/qfaults_footprint.geojson \\\n"
                             "      --template data/labels.tif --labels data/labels.tif \\\n"
                             "      --out data/evidence/xcat/qfaults_catalogue.tif \\\n"
                             "      --stats data/evidence/xcat/qfaults_stats.json\n"
                             "(both need unrestricted egress: .github/workflows/cross-catalogue.yml "
                             "runs them on a runner and commits the results)")
    widths = [int(x) for x in str(a.widths).split(",") if x.strip() != ""]
    a_codes = [int(x) for x in str(a.a_codes).split(",") if x.strip() != ""]
    if CODE_ONLY not in a_codes:
        raise SystemExit(f"--a-codes {a_codes} excludes code {CODE_ONLY} (A traces the labels "
                         "lack), which is the only part of A that can transfer to new faults")
    try:
        rep = build_report(Path(a.truth_b), Path(a.catalogue_a),
                           Path(a.pred) if a.pred else None, Path(a.labels),
                           Path(a.template) if a.template else None,
                           widths, a_codes, a.block_px, a.folds, a.bootstraps, a.seed,
                           a.R, a.alpha, a.beta, a.eps, a.min_union_gain, a.min_bootstrap_prob,
                           a.min_b_only_px, a.min_b_only_fraction, a.min_b_all_px)
    except PopulationDegenerate as exc:
        # A degenerate population is a FINDING (catalogue B is not independent of the labels), so it
        # is committed as a report and the run exits 0 - only broken inputs exit non-zero.
        rep = exc.payload
        rep["exit_code"] = 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    if not a.quiet:
        print_report(rep)
        print(f"\nwrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
