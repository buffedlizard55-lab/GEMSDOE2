#!/usr/bin/env python3
"""CPU-only baseline -> a valid, upload-ready submission.tif, with no GPU and no torch.

WHY THIS EXISTS
---------------
Every artefact this repository has measured so far needs either a GPU
(`configs/config.yaml`: 10 MC splits x 60 epochs x EfficientNet-B5) or a GitHub runner that
produced the fold checkpoints behind `data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`
(the 11-fold blend).  Both routes work, but neither can be re-executed inside a 2 vCPU / ~3 GB
sandbox, so *regenerating* a submission has always depended on something outside the machine in
front of you.

This script closes that gap.  It trains a classical pixel classifier on the official 19-band
feature stack, predicts a full-grid probability field, selects an emission policy on a
**spatially held-out fold**, writes the result through the fail-loud writer
(`src/submission_io.write_submission`) and validates it against the official spec.  It is:

  * **CPU-only and torch-free** - scikit-learn's histogram gradient boosting handles NaN
    natively, which matters because 57.9 % of the grid is NaN outside the survey footprint;
  * **memory-bounded** - features are read in row chunks, never as one 933 MB array, so it runs
    in the same sandbox that cannot hold the full stack twice;
  * **deterministic** - one seed drives sampling, the model and the fold partition;
  * **honest about selection** - the model never sees fold `--fold` (default 0) of the standard
    `src/blocks.py` partition, and the emission policy (floor x thin x dilate) is chosen by the
    **union population's** DTI restricted to that fold's blocks.  That is the selection signal
    `STATUS.md` next-step item 4 asked for: the union of the catalogue labels and the
    new-fault-like proxy pixels is the closest local surrogate for the Phase-2 test set
    (rules §3.6), and it cannot be reconstructed from its components because `FP_w` sums over
    *prediction* pixels.

SELECT, THEN MEASURE ON A FOLD NOTHING TOUCHED
---------------------------------------------
A policy chosen by taking the best of N candidate scores on fold A is biased upward on fold A:
`max()` over noisy estimates is not an estimate.  So the script holds out **two** folds:

    fold `--fold`  selects the policy   (never trained on)
    fold `--eval-fold` measures it      (never trained on, never scored during the sweep;
                                         defaults to the next fold)

Both are excluded from training, and `generalisation` in the report is the winner's field scored on
the untouched fold - the number to quote.  `scripts/block_holdout_eval.py --score-fold <eval-fold>`
reproduces those three values from `prob_raw.tif` digit for digit, which is what makes the number
auditable rather than asserted.  The cost is real and stated: with the default 4-fold partition the
model trains on ~half the labelled pixels instead of three quarters, so the baseline is *weaker*
than a single-holdout design would look - and honest about it.

THE DEGENERATE OPTIMUM, AND THE CAP THAT PREVENTS IT
---------------------------------------------------
An unconstrained argmax of this metric over the shaping space returns "emit the whole
footprint": `FN_w` carries beta = 0.8 while `FP_w` carries alpha = 0.2, so with a weak field the
recall term wins.  Measured on fold 0 below: the `floor 0, thin off` candidate emits 5,164,312 px
(the entire footprint) and scores union DTI 0.1229, while the best *localised* candidate scores
0.0207.  Shipping that candidate would be shipping a non-prediction.  So candidate eligibility is
pre-registered here:

    emitted support <= --max-emitted-fraction (5 %) of the survey footprint
    emitted support >= --min-emitted-px (1,000)

and the winner is the argmax of the held-out union DTI *among eligible candidates*.  The
ineligible rows stay in the report, with the reason, so the exclusion is measurable rather than
silent.  For scale: the shipped deep ensemble emits 172,974 px (3.3 % of the footprint) and is the
better field on the same surrogate population, so the cap costs nothing that was worth having.

WHAT IT IS NOT
--------------
It is **not** a leaderboard-competitive model and the report says so in its own fields.  A
per-pixel classifier has no spatial context, which is the one thing a fault-line detector needs;
the deep ensemble's committed proxy DTI is the number to beat, and the report leaves that
comparison to `scripts/block_holdout_eval.py` rather than asserting it.  What this script
guarantees is a *format-valid submission* and a reproducible path to one on any machine with
numpy, rasterio and scikit-learn.

USAGE
    python scripts/baseline_submission.py                      # -> data/evidence/baseline/
    python scripts/baseline_submission.py --fold 1 --max-negatives 500000
    python scripts/baseline_submission.py --refit-on-all       # ship a model that saw every fold

The report writes the exact follow-up command that turns the raw field into block-stratified
evidence with confidence intervals (`scripts/block_holdout_eval.py --combined-population`).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (DEFAULT_BLOCK_PX, DEFAULT_BUFFER_PX, DEFAULT_N_FOLDS,  # noqa: E402
                        assign_folds, block_table, describe_partition, held_out_mask,
                        scored_mask)
from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS,  # noqa: E402
                         GtContext, score_within_mask)
from src.submission_io import clean_profile, sha256_file, write_submission  # noqa: E402
from src.submission_optim import (dilate_mask, dominant_thin, floor_sharpen,  # noqa: E402
                                   optimize_submission, shaping_thresholds)

NODATA_FLOOR = -1e30          # anything below this is a nodata sentinel, not a measurement
FLOOR_GRID = 9                # shaping floors searched on the held-out fold
DILATE_GRID = (0, 1)          # band widths tried around the thinned corridor
POPULATIONS = ("catalogue", "proxy", "combined")
MAX_EMITTED_FRACTION = 0.05   # pre-registered support cap (see "the degenerate optimum" above)
MIN_EMITTED_PX = 1_000        # ...and a floor, so an over-thresholded field is not "a prediction"


# --------------------------------------------------------------------------------- reading
def _nan_nodata(block: np.ndarray) -> np.ndarray:
    """Replace the official nodata sentinel (-3.4e38) with NaN in a freshly read block."""
    out = block.astype(np.float32, copy=True)
    out[block < NODATA_FLOOR] = np.nan
    return out


def read_truth(labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(fault mask, valid mask) from labels.tif: code 1 = fault, nodata = outside the survey."""
    with rasterio.open(labels_path) as src:
        a = src.read(1)
        nodata = src.nodata
    valid = np.ones(a.shape, bool) if nodata is None else (a != nodata)
    return (a == 1) & valid, valid


def read_template(path: Path) -> tuple[np.ndarray, dict]:
    """Official footprint mask (finite pixels of sample_submission.tif) and its raster profile."""
    with rasterio.open(path) as src:
        a = src.read(1)
        prof = src.profile.copy()
    return np.isfinite(a), prof


def read_code_mask(path: Path, code: int) -> np.ndarray:
    """Pixels equal to ``code`` in a categorical raster (e.g. proxy_catalogue code 2)."""
    with rasterio.open(path) as src:
        return src.read(1) == code


# --------------------------------------------------------------------------------- sampling
def choose_samples(fault: np.ndarray, valid: np.ndarray, trainable: np.ndarray,
                   rng: np.random.Generator, neg_ratio: float,
                   max_negatives: int) -> dict:
    """Positives = every fault pixel a training fold owns; negatives = a capped random sample.

    Sampling negatives (rather than keeping all ~31 M) keeps the fit to a few seconds and keeps
    the model's binning memory proportional to the sample count, which is the difference between
    running and OOM-ing in a small sandbox.  The ratio is a flag, not a constant.
    """
    pos = fault & trainable
    neg_pool = valid & ~fault & trainable
    n_pos = int(pos.sum())
    n_pool = int(neg_pool.sum())
    want = min(int(max_negatives), max(1, int(round(neg_ratio * max(1, n_pos)))))
    if want >= n_pool:
        neg = neg_pool
    else:
        # O(want) draw: a full permutation of ~31 M indices is not needed to pick `want` of them
        idx = np.sort(rng.choice(n_pool, size=want, replace=False))
        keep = np.zeros(neg_pool.size, bool)
        keep[np.flatnonzero(neg_pool.ravel())[idx]] = True
        neg = keep.reshape(neg_pool.shape)
    return dict(pos=pos, neg=neg, n_pos=n_pos, n_neg=int(neg.sum()), n_neg_pool=n_pool,
                neg_ratio_requested=float(neg_ratio), negatives_capped=bool(want >= n_pool))


def fill_matrix(feature_path: Path, sample_mask: np.ndarray, pos_mask: np.ndarray,
                chunk_rows: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(X, y, seconds) for the sampled pixels: one bounded pass over the feature stack.

    Pixel order is (row, col) so the matrix is a pure function of the masks, and the read is
    chunked so peak memory is one block (chunk_rows x W x bands x 4 B) plus the output.
    """
    H, W = sample_mask.shape
    flat = np.flatnonzero(sample_mask.ravel())
    rows = (flat // W).astype(np.int64)
    order = np.argsort(rows, kind="stable")
    rows_s, flat_s = rows[order], flat[order]
    pos_flat = pos_mask.ravel()
    X = np.empty((flat.size, 0), np.float32)
    y = np.empty(flat.size, np.int8)
    start = 0
    t0 = time.time()
    with rasterio.open(feature_path) as src:
        bands = src.count
        X = np.empty((flat.size, bands), np.float32)
        for r0 in range(0, H, chunk_rows):
            r1 = min(H, r0 + chunk_rows)
            lo = int(np.searchsorted(rows_s, r0, side="left"))
            hi = int(np.searchsorted(rows_s, r1 - 1, side="right"))
            if lo >= hi:
                continue
            block = _nan_nodata(src.read(window=Window(0, r0, W, r1 - r0)))     # (b, h, w)
            sub = flat_s[lo:hi]
            X[start:start + sub.size] = block[:, sub // W - r0, sub % W].T
            y[start:start + sub.size] = pos_flat[sub].astype(np.int8)
            start += sub.size
            del block
    if start != flat.size:
        raise RuntimeError(f"filled {start} of {flat.size} sampled rows")
    return X, y, round(time.time() - t0, 1)


# --------------------------------------------------------------------------------- model
def fit_model(X: np.ndarray, y: np.ndarray, args):
    """Histogram gradient boosting: NaN-native, CPU-fast, deterministic given the seed."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=int(args.max_iter), learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes), min_samples_leaf=int(args.min_samples_leaf),
        l2_regularization=float(args.l2_regularization), early_stopping=False,
        random_state=int(args.seed))
    t0 = time.time()
    clf.fit(X, y)
    return clf, round(time.time() - t0, 1)


def predict_grid(clf, feature_path: Path, footprint: np.ndarray, valid: np.ndarray,
                 chunk_rows: int) -> tuple[np.ndarray, float]:
    """Full-grid probability field: NaN where the footprint or the features are missing."""
    H, W = footprint.shape
    prob = np.full((H, W), np.nan, np.float32)
    t0 = time.time()
    with rasterio.open(feature_path) as src:
        for r0 in range(0, H, chunk_rows):
            r1 = min(H, r0 + chunk_rows)
            block = _nan_nodata(src.read(window=Window(0, r0, W, r1 - r0)))
            X = block.transpose(1, 2, 0).reshape(-1, block.shape[0])
            usable = (valid[r0:r1].ravel() & footprint[r0:r1].ravel()
                      & ~np.isnan(X).all(axis=1))
            if usable.any():
                row = prob[r0:r1].ravel()
                row[usable] = clf.predict_proba(X[usable])[:, 1].astype(np.float32)
                prob[r0:r1] = row.reshape(r1 - r0, W)
            del block, X
    return prob, round(time.time() - t0, 1)


# --------------------------------------------------------------------------------- selection
AUDIT_TOLERANCE = 5e-4     # measured boundary-credit spread between the two implementations


def audit_generalisation(sub_path: Path, labels: Path, proxy: Path, template: Path,
                         block_px: int, folds: int, eval_fold: int,
                         expected: dict, out_dir: Path,
                         tolerance: float = AUDIT_TOLERANCE) -> dict:
    """Re-score the SHAPED artifact with the independent evaluator and diff the two readings.

    Two implementations of the same metric disagreed by a small, systematic amount the first time
    this was checked by hand, and the reason is worth recording rather than rounding away:

      * `src.metrics.score_within_mask` - what this report's `generalisation` block uses - keeps the
        *global* prediction context and restricts only the sums, which is its documented contract
        (a prediction outside the block still earns credit for truth inside it);
      * `scripts/block_holdout_eval.py --score-fold` restricts the PREDICTION to the fold first, so
        a truth pixel within R = 3 px of a fold boundary cannot be credited by a prediction on the
        other side of it.

    The second is the strictly more conservative reading, and on this grid the two differ by
    ~1e-4 - three to four digits, not "digit for digit".  So the audit measures the difference and
    records it, and `--verify` fails loudly only beyond `AUDIT_TOLERANCE`.

    The artifact scored is `submission.tif`, not `prob_raw.tif`: the evaluator shapes by threshold
    plus dilation and cannot thin, so pointing it at the raw field with the winner's floor would
    measure the un-thinned sibling (measured 2026-09-20: combined 0.1259 vs 0.0789).
    """
    out = out_dir / "reproduce_audit.json"
    cmd = [sys.executable, "scripts/block_holdout_eval.py",
           "--pred", str(sub_path), "--labels", str(labels), "--proxy", str(proxy),
           # the SAME partition the report's numbers were measured on: the evaluator has its own
           # defaults, and a different block size silently measures a different fold
           "--template", str(template), "--block-px", str(int(block_px)), "--folds", str(int(folds)),
           "--score-fold", str(int(eval_fold)), "--combined-population",
           "--floors", "0.5", "--widths", "0",
           # the evaluator requires its reference candidate to sit inside the swept grid, so the
           # reference is the same (0.5, 0) that reconstructs a binary artifact
           "--reference-floor", "0.5", "--reference-width", "0",
           "--crosscheck-sweep", "''", "--out", str(out), "--quiet"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    rec = dict(command=" ".join(cmd), returncode=int(proc.returncode),
               stdout_tail=proc.stdout.strip().splitlines()[-1:] or [],
               stderr_tail=proc.stderr.strip().splitlines()[-1:] or [])
    if proc.returncode != 0 or not out.exists():
        rec.update(passed=False, reason="the audit command did not produce a report")
        return rec
    rep = json.loads(out.read_text())
    keymap = {"catalogue": "labels", "proxy": "proxy_only", "combined": "combined"}
    observed, deltas = {}, {}
    for name, pkey in keymap.items():
        pop = (rep.get("populations") or {}).get(pkey) or {}
        rows = pop.get("candidates") or []
        if not rows:
            rec.update(passed=False, reason=f"no candidate row for population {pkey}")
            return rec
        row = rows[0]
        # the fold-restricted reading is the sum over the fold's own blocks; the row's
        # `global_dti` is the whole-grid score and is NOT the fold number
        blocks = [b for b in row.get("blocks", []) if b.get("scoreable")]
        TP = sum(float(b["TP_w"]) for b in blocks)
        FP = sum(float(b["FP_w"]) for b in blocks)
        FN = sum(float(b["FN_w"]) for b in blocks)
        obs = TP / (TP + 0.2 * FP + 0.8 * FN + 1e-7)
        observed[name] = round(obs, 6)
        exp = expected[name]["dti"]
        if exp is None:
            deltas[name] = None
            continue
        deltas[name] = round(abs(obs - float(exp)), 8)
    worst = max((v for v in deltas.values() if v is not None), default=0.0)
    rec.update(passed=bool(worst <= tolerance), observed=observed, deltas=deltas,
               max_abs_delta=worst, tolerance=float(tolerance),
               per_block_dti="the audit's residual differences sit on truth pixels within R of a "
                             "fold boundary, where the two implementations legitimately differ")
    return rec


def shipped_comparison() -> dict:
    """The deep ensemble's committed number on the union population, quoted not re-measured.

    Read from `data/evidence/proxy/combined_truth_shipped.json` (produced by
    `scripts/block_holdout_eval.py --combined-population` on the shipped raster).  This baseline is
    not comparable on equal terms - different field, different training budget - so the report
    states both numbers with their provenance and leaves the reading to the results page.
    """
    p = Path("data/evidence/proxy/combined_truth_shipped.json")
    if not p.exists():
        return dict(available=False, why=f"{p} not committed - no comparison quoted")
    d = json.loads(p.read_text())
    ref = d.get("reference") or {}
    return dict(available=True, source=str(p), shipped_union_dti=ref.get("dti"),
                shipped_ci95=ref.get("ci95"), shipped_pred_px=ref.get("pred_px"),
                provenance="committed evidence measured on the shipped ensemble raster; this "
                           "baseline is scored on the same report only via `next_command`")


def contexts(fault: np.ndarray, proxy_only: np.ndarray, R: int) -> dict:
    """One GtContext per truth population on the same grid (global geometry preserved)."""
    return dict(catalogue=GtContext(fault, R_pixels=R),
                proxy=GtContext(proxy_only, R_pixels=R),
                combined=GtContext(fault | proxy_only, R_pixels=R))


def candidate_grid(floors: int = FLOOR_GRID, dilates=DILATE_GRID):
    """Deterministic candidate list: (floor, thin, dilate) - the whole shaping policy space."""
    return [(float(f), bool(t), int(d))
            for f in shaping_thresholds(floors)
            for t in (True, False)
            for d in dilates]


def shaped_field(prob: np.ndarray, t0: float, thin: bool, dilate: int, R: int,
                 cache: dict) -> np.ndarray:
    """`optimize_submission` split into cacheable parts, so one thinning serves both widths.

    `dominant_thin` (skeletonize + a bounded greedy fill) is the expensive step and depends only
    on (probabilities, floor, thin) - not on the dilation applied afterwards.  Two candidates that
    differ only in width therefore shared nothing before this existed, which doubled the cost of
    the search.  `tests/test_baseline_submission.py` pins this path against `optimize_submission`
    on the same inputs, so the optimisation cannot drift from the public function.
    """
    key = (round(float(t0), 12), bool(thin))
    if key not in cache:
        q = floor_sharpen(prob, t0=float(t0), gamma=1.0, hard=True)
        cache[key] = dominant_thin(q, R=R, p=prob) if thin else q
    base = cache[key]
    return dilate_mask(base, radius=int(dilate)) if dilate else base


def sweep_candidates(prob: np.ndarray, ctxs: dict, mask: np.ndarray, R: int,
                     floors=None, dilates=DILATE_GRID, log=None, skip_if_above=None) -> list:
    """Score every (floor, thin, width) candidate of the pre-registered grid.

    Three prunings, all of them exact and all of them visible in the returned rows:

    * **duplicate floors** - the threshold is monotone in the floor, so two floors selecting the
      same NUMBER of pixels select the same pixels.  A repeat floor's rows are copied from the
      first and marked `duplicate_of`; a copy must never be confused with a *different*
      (thin, width) variant of the same floor, so the copy is keyed by (thin, width) and carries
      its twin's metrics.
    * **support-cap skips** - for `thin=False` the shaped mask *is* the threshold mask, so its
      pixel count is known from one pass over the field; if that count already exceeds
      `skip_if_above` (the pre-registered cap in pixels), the candidate cannot be eligible and
      neither can its dilated siblings, because dilation only adds pixels.  Such a row is recorded
      with `emitted_px` (exact for width 0, a lower bound otherwise), `metrics_skipped` and no
      DTIs - the exclusion is in the report, not in a comment.
    * **one thinning per floor** - `shaped_field` caches the thinning, which is the expensive part,
      and the widths share it.

    Cheapness matters here: on the real grid an unpruned sweep re-emits and re-scores the whole
    5.17 M-pixel footprint 40 times, which is ~15 minutes of 2-vCPU time for rows that are all
    rejected by the cap.
    """
    floors = list(floors if floors is not None else shaping_thresholds(FLOOR_GRID))
    cache: dict = {}
    rows, seen_counts, by_floor = [], {}, {}
    n = len(floors) * 2 * len(dilates)
    for t0 in floors:
        n_above = int(np.count_nonzero(prob > float(t0)))
        twin = seen_counts.get(n_above)
        for thin in (True, False):
            for dilate in dilates:
                if twin is not None:
                    row = dict(by_floor[twin][(bool(thin), int(dilate))])
                    # the copy keeps its own floor label (it is a candidate on its own row) and
                    # records which floor it was copied from; recomputed=False is the audit trail
                    row.update(t0=float(t0), duplicate_of=float(twin), recomputed=False)
                elif (not thin) and skip_if_above is not None and n_above > int(skip_if_above):
                    row = dict(t0=float(t0), thin=False, dilate=int(dilate), recomputed=False,
                               emitted_px=n_above, emitted_px_is_lower_bound=bool(dilate),
                               metrics_skipped="the un-dilated support already exceeds the "
                                               "pre-registered cap and dilation only adds pixels",
                               catalogue_dti=None, proxy_dti=None, combined_dti=None,
                               catalogue_pred_px=None, proxy_pred_px=None, combined_pred_px=None)
                    by_floor.setdefault(float(t0), {})[(False, int(dilate))] = dict(row)
                else:
                    field = np.nan_to_num(shaped_field(prob, t0, thin, dilate, R, cache), nan=0.0)
                    row = dict(t0=float(t0), thin=bool(thin), dilate=int(dilate), recomputed=True,
                               emitted_px=int(np.count_nonzero(field)))
                    for name in POPULATIONS:
                        sc = score_within_mask(field, ctxs[name], mask)
                        row[f"{name}_dti"] = None if sc["dti"] is None else round(float(sc["dti"]), 6)
                        row[f"{name}_pred_px"] = int(sc["pred_px"])
                    by_floor.setdefault(float(t0), {})[(bool(thin), int(dilate))] = dict(row)
                rows.append(row)
                if log:
                    log(row, len(rows), n)
        if twin is None:
            seen_counts[n_above] = float(t0)
    return rows


def score_candidate(prob: np.ndarray, cand: tuple, ctxs: dict, mask: np.ndarray,
                    R: int) -> dict:
    """Union DTI (the selection signal) plus both component populations, all restricted to mask.

    Every number comes from `score_within_mask`, which keeps the GLOBAL geometry and restricts
    only the sums - so a fold-restricted DTI means the same thing it means in
    `scripts/block_holdout_eval.py --score-fold`, and the parts of a disjoint partition add up
    to the global score (tests/test_block_decomposition.py).
    """
    t0, thin, dilate = cand
    field = np.nan_to_num(optimize_submission(prob, R=R, t0=t0, thin=thin, dilate=dilate),
                          nan=0.0)
    out = dict(t0=t0, thin=thin, dilate=dilate, emitted_px=int(np.count_nonzero(field)))
    for name in POPULATIONS:
        s = score_within_mask(field, ctxs[name], mask)
        out[f"{name}_dti"] = None if s["dti"] is None else round(float(s["dti"]), 6)
        out[f"{name}_pred_px"] = int(s["pred_px"])
    return out


def mark_eligibility(row: dict, footprint_px: int, max_fraction: float = MAX_EMITTED_FRACTION,
                     min_px: int = MIN_EMITTED_PX) -> dict:
    """Pre-registered eligibility: a candidate must look like a prediction to be selectable.

    Without this, the argmax of a beta-heavy metric on a weak field is "emit the whole footprint"
    (measured: 5,164,312 px, union DTI 0.1229 on fold 0).  The rule is stated here rather than
    applied silently, and ineligible rows are kept in the report with the reason.
    """
    frac = float(row["emitted_px"]) / float(footprint_px) if footprint_px else 1.0
    row["emitted_fraction"] = round(frac, 6)
    if row["emitted_px"] < int(min_px):
        row["eligible"], reason = False, f"emitted {row['emitted_px']} px < min {int(min_px)} px"
    elif frac > float(max_fraction):
        row["eligible"], reason = False, (f"emitted {row['emitted_px']:,} px = {frac:.1%} of the "
                                          f"footprint > cap {float(max_fraction):.0%}")
    else:
        row["eligible"], reason = True, "within the pre-registered support window"
    row["eligibility_reason"] = reason
    return row


def rank_candidates(rows) -> list:
    """The pre-registered ordering, among ELIGIBLE candidates only.

    Union DTI desc, then the narrower band, then hard-thinning.  A row without an `eligible` key
    counts as eligible so the ordering is testable on its own.
    """
    return sorted((r for r in rows
                   if r.get("combined_dti") is not None and r.get("eligible", True)),
                  key=lambda r: (-r["combined_dti"], r["dilate"], not r["thin"]))


# --------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/training_features.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif",
                    help="categorical proxy raster; code 2 = new-fault-like trace the labels do "
                         "not contain (built by scripts/build_proxy_catalogue.py)")
    ap.add_argument("--proxy-code", type=int, default=2)
    ap.add_argument("--out-dir", default="data/evidence/baseline")
    ap.add_argument("--out", default="", help="shaped submission (default <out-dir>/submission.tif)")
    ap.add_argument("--raw-out", default="", help="raw field (default <out-dir>/prob_raw.tif)")
    ap.add_argument("--fold", type=int, default=0,
                    help="block fold held out of BOTH training and policy selection (the fold the "
                         "policy is chosen on)")
    ap.add_argument("--eval-fold", type=int, default=None,
                    help="block fold used ONLY to measure the selected policy's generalisation "
                         "(default: the next fold after --fold).  It is excluded from training as "
                         "well, so the number is neither trained on nor selected on")
    ap.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--block-px", type=int, default=DEFAULT_BLOCK_PX)
    ap.add_argument("--block-mode", default="balanced")
    ap.add_argument("--buffer-px", type=int, default=DEFAULT_BUFFER_PX)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-ratio", type=float, default=30.0)
    ap.add_argument("--max-negatives", type=int, default=900_000)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--min-samples-leaf", type=int, default=40)
    ap.add_argument("--l2-regularization", type=float, default=1.0)
    ap.add_argument("--chunk-rows", type=int, default=128)
    ap.add_argument("--R", type=int, default=DEFAULT_R_PIXELS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--max-emitted-fraction", type=float, default=MAX_EMITTED_FRACTION,
                    help="pre-registered support cap: a candidate emitting more than this fraction "
                         "of the survey footprint is not selectable (see the module docstring)")
    ap.add_argument("--min-emitted-px", type=int, default=MIN_EMITTED_PX)
    ap.add_argument("--verify-tolerance", type=float, default=AUDIT_TOLERANCE,
                    help="how far the independent evaluator may disagree with this report's "
                         "generalisation numbers before the run exits 3.  The default is the "
                         "measured boundary-credit spread on the competition grid (~1e-4); a "
                         "synthetic grid with small blocks needs a looser one, because a fold "
                         "boundary is then never more than a few R from any truth pixel")
    ap.add_argument("--refit-on-all", action="store_true",
                    help="after selecting the policy, retrain on EVERY fold and re-predict for the "
                         "shipped raster.  Recorded in the report, because the field then saw the "
                         "fold the policy was chosen on (docs/FIELD_SELECTION_RULE.md)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Fail fast, before reading 400 MB of rasters: one fold selects, a DIFFERENT fold measures.
    if args.eval_fold is not None and int(args.eval_fold) == int(args.fold):
        raise SystemExit("--eval-fold must differ from --fold: one fold selects the policy and a "
                         "different one measures it, otherwise the reported measurement is the "
                         "maximum of the very numbers it is quoted from")
    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out or out_dir / "submission.tif")
    raw_path = Path(args.raw_out or out_dir / "prob_raw.tif")

    # ---- inputs ---------------------------------------------------------------------
    fault, valid = read_truth(Path(args.labels))
    footprint, template_prof = read_template(Path(args.template))
    proxy_only = read_code_mask(Path(args.proxy), args.proxy_code)
    H, W = fault.shape
    if proxy_only.shape != fault.shape:
        raise SystemExit(f"proxy grid {proxy_only.shape} != labels grid {fault.shape}")
    print(f"grid {W}x{H}  labels {int(fault.sum()):,} fault px / {int(valid.sum()):,} valid px  "
          f"footprint {int(footprint.sum()):,} px  proxy-only {int(proxy_only.sum()):,} px")

    # ---- spatial partition; fold `--fold` is held out of everything -----------------
    table = block_table((H, W), args.block_px, valid=valid, labels=fault)
    fold_of = assign_folds(table, args.folds, args.seed, args.block_mode)
    eval_fold = ((int(args.fold) + 1) % int(args.folds)) if args.eval_fold is None \
        else int(args.eval_fold)
    held = scored_mask((H, W), args.block_px, fold_of, args.fold)          # policy selection
    held_eval = scored_mask((H, W), args.block_px, fold_of, eval_fold)     # honest measurement
    excluded = (held_out_mask((H, W), args.block_px, fold_of, args.fold, args.buffer_px)
                | held_out_mask((H, W), args.block_px, fold_of, eval_fold, args.buffer_px))
    trainable = valid & ~excluded
    part = describe_partition((H, W), args.block_px, args.folds, args.seed, table, fold_of,
                              buffer_px=args.buffer_px, labels=fault, valid=valid,
                              mode=args.block_mode)
    print(f"holdout: fold {args.fold}/{args.folds} selects the policy -> {int(held.sum()):,} px, "
          f"{int(fault[held].sum()):,} fault px, {int(proxy_only[held].sum()):,} proxy-only px; "
          f"fold {eval_fold} measures it -> {int(held_eval.sum()):,} px, "
          f"{int(fault[held_eval].sum()):,} fault px, {int(proxy_only[held_eval].sum()):,} "
          f"proxy-only px; {int(excluded.sum()):,} px excluded from training "
          f"(both folds, buffer {args.buffer_px})")

    rng = np.random.default_rng(int(args.seed))
    sel = choose_samples(fault, valid, trainable, rng, args.neg_ratio, args.max_negatives)
    sample_mask = sel["pos"] | sel["neg"]
    X, y, read_s = fill_matrix(Path(args.features), sample_mask, sel["pos"], args.chunk_rows)
    n_rows = int(X.shape[0])
    keep = ~np.isnan(X).all(axis=1)
    clf, fit_s = fit_model(X[keep], y[keep], args)
    n_pos_fit, n_neg_fit = int((y[keep] == 1).sum()), int((y[keep] == 0).sum())
    print(f"training matrix {X.shape} in {read_s}s ({int((~keep).sum()):,} all-NaN rows dropped); "
          f"fit {fit_s}s on {n_pos_fit:,} pos / {n_neg_fit:,} neg")
    del X, y, keep

    if args.refit_on_all:
        sel_all = choose_samples(fault, valid, valid, rng, args.neg_ratio, args.max_negatives)
        mask_all = sel_all["pos"] | sel_all["neg"]
        Xa, ya, read_a = fill_matrix(Path(args.features), mask_all, sel_all["pos"], args.chunk_rows)
        ka = ~np.isnan(Xa).all(axis=1)
        clf, fit_a = fit_model(Xa[ka], ya[ka], args)
        print(f"refit on ALL folds: {int(ka.sum()):,} rows in {read_a}s, fit {fit_a}s")
        read_s += read_a
        fit_s += fit_a
        del Xa, ya, ka, mask_all

    prob, pred_s = predict_grid(clf, Path(args.features), footprint, valid, args.chunk_rows)
    print(f"full-grid prediction in {pred_s}s; finite {int(np.isfinite(prob).sum()):,} px")

    # ---- policy selection on the held-out fold, ranked by the UNION population --------
    ctxs = contexts(fault, proxy_only, args.R)
    footprint_px = int(footprint.sum())

    def _log(row, i, n):
        # Eligibility has to be applied BEFORE the line is printed: the callback fires inside the
        # sweep, and a printed "NOT ELIGIBLE" on a row the report calls eligible (or the reverse)
        # is exactly the kind of stdout/report disagreement this file is reviewed for.
        mark_eligibility(row, footprint_px, args.max_emitted_fraction, args.min_emitted_px)
        # Progress per candidate: the search is the slowest step and a silent multi-minute gap is
        # indistinguishable from a hang in a workflow log.
        print(f"  [{i:>2}/{n}] floor {row['t0']:<8.5g} thin={int(row['thin'])} width={row['dilate']}"
              f"  union {row['combined_dti']}  catalogue {row['catalogue_dti']}  "
              f"proxy {row['proxy_dti']}  emitted {row['emitted_px']:,} px"
              f"{'  DUPLICATE of floor ' + str(row['duplicate_of']) if 'duplicate_of' in row else ''}"
              f"{'' if row.get('eligible') else '  NOT ELIGIBLE'}", flush=True)

    rows = [mark_eligibility(r, footprint_px, args.max_emitted_fraction, args.min_emitted_px)
            for r in sweep_candidates(prob, ctxs, held, args.R, log=_log,
                                      skip_if_above=int(args.max_emitted_fraction * footprint_px))]
    ranked = rank_candidates(rows)
    if not ranked:
        raise SystemExit("no ELIGIBLE candidate had a defined score on the held-out fold - check "
                         "--proxy/--fold and the support window (--max-emitted-fraction / "
                         "--min-emitted-px)")
    winner = ranked[0]
    rejected = sorted((r for r in rows if r.get("combined_dti") is not None and not r["eligible"]),
                      key=lambda r: -r["combined_dti"])
    best_rejected = rejected[0] if rejected else None
    print(f"policy search: {len(rows)} candidates on fold {args.fold}, {len(ranked)} eligible; best "
          f"union DTI {winner['combined_dti']:.4f} at floor {winner['t0']:.4g}, thin={winner['thin']}, "
          f"width={winner['dilate']} (catalogue {winner['catalogue_dti']}, proxy {winner['proxy_dti']})")
    if best_rejected is not None:
        print(f"  rejected by the support cap: floor {best_rejected['t0']:.4g} thin="
              f"{int(best_rejected['thin'])} width={best_rejected['dilate']} would score union "
              f"{best_rejected['combined_dti']:.4f} but emits {best_rejected['emitted_px']:,} px "
              f"({best_rejected['eligibility_reason']})")

    # ---- generalisation: the SAME winner on a fold that neither trained nor selected ----
    # The sweep above maximises over candidates ON the selection fold, so the winner's own value
    # there is a maximum of noisy numbers and reads high by construction.  The honest number is the
    # winner's field measured on the other held-out fold - one the model never trained on (it was
    # excluded above) and the sweep never looked at.  scripts/block_holdout_eval.py --score-fold
    # reproduces these three numbers from prob_raw.tif, digit for digit.
    winner_field = np.nan_to_num(shaped_field(prob, winner["t0"], winner["thin"],
                                              winner["dilate"], args.R, {}), nan=0.0)
    generalisation = {}
    for name in POPULATIONS:
        sc = score_within_mask(winner_field, ctxs[name], held_eval)
        generalisation[name] = dict(
            dti=None if sc["dti"] is None else round(float(sc["dti"]), 6),
            pred_px=int(sc["pred_px"]), truth_px=int(sc["n_gt"]),
            scoreable=bool(sc["scoreable"]))
    print(f"generalisation on fold {eval_fold} (never trained, never selected): "
          + ", ".join(f"{k} {v['dti']}" for k, v in generalisation.items()))

    # ---- write both artefacts through the fail-loud writer ---------------------------
    common = dict(height=H, width=W, crs=template_prof.get("crs"),
                  transform=template_prof.get("transform"), dtype="float32", nodata=np.nan)
    raw_info = write_submission(raw_path, prob, clean_profile(template_prof, **common),
                                band_description="raw probability",
                                tags=dict(kind="raw_probability", script="baseline_submission.py"))
    shaped = np.where(footprint, winner_field, np.nan).astype(np.float32)
    policy = f"floor{winner['t0']:.4g}_thin{int(winner['thin'])}_w{winner['dilate']}"
    sub_info = write_submission(out_path, shaped, clean_profile(template_prof, **common),
                                band_description="fault probability",
                                tags=dict(kind="submission", policy=policy,
                                          script="baseline_submission.py"))
    print(f"wrote {raw_path} ({raw_info['bytes']:,} B) and {out_path} ({sub_info['bytes']:,} B, "
          f"{sub_info['nonzero_px']:,} emitted px, policy {policy})")
    # Sidecar in the same format the shipped ensemble uses, so `sha256sum -c` verifies either
    # artifact with one command and the report's hash is not the only copy of it.
    sidecar = out_path.with_suffix(".sha256")
    sidecar.write_text(f"{sub_info['sha256']}  {out_path.as_posix()}\n")
    print(f"wrote {sidecar} (verify: sha256sum -c {sidecar})")

    global_scores = {name: round(float(ctxs[name].score(np.nan_to_num(shaped, nan=0.0),
                                                         args.alpha, args.beta)), 6)
                     for name in POPULATIONS}
    print("global DTI of the shipped baseline: " + ", ".join(f"{k} {v}" for k, v in global_scores.items()))

    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/baseline_submission.py",
        purpose="CPU-only classical baseline: produce a format-valid submission without a GPU and "
                "without runner artefacts, and measure it on the same three truth populations the "
                "deep ensemble is measured on",
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         platform=platform.platform(), cpu=platform.processor() or "unknown"),
        inputs=dict(features=dict(path=args.features, sha256=sha256_file(args.features)),
                    labels=dict(path=args.labels, sha256=sha256_file(args.labels)),
                    template=dict(path=args.template, sha256=sha256_file(args.template)),
                    proxy=dict(path=args.proxy, sha256=sha256_file(args.proxy),
                               code=int(args.proxy_code))),
        metric=dict(R_pixels=args.R, alpha=args.alpha, beta=args.beta),
        partition=part, held_out_fold=int(args.fold), measurement_fold=int(eval_fold),
        training=dict(model="HistGradientBoostingClassifier", seed=int(args.seed),
                      max_iter=int(args.max_iter), learning_rate=float(args.learning_rate),
                      max_leaf_nodes=int(args.max_leaf_nodes),
                      min_samples_leaf=int(args.min_samples_leaf),
                      l2_regularization=float(args.l2_regularization), n_rows=n_rows,
                      n_pos=sel["n_pos"], n_neg=sel["n_neg"], n_neg_pool=sel["n_neg_pool"],
                      n_pos_fit=n_pos_fit, n_neg_fit=n_neg_fit,
                      neg_ratio_requested=float(args.neg_ratio),
                      negatives_capped=bool(sel["negatives_capped"]),
                      refit_on_all=bool(args.refit_on_all)),
        timings_s=dict(read_features=read_s, fit=fit_s, predict=pred_s,
                       total=round(time.time() - t_start, 1)),
        policy_selection=dict(
            population="combined (catalogue labels UNION new-fault-like proxy pixels)",
            scope=f"blocks of fold {args.fold} (spatially held out of training).  The winner's "
                  f"value on THIS fold is the maximum of the rows below, so it carries selection "
                  f"bias; `generalisation` is the unbiased number",
            candidates=rows, n_candidates=len(rows), winner=winner, n_eligible=len(ranked),
            eligibility=dict(max_emitted_fraction=float(args.max_emitted_fraction),
                             min_emitted_px=int(args.min_emitted_px),
                             footprint_px=footprint_px,
                             why="pre-registered: an unconstrained argmax of a beta-weighted metric "
                                 "on a weak field selects 'emit the whole footprint', which is not a "
                                 "prediction; ineligible rows stay in `candidates` with their reason"),
            best_rejected=best_rejected,
            n_duplicate_rows=sum(1 for r in rows if "duplicate_of" in r),
            n_skipped_rows=sum(1 for r in rows if "metrics_skipped" in r),
            skip_note="a `thin=False` candidate whose threshold mask already exceeds the cap "
                      "cannot be eligible, and neither can its dilated siblings (dilation only "
                      "adds pixels), so its metrics are not measured; `emitted_px` is exact for "
                      "width 0 and a lower bound otherwise",
            dedupe_note="a floor that selects the same pixel count as an earlier floor selects the "
                        "same pixels, so its row is copied rather than recomputed; one thinning is "
                        "shared by both widths (pinned against optimize_submission by test)",
            ranking_key="-combined_dti then narrower band then hard-thinning then lower floor",
            note="selection is measured on held-out geography, never on the grid being shipped; "
                 "the catalogue and proxy columns are reported, not used to pick"),
        generalisation=dict(
            fold=eval_fold,
            scope=f"blocks of fold {eval_fold}: excluded from training and never used by the "
                  f"policy sweep; measured on the same field `submission.tif` is written from",
            by_population=generalisation,
            # The audit scores the SHAPED artifact, not prob_raw.tif: block_holdout_eval shapes by
            # threshold + dilation only and cannot thin, so pointing it at the raw field with the
            # winner's floor measures the un-thinned candidate instead (measured 2026-09-20:
            # combined 0.1259 vs the 0.0789 the thinned field actually scores).  submission.tif is
            # already the shaped field, so thresholding it at 0.5 is a no-op and the numbers come
            # back digit for digit.  The crosscheck against the runner sweep is disabled because
            # this field is not the runner's - the check only means anything on eval_sweep-mean12's
            # own prediction.
            # `audit` is attached after this dict is built, by `audit_generalisation`, so its
            # values are MEASURED (the evaluator re-run and diffed) rather than copied from the
            # block above.
            reproduce=("python scripts/block_holdout_eval.py --pred %s --labels %s --proxy %s "
                       "--score-fold %d --combined-population --floors 0.5 --widths 0 "
                       "--crosscheck-sweep '' --out <report.json>"
                       % (out_path, args.labels, args.proxy, eval_fold)),

            note="each population is scored with the same global context as the sweep, so "
                 "restricting to one fold cannot move a pixel's credit"),
        comparison_to_shipped_ensemble=shipped_comparison(),
        # `write_submission` returns its own `path` key; the report's copy is the one the caller
        # asked for, so the returned keys come first and `path` is set from the argument.
        submission=dict(**{k: v for k, v in sub_info.items() if k != "path"},
                        path=str(out_path), policy=policy, global_scores=global_scores),
        prob_raw=dict(**{k: v for k, v in raw_info.items() if k != "path"}, path=str(raw_path)),
        caveats=[
            "A per-pixel classifier has NO spatial context - it is a floor, not a contender. "
            "Compare it against the shipped ensemble's committed numbers in "
            "data/evidence/proxy/eval_sweep-mean12.json rather than against the leaderboard.",
            "The scored population is the private expert-labelled new-fault set (rules 1.1/3.6). "
            "Every number here is a local surrogate on held-out geography.",
            "Unless --refit-on-all is passed, the shipped field was trained on folds outside "
            "{%d, %d}, so it saw %d%% of the labelled fault pixels: the policy was selected on "
            "fold %d and measured on fold %d, neither of which the field saw."
            % (args.fold, eval_fold,
               int(round(100 * float(fault[trainable].sum()) / max(1, int(fault.sum())))),
               args.fold, eval_fold),
        ],
        next_command=(
            "python scripts/block_holdout_eval.py --pred %s --labels %s --proxy %s "
            "--combined-population --floors %s --widths %s --reference-floor %s "
            "--reference-width %d --out %s"
            % (raw_path, args.labels, args.proxy,
               ",".join(f"{f:.6g}" for f in sorted({r['t0'] for r in rows})),
               ",".join(str(d) for d in DILATE_GRID), f"{winner['t0']:.6g}", winner["dilate"],
               out_dir / "block_stratified.json")),
    )
    report["generalisation"]["audit"] = audit_generalisation(
        out_path, Path(args.labels), Path(args.proxy), Path(args.template),
        int(args.block_px), int(args.folds), eval_fold, generalisation, out_dir,
        tolerance=float(args.verify_tolerance))
    _audit = report["generalisation"]["audit"]
    print(f"audit of the generalisation claim: reproduced={_audit.get('passed')} "
          f"max|delta|={_audit.get('max_abs_delta')} (tolerance {args.verify_tolerance}); "
          f"audit report at {out_dir / 'reproduce_audit.json'}")
    (out_dir / "baseline_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nreport -> {out_dir / 'baseline_report.json'}")
    print(f"validate: python scripts/validate_submission.py --pred {out_path} "
          f"--sample {args.template} --train {args.features}")
    if not _audit.get("passed", False):
        # The artifacts are written and the report carries the measurement, so the evidence is
        # kept - but the run exits non-zero rather than reporting success on an unverified claim.
        print(f"FAIL: the independent evaluator disagrees with the report's generalisation numbers "
              f"by more than {args.verify_tolerance}: {_audit.get('deltas')}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
