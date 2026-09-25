"""Discovery diagnostics: how much of a prediction lives OUTSIDE the known-fault catalog.

WHY THIS MODULE EXISTS (the single most consequential fact in this competition)
------------------------------------------------------------------------------
Official rules, §1.1 (fetched 2026-09-16, https://docs.nlr.gov/docs/fy26osti/96647.pdf):

    "In Phase 1, submissions will be evaluated against a privately withheld subset of the
     original NEW fault dataset compiled by expert reviewers."
    "Submissions will be reevaluated against the full, revised NEW fault dataset using the
     same distance-weighted Tversky index."

and §3.3:

    "The training labels contain EXISTING fault data at 100-m resolution ... These labels
     were obtained from the INGENIOUS project's Great Basin Regional Dataset Compilation."

So the thing we train on (the existing/known fault catalog) and the thing we are scored on
(faults that are *not* in that catalog) are DISJOINT populations.  Every DTI number computed
against the known catalog in this repo measures the wrong universe: a model that perfectly
reproduces `labels.tif` would score ~1.0 against the catalog and, by construction, ~0 against
the competition target.  The catalog-DTI is a *plumbing* monitor.

What is missing is a number that says how much of a prediction is a *discovery*.  This module
computes it, with no extra data and no trained model:

    novel_mass        = sum_x p(x) * (1 - max_{g in catalog} k(d(x,g)))     [FP_w vs catalog]
    novel_fraction    = novel_mass / sum_x p(x)
    catalog_recall_R  = |{g in catalog : exists x with p(x)>thr, d(x,g)<=R}| / |catalog|
    candidate_new     = connected components of {p > thr} that are farther than R from every
                        cataloged fault pixel, with their area / bounding box / length

Interpretation.  A model that merely memorises the catalog has novel_fraction ~ 0 and
candidate_new ~ 0 -> it cannot score on the competition target no matter how high its
catalog-DTI is.  A model with a high novel_fraction is proposing faults the catalog does not
contain: those are the pixels that can earn TP credit in Phase 1/2, and the features the
expert panel would review before Phase 2.  This is a *diagnostic*, not a score: a submission
of pure noise also has novel_fraction ~ 1, which is why it is always reported next to the
catalog-DTI, the positive area and the component statistics.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .metrics import kernel_offsets

__all__ = ["credit_map_known", "novel_mass", "candidate_new_faults", "discovery_report"]

# 8-connectivity: fault traces are thin diagonal lines, so 4-connectivity would shred them
_CONN8 = np.ones((3, 3), dtype=bool)


def credit_map_known(known: np.ndarray, R: int = 3) -> np.ndarray:
    """m(g) = max_{g' in catalog, d<=R} k(d(x,g')) for every pixel x.

    Same exhaustive (2R+1)^2 kernel enumeration as src/metrics.py, so ``1 - m`` is exactly
    the false-positive weight the official metric would apply if the catalog *were* the
    ground truth.
    """
    known_bool = np.asarray(known) > 0.5
    if not known_bool.any():
        return np.zeros(known_bool.shape, dtype=np.float64)
    # distance to the *nearest* catalog pixel, then triangular kernel; exact for integer
    # pixel distances because the EDT is Euclidean and the kernel is monotone in d
    d = distance_transform_edt(~known_bool)
    m = np.maximum(1.0 - d / float(R), 0.0)
    return m


def novel_mass(pred: np.ndarray, known: np.ndarray, R: int = 3) -> dict:
    """Probability mass that sits farther than R from any cataloged fault pixel."""
    p = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(p, 0.0, 1.0, out=p)
    m = credit_map_known(known, R=R)
    novel = float(np.sum(p * (1.0 - m)))
    total = float(np.sum(p))
    return dict(R_pixels=int(R), prob_mass=total, novel_mass=novel,
                catalog_adjacent_mass=float(total - novel),
                novel_fraction=float(novel / total) if total > 0 else None)


def candidate_new_faults(pred: np.ndarray, known: np.ndarray, R: int = 3,
                         thr: float = 0.5, min_px: int = 3) -> dict:
    """Connected components of the positive mask that never come within R of the catalog."""
    p = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(p, 0.0, 1.0, out=p)
    pos = p > float(thr)
    if not pos.any():
        return dict(threshold=float(thr), positive_px=0, n_components=0, n_novel_components=0,
                    novel_px=0, novel_px_fraction=None, largest_novel_px=0,
                    novel_bboxes=[], novel_component_px=[])
    # dilate the catalog by R and ask which components miss it
    m = credit_map_known(known, R=R) > 0.0
    lab, n = label(pos, structure=_CONN8)
    idx = np.arange(1, n + 1)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    # does the component contain (or touch) any pixel within R of the catalog?
    touches = np.zeros(n + 1, dtype=bool)
    near = np.unique(lab[m & (lab > 0)])
    touches[near] = True
    novel_ids = idx[~touches[1:]]
    novel_sizes = sizes[novel_ids - 1] if novel_ids.size else np.zeros(0, dtype=np.int64)
    keep = novel_ids[novel_sizes >= int(min_px)]
    boxes = []
    for cid in keep.tolist():
        ys, xs = np.where(lab == cid)
        boxes.append(dict(px=int(ys.size), y0=int(ys.min()), y1=int(ys.max()),
                          x0=int(xs.min()), x1=int(xs.max()),
                          length_px=int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)))
    boxes.sort(key=lambda b: -b["px"])
    novel_px = int(novel_sizes[novel_sizes >= int(min_px)].sum()) if novel_ids.size else 0
    # the size list is derived from the same largest-first boxes the report shows, so the two
    # fields cannot disagree (they used to: boxes were largest-first while this list was the
    # last 25 component ids in label order, unfiltered by min_px — fixed 2026-09-16, session 6)
    return dict(threshold=float(thr), positive_px=int(pos.sum()), n_components=int(n),
                n_novel_components=int(keep.size), novel_px=novel_px,
                novel_px_fraction=float(novel_px / int(pos.sum())) if pos.sum() else None,
                largest_novel_px=int(novel_sizes.max()) if novel_sizes.size else 0,
                novel_bboxes=boxes[:25], novel_component_px=[b["px"] for b in boxes[:25]])


def discovery_report(pred: np.ndarray, known: np.ndarray, R: int = 3,
                     thresholds=(0.1, 0.5, 0.9), alpha: float = 0.2, beta: float = 0.8,
                     min_px: int = 3) -> dict:
    """One dict with the discovery diagnostics (JSON-serialisable, no NaN/inf)."""
    known_bool = np.asarray(known) > 0.5
    out = dict(
        catalog_px=int(known_bool.sum()),
        catalog_coverage_fraction=float(known_bool.mean()),
        novel_mass=novel_mass(pred, known_bool, R=R),
        per_threshold={str(t): candidate_new_faults(pred, known_bool, R=R, thr=float(t),
                                                    min_px=min_px) for t in thresholds},
    )
    # catalog recall with the R tolerance (how much of the known catalog is "hit")
    p = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0)
    for t in thresholds:
        pos = p > float(t)
        if pos.any() and known_bool.any():
            d = distance_transform_edt(~pos)
            out[f"catalog_recall_R_t{t}"] = float((d[known_bool] <= R).mean())
        else:
            out[f"catalog_recall_R_t{t}"] = 0.0
    out["note"] = ("Diagnostics vs the KNOWN catalog = training labels. The competition scores "
                   "the NEW fault dataset (rules §1.1/§3.3), which is disjoint from the catalog, "
                   "so catalog-adjacent mass is NOT scored as TP. Use novel_fraction + candidate "
                   "components as the discovery signal; a pure-noise submission also scores "
                   "novel_fraction ~ 1, so always read them next to the catalog DTI and the area.")
    return out
