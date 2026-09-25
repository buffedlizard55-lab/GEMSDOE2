#!/usr/bin/env python3
"""What does the DTI metric actually reward?  Measured, not assumed.

This matters more than model architecture.  The scoring function is unusual, and a
closed-form rearrangement of it changes how a submission should be shaped.

ALGEBRA (exact, from the official definition)
--------------------------------------------
The problem page defines
    TP_w = sum_{g in G} max_{x: d(x,g)<=R} p(x) k(d(x,g))
    FN_w = sum_{g in G} [1 - max_{x: d(x,g)<=R} p(x) k(d(x,g))]
so term-by-term  TP_w + FN_w = |G|  identically, for ANY prediction p.
Substituting FN_w = |G| - TP_w into DTI = TP_w / (TP_w + a FP_w + b FN_w):

    DTI = TP_w / ( (1-b) TP_w + a FP_w + b |G| )
        = TP_w / ( 0.2 TP_w + 0.2 FP_w + 0.8 |G| )        [a=0.2, b=0.8]
        = TP_w / ( 0.2 (TP_w + FP_w) + 0.8 |G| )

Consequences, which this script verifies numerically:
  1. |G| is a CONSTANT of the test set. The denominator's 0.8|G| term is a fixed floor,
     so the achievable range is bounded and small unless TP_w approaches |G|.
  2. TP_w and FP_w enter with the SAME coefficient 0.2. Every unit of probability mass
     you add costs 0.2 in the denominator regardless of where it lands; it only earns
     numerator credit if it is the nearest-and-strongest mass to a ground-truth pixel.
  3. TP_w saturates: a ground-truth pixel contributes at most 1, achieved by ANY pixel
     within R at p=1 (k=1 needs d=0, but p=1 at d=1 gives 0.667). So piling extra mass
     near an already-covered fault is pure FP cost. -> THINNING is valuable.
  4. Because b=0.8 >> a=0.2, missing a fault is 4x worse than a spurious one.
     -> RECALL-first, and blanket coverage is a real (if weak) baseline.

BASELINES COMPUTED HERE (on whatever labels you point it at):
  * all-zeros, all-ones, uniform-p, and "dilate the known faults by r" families
  * the score of predicting the KNOWN public faults themselves
These bound what any model must beat, and are printed with their TP/FP/FN parts.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import compute_distance_weighted_tversky as dti  # noqa: E402


def parts(p, g, R=3, a=0.2, b=0.8):
    d, (tp, fp, fn) = dti(p, g, R_pixels=R, alpha=a, beta=b, return_components=True)
    return d, tp, fp, fn


def closed_form_check(tp, fp, n_g, a=0.2, b=0.8, eps=1e-7):
    """DTI computed via the rearranged formula; must match the direct computation."""
    return tp / ((1 - b) * tp + a * fp + b * n_g + eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="label GeoTIFF; default = dev fixture")
    ap.add_argument("--out", default="data/evidence/metric_strategy.json")
    ap.add_argument("--R", type=int, default=3)
    a_ = ap.parse_args()

    if a_.labels:
        import rasterio

        with rasterio.open(a_.labels) as src:
            y = src.read(1).astype(np.float32)
            nod = src.nodata
        valid = np.isfinite(y)
        if nod is not None:
            valid &= y != nod
        g = ((y > 0.5) & valid).astype(np.float32)
        src_name = a_.labels
    else:
        from src.fixture import load_fixture

        X, g, meta, man = load_fixture()
        valid = np.isfinite(X[..., 0])
        src_name = "data/fixture (real competition window)"

    R = a_.R
    n_g = int(g.sum())
    n_valid = int(valid.sum())
    print(f"labels: {src_name}")
    print(f"grid {g.shape}  |G|={n_g}  valid={n_valid}  fault frac={n_g / max(n_valid,1):.4%}\n")

    rows = []

    def add(name, p, note=""):
        p = np.clip(np.nan_to_num(p, nan=0.0), 0, 1).astype(np.float32)
        p = p * valid            # never predict outside the data footprint
        d, tp, fp, fn = parts(p, g, R)
        cf = closed_form_check(tp, fp, n_g)
        assert abs(d - cf) < 1e-9, (name, d, cf)     # verifies the algebra above
        rows.append(dict(strategy=name, dti=float(d), TP_w=float(tp), FP_w=float(fp),
                         FN_w=float(fn), mass=float(p.sum()), note=note))
        print(f"  {name:38s} DTI={d:.5f}  TP={tp:9.1f} FP={fp:10.1f} FN={fn:9.1f} "
              f"mass={p.sum():10.0f}")

    print("--- trivial baselines ---")
    add("all zeros", np.zeros_like(g))
    add("all ones (blanket coverage)", np.ones_like(g),
        "upper bound of 'predict everything'; shows the 0.8|G| floor")
    for c in (0.05, 0.1, 0.25, 0.5):
        add(f"uniform p={c}", np.full_like(g, c),
            "uniform mass: TP and FP scale together -> DTI nearly flat in c")

    print("\n--- oracle-ish: the known public faults themselves ---")
    add("exact known faults, p=1", g.copy(), "what a perfect copy of the PUBLIC labels scores")
    for r in (1, 2, 3, 4, 6):
        add(f"known faults dilated r={r}, p=1", binary_dilation(g > 0.5, iterations=r).astype(np.float32),
            "shows the cost of smearing: TP saturates at |G|, FP grows with area")

    print("\n--- soft distance ramps around the truth (what a well-calibrated model looks like) ---")
    d2g = distance_transform_edt(~(g > 0.5))
    for rr in (3, 5, 8):
        ramp = np.clip(1.0 - d2g / rr, 0, 1)
        add(f"triangular ramp radius {rr}px", ramp,
            "soft halo; compare against the hard dilation of the same radius")

    print("\n--- partial recall: what fraction of faults must be found? ---")
    rng = np.random.default_rng(0)
    idx = np.array(np.nonzero(g > 0.5)).T
    for frac in (0.25, 0.5, 0.75):
        keep = rng.choice(len(idx), size=int(frac * len(idx)), replace=False)
        p = np.zeros_like(g)
        p[idx[keep, 0], idx[keep, 1]] = 1.0
        add(f"{frac:.0%} of known faults, p=1", p,
            "DTI is roughly linear in recall -> recall is the dominant lever")

    best = max(rows, key=lambda r: r["dti"])
    payload = {
        "labels_source": src_name,
        "R_pixels": R,
        "alpha": 0.2,
        "beta": 0.8,
        "n_ground_truth": n_g,
        "n_valid_pixels": n_valid,
        "closed_form": "DTI = TP_w / (0.2*(TP_w + FP_w) + 0.8*|G|)   [verified against the direct computation for every row]",
        "identity": "TP_w + FN_w = |G| exactly, for any prediction",
        "rows": rows,
        "best_strategy": best["strategy"],
        "best_dti": best["dti"],
        "interpretation": [
            "TP_w and FP_w share the coefficient 0.2, so probability mass is charged the same "
            "wherever it lands; only mass that is the strongest within R of a ground-truth pixel earns credit.",
            "TP_w per ground-truth pixel saturates at 1, so redundant mass along an already-covered "
            "fault is pure cost -> thinning/skeletonising a confident prediction raises DTI.",
            "beta/alpha = 4, so a missed fault costs 4x a spurious one -> optimise recall first.",
            "Blanket coverage is a non-trivial baseline any model must beat; see the 'all ones' row.",
        ],
    }
    Path(a_.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a_.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nbest: {best['strategy']} DTI={best['dti']:.5f}")
    print(f"wrote {a_.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
