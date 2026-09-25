#!/usr/bin/env python3
"""Score every candidate submission this repository can hand a reader, the way the platform scores.

WHY THIS EXISTS
---------------
By session 26 the repository had six downloadable `.tif` files and no table that put them on the
same scored population.  Three of them were selected with a population that *rewards echoing the
supplied catalogue*: the organizers stated on the forum that pixels corresponding to known
USGS/INGENIOUS faults are masked out of evaluation (forum 11516 #2 and #4, quoted in
`data/evidence/rules_quotes.json`), which means a pixel painted on the catalogue earns exactly
nothing, while the repository's held-out-segment proxy counts it as a hit.

This script measures every candidate under four populations and two controls, all from the bytes:

  L  third-party protocol (gems-eval): scored population = background + held-out segments, the
     3 px buffer around the *training* faults excluded.
  S  organizer-faithful: scored population = footprint minus the supplied catalogue pixels; truth
     = held-out segments; predicted mass next to the training catalogue is charged.
  I  S, with the truth restricted to held-out segments more than 5 px from the training catalogue.
  P  **the mask applied**: as S, but the prediction is zeroed on every supplied-catalogue pixel
     before scoring -- exactly what the platform's pixel-exact mask does to it.  This is the
     number that predicts what the leaderboard will see, and it is the one the recommendation
     uses.
  controls: a uniformly random emission at the same support (what "no detection at all" scores)
     and the held-out truth itself dilated by one pixel (an upper bound for a real detector).

Every candidate is scored from its own bytes on one machine, so the ranking is a property of the
files, not of the prose around them.  Output: `data/evidence/submission_portfolio.json`, which
`scripts/build_site.py` renders.

USAGE
    python scripts/measure_submission_portfolio.py
    python scripts/measure_submission_portfolio.py --json data/evidence/submission_portfolio.json
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

from scripts.segment_holdout_eval import build_split, dti  # noqa: E402
from src.submission_io import conformance_findings  # noqa: E402

#: Every file the site or the READMEs offer a reader, plus the two built this session.
#:
#: A third tuple element, when present, is an options dict.  ``measure`` names the bytes to
#: score when they are not the ones at ``path`` -- the 6-fold run's raw GeoTIFF is the file
#: the platform rejected on 2026-09-24 (3,061 NaN px inside the template's valid region,
#: 1,540 finite px outside it), so its row is scored on the conformed sibling, which is also
#: what ``scripts/package_portfolio.py`` ships.  Scoring bytes that no reader can be handed
#: is how a table and a download drift apart; the gate at the end of ``main`` refuses it.
CANDIDATES = [
    ("dual-family union (the shipped artifact)", "data/evidence/union/submission.tif"),
    ("11-fold ensemble (ens12-adopted, the recall arm)",
     "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"),
    ("6-fold blend (35042805806, the precision arm)", "data/evidence/runs/35042805806/submission.tif",
     {"measure": "data/evidence/runs/35042805806/submission_conformant.tif"}),
    ("suture-v1 (structure-tensor lineaments, this session)",
     "data/evidence/suture/submission.tif"),
    ("ens-only skeleton (suture ablation)", "data/evidence/suture/ens_only.tif"),
    ("recall-union v1 (the site's previous primary)", "docs/gemsdoe2_recall_union_submission.tif"),
    ("extension arm (union + 1 px catalogue corridor, this session)",
     "data/evidence/extension/submission.tif"),
    ("structure-consensus v1", "docs/gemsdoe2_structure_consensus_submission.tif"),
    ("fusion v2", "data/evidence/runs/gemsdoe2-fusion-v2/submission.tif"),
    ("turbo-binary (the run the TURBO README cites)",
     "data/evidence/runs/turbo-binary-20260925T042608Z/submission.tif"),
    ("turbo (the file the site's TURBO button downloads)", "docs/gemsdoe2_turbo_submission.tif"),
    ("cpu baseline (histogram gradient boosting)", "data/evidence/baseline/submission.tif"),
]


#: Candidates whose local score the local populations cannot rank, with the reason.  The extension
#: arm is a corridor drawn along the supplied catalogue; every local truth pixel *is* a pixel of that
#: catalogue, so the corridor hugs the truth by construction and scores 2.6x the random floor on the
#: very population a new-fault detector should be judged on.  It is built, measured and shipped as a
#: pre-registered leaderboard bet (`data/evidence/extension/extension_report.json`), but it may not
#: decide the *local* recommendation, or the recommendation would just be "print the catalogue".
LOCAL_RANKING_EXCLUDED = {
    "data/evidence/extension/submission.tif":
        "corridor along the supplied catalogue: the local truth is made of catalogue pixels, so the "
        "arm's local score measures label proximity, not detection (see its report's H1/H2)",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--split-dir", default="data/evidence/seghold")
    ap.add_argument("--json", default="data/evidence/submission_portfolio.json")
    ap.add_argument("--independent-px", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    for p in (Path(args.labels), Path(args.template)):
        if not p.exists():
            print(f"FAIL: {p} missing - run python scripts/assemble_data_bridge.py",
                  file=sys.stderr)
            return 2
    split_dir = Path(args.split_dir)
    if not (split_dir / "eval_labels.tif").exists():
        build_split(Path(args.labels), split_dir, 0.3, args.seed, 3)

    with rasterio.open(split_dir / "eval_labels.tif") as s:
        ev = s.read(1)
    with rasterio.open(split_dir / "train_labels.tif") as s:
        train_known = s.read(1) == 1
    with rasterio.open(args.labels) as s:
        supplied = s.read(1) >= 1
    with rasterio.open(args.template) as s:
        template_raw = s.read(1)
    footprint = np.isfinite(template_raw)

    valid_L = (ev == 0) | (ev == 1)
    valid_S = footprint & ~train_known
    d_train = ndimage.distance_transform_edt(~train_known).astype(np.float32)
    truth_S = ev == 1
    truth_I = (ev == 1) & (d_train > args.independent_px)
    truth_perfect = np.where(truth_S, 1.0, 0.0).astype(np.float32)
    rng = np.random.default_rng(args.seed)

    rows = []
    nonconformant = []
    for entry in CANDIDATES:
        name, rel = entry[0], entry[1]
        opts = entry[2] if len(entry) > 2 else {}
        measure_rel = str(opts.get("measure") or rel)
        path = ROOT / measure_rel
        if not path.exists():
            rows.append({"name": name, "path": measure_rel, "source_path": rel, "present": False,
                         "error": f"bytes not found: {measure_rel}"})
            print(f"  MISSING {measure_rel}")
            continue
        raw = read(path)
        findings = conformance_findings(raw, template_raw)
        if not findings["conformant"]:
            nonconformant.append(f"{measure_rel} (nan inside {findings['nan_inside_px']}, "
                                 f"finite outside {findings['finite_outside_px']}, "
                                 f"out of range {findings['out_of_range_px']})")
        a = np.nan_to_num(raw, nan=0.0)
        emitted = a > 0
        on_catalogue = int((emitted & supplied).sum())
        p_S = np.where(valid_S, a, 0.0).astype(np.float32)
        p_P = np.where(valid_S & ~supplied, a, 0.0).astype(np.float32)
        row = {
            "name": name, "path": measure_rel, "source_path": rel, "present": True,
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "conformant": bool(findings["conformant"]), "conformance": findings,
            "values_ge_0p5": int((a >= 0.5).sum()),
            "emitted_px": int(emitted.sum()),
            "emitted_on_supplied_catalogue_px": on_catalogue,
            "emitted_off_catalogue_px": int(emitted.sum()) - on_catalogue,
            "dti_L": dti(p_S, truth_S, valid=valid_L),
            "dti_S": dti(p_S, truth_S, valid=valid_S),
            "dti_I": dti(p_S, truth_I, valid=valid_S),
            "dti_P_platform_mask": dti(p_P, truth_S, valid=valid_S),
            "dti_P_platform_mask_I": dti(p_P, truth_I, valid=valid_S),
            "local_ranking_eligible": measure_rel not in LOCAL_RANKING_EXCLUDED,
            "local_ranking_exclusion_reason": LOCAL_RANKING_EXCLUDED.get(measure_rel),
        }
        n = max(row["emitted_off_catalogue_px"], 1)
        idx = rng.choice(np.flatnonzero(valid_S & ~supplied), min(n, 400_000), replace=False)
        r = np.zeros(ev.shape, np.float32)
        r.ravel()[idx] = 1.0
        row["random_control_same_support_P"] = dti(r, truth_S, valid=valid_S)
        row["random_control_n"] = int(idx.size)
        row["lift_over_random"] = (row["dti_P_platform_mask"] /
                                   max(row["random_control_same_support_P"], 1e-12))
        rows.append(row)
        print(f"  {name:48s} L {row['dti_L']:.4f} S {row['dti_S']:.4f} "
              f"I {row['dti_I']:.4f} P {row['dti_P_platform_mask']:.4f} "
              f"(random {row['random_control_same_support_P']:.4f}, "
              f"lift {row['lift_over_random']:.2f}x)")
        del a, p_S, p_P, emitted

    controls = {
        "perfect_field": dti(truth_perfect, truth_S, valid=valid_S),
        "truth_dilated_1px": dti(ndimage.binary_dilation(truth_S, iterations=1).astype(np.float32),
                                 truth_S, valid=valid_S),
        "supplied_catalogue_echo": dti(np.where(supplied, 1.0, 0.0).astype(np.float32),
                                       truth_S, valid=valid_S),
        "random_note": ("a uniformly random emission of the same size; it is the floor a real "
                        "detector must beat, and on this population it is a high floor because "
                        "the metric's TP term is a max over the 300 m cone"),
    }
    scored = [r for r in rows if r.get("present")]
    # the recommendation may only be decided by candidates whose local score means something; see
    # LOCAL_RANKING_EXCLUDED.  Everything is still measured and reported.
    rankable = [r for r in scored if r.get("local_ranking_eligible", True)]
    best = max(rankable, key=lambda r: r["dti_P_platform_mask"]) if rankable else None
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/measure_submission_portfolio.py",
        "format_gate": {
            "rule": ("every scored file must be conformant to data/sample_submission.tif's mask: "
                     "finite and in [0, 1] inside the template's valid region, NaN outside it "
                     "(src/submission_io.conformance_findings)"),
            "all_conformant": not nonconformant,
            "nonconformant": nonconformant,
        },
        "populations": {
            "L": "gems-eval held-out-segment protocol (third party)",
            "S": "organizer-faithful: footprint minus supplied catalogue; truth = held-out segments",
            "I": f"S with truth > {args.independent_px:g} px from the training catalogue",
            "P": "S with the prediction masked on the supplied catalogue (what the platform sees)",
        },
        "sources": {
            "organizer_answer_masking": "https://community.drivendata.org/raw/11516 (#2 and #4)",
            "metric": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
            "third_party_protocol": "https://github.com/Gameassassin777/gems-eval",
        },
        "split": json.loads((split_dir / "split.json").read_text())
        if (split_dir / "split.json").exists() else {},
        "controls": controls,
        "candidates": rows,
        "recommended": {
            "name": best["name"] if best else None,
            "path": best["path"] if best else None,
            "sha256": best["sha256"] if best else None,
            "reason": ("highest platform-masked score (P) among the candidates whose local score "
                       "is admissible; the recommendation uses P because it is the only one of the "
                       "four populations that applies the organizer's pixel-exact mask to the "
                       "prediction"),
            "excluded_from_this_ranking": sorted(LOCAL_RANKING_EXCLUDED),
        },
        "caveat": ("These are local numbers on a stand-in truth set.  A field that paints the "
                   "supplied catalogue scores high under L/S/I and near zero under P -- which is "
                   "why P decides.  Only DrivenData can score against the private new-fault set."),
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.json}")
    print(f"recommended: {report['recommended']['name']} "
          f"(P {best['dti_P_platform_mask']:.4f})" if best else "nothing to recommend")
    if nonconformant:
        print(f"FAIL: {len(nonconformant)} scored candidate(s) are not conformant to the template "
              f"mask, so the table scores bytes no reader can be handed:", file=sys.stderr)
        for n in nonconformant:
            print(f"  {n}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
