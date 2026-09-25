#!/usr/bin/env python3
"""Decide the submission-shaping policy in public, from measurements that disagree.

THE QUESTION
------------
``src/submission_optim.optimize_submission`` shapes the ensemble's probability field into a
submission with three knobs: a floor ``t0``, distance-R thinning (keep one skeleton pixel per
R-neighbourhood), and an emission width ``dilate`` that grows the kept set by k pixels.  The
shipped default is ``t0 = 0.470``, ``thin = True``, ``dilate = 0`` (a ~21,500-pixel skeleton).

Three measurements of the width question exist in this repository, and they point in opposite
directions:

  A. in-domain held-out crops (truth = the catalogue the model trained on)
       skeleton 0.1903  >  1 px 0.1525  >  2 px 0.1281  >  3 px 0.1128  >  6 px 0.0908
     -> narrower wins, decisively.

  B. shifted-label stress test (truth = the same catalogue translated 3-4 px)
       0.0555 -> 0.0991 (3 px) -> 0.1190 (6 px)
     -> wider wins, decisively.

  C. proxy-only population (truth = 6,166 km of USGS SGMC fault trace that the training labels
     do NOT contain; the closest measurable stand-in for the scored "new fault" universe)
     The 2026-09-16 grids moved the FLOOR only through `shaping_thresholds()`'s log spacing, whose
     low end is 0.043 -- so "the floor is binary on this field" was true of the grid, not of the
     field.  The 2026-09-17 sweep (session 11, run 35262778745) swept explicit floors
     0 .. 0.9 and found that the winning policy is a FLOOR change with no widening at all:

       floor 0.1, width 0 px  ->  DTI 0.1365   (shipped policy on the same field: 0.0410)
       floor 0.1, width 1 px  ->  DTI 0.0918   ... widening HURTS once the floor is right,
       floor 0.6, width 20 px ->  DTI 0.0741   ... and every floor >= 0.2 still prefers a band.

    -> the decision variable is the joint (floor, thinning, width) policy, not the width alone.

The disagreement is not noise, it is the point.  A and C measure the SAME operator against
populations that differ in exactly the way the competition cares about: in A the model has seen
the truth, so its trace sits on the label and widening only adds false-positive mass; in C the
truth is a fault population the labels lack, so the trace can be off by more than the metric's
R = 3-pixel tolerance and a wider band converts misses into hits.

WHAT THIS SCRIPT ADDS
---------------------
Absolute DTIs are not comparable across populations, and the scored set's size |G| is unknown,
so "which policy is better" cannot be read off any single number.  But the metric's two error
terms scale differently with |G|: the prediction fixes the wrong-mass F = FP_w, while the missing
mass beta*(|G| - TP_w) grows with the hidden truth.  Writing c = TP_w/|G_proxy| for the measured
coverage fraction and holding it fixed, a policy measured on the proxy population has

    DTI(|G|) = c|G| / ( alpha*(c|G| + F) + beta*|G| )          [exact at |G| = |G_proxy|]

(algebraically identical to the official TP/(TP + alpha*FP + beta*FN) because alpha + beta = 1 --
that identity is asserted below, since the whole projection rests on it).  Two consequences:

  * every policy has an asymptote c/(alpha*c + beta) as |G| grows: a policy that emits more
    coverage wins in the limit no matter how much wrong mass it carries, because wrong mass is
    discounted by alpha = 0.2 and never scales with |G|;
  * for any two policies there is a single crossover size G* above which the higher-coverage
    policy wins, and below which the cleaner one wins.

That makes the decision a *sensitivity* question with a computable answer: which of the plausible
scored-truth sizes are we actually in?  This script reports the crossovers, projects every
measured policy onto a range of plausible |G|, and states the verdict against the repository's
pre-registered rule.  The rule is evaluated CANDIDATE-WISE, against the best measured candidate
(a joint floor x width policy, ranked on the first ensemble's sweep):

  1. it beats the shipped policy on the new-fault-like population by > 0.01;
  2. every plausible scored-truth anchor favours it (a recorded crossover, not a boast);
  3. its CONTRAST reproduces on every further independently trained ensemble that was handed in
     with --second-sweep (absolute proxy DTI is not comparable across fold sets, so the record
     keeps each sweep's own reference policy and compares contrasts).

Session 13 added (3) as a LIST and added the cross-ensemble robustness ranking printed at the end:
the candidate is ranked by its WORST contrast across every sweep in the record, because a maximum
that exists on one field only is a field-specific maximum, not a policy.

THE FINDING THAT MATTERS MORE THAN THE WIDTH
--------------------------------------------
The sweep measured a constant-ones submission (fill the data footprint with 1.0) on population C:
DTI 0.0585, which beat the shipped skeleton's 0.0247 and, at that point, every swept shaping
candidate.  A constant map has no skill at all; it won because recall is worth four times
precision under this metric (alpha = 0.2, beta = 0.8) and because it pays its false-positive mass
in a currency the denominator discounts -- i.e. the shipped skeleton was not "conservative", it was
*under-emitting*.  Session 13 then swept explicit floors and found a policy with actual skill that
beats the constant map on the same population: floor 0.1 with NO widening scores 0.1365 there
(constant-ones 0.0585, shipped 0.0247), so a low floor is not the same kind of bet as filling the
footprint -- the candidate has to keep its contrast on further ensembles, which is what
condition 3 measures.  The uncomfortable ordering is kept here on purpose: it is what made the
floor axis, rather than the width axis, the thing worth measuring.

NOT A LEADERBOARD PREDICTION.  Population C is state-geological-survey surface mapping, not the
expert interpretation of GeoDAWN geophysics the prize scores; the numbers are for comparing
policies, and the projected |G| values are assumptions printed as assumptions.

USAGE
    python scripts/decide_emission_width.py --out data/evidence/emission_decision.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import json
from pathlib import Path

ALPHA, BETA = 0.2, 0.8
# The published leaderboard's best public score.  Unlike the previous hard-coded pin, this is read
# from the committed evidence data/evidence/independent_verification.json
# (competition_standing.top_dti) so the sentence it feeds cannot drift from the snapshot the site
# already renders.  The literal below is the last known value, used ONLY when the evidence file is
# absent (unit tests that synthesise evidence without the live record).
LEADERBOARD_TOP_FALLBACK = 0.1972
PX_KM = 0.1            # 100 m pixels

ROOT = Path(__file__).resolve().parents[1]
AOI_AREA_KM2 = 3292 * 3730 * PX_KM ** 2          # 122,791.6 km^2
GEODAWN_AREA_KM2 = 51857.0                       # USGS GeoDAWN data release, 10.5066/P93LGLVQ


def load(path: Path) -> dict:
    return json.loads(path.read_text())


INDEPENDENT_VERIFICATION = Path("data/evidence/independent_verification.json")


def _leaderboard_top() -> dict:
    """The published leaderboard's best public score, from the committed evidence.

    The previous implementation hard-coded a snapshot (0.1972, 2026-09-16) which went stale when
    the evidence was refreshed (0.2854, 2026-09-21); the site rendered the new number while this
    sentence kept the old one.  `ROOT` is the repository root (the script's own location), so a
    caller that runs from another cwd or writes ``--out`` somewhere else still resolves the
    evidence from the same committed file the site reads.
    """
    rec = dict(dti=LEADERBOARD_TOP_FALLBACK, observed_utc="NOT RECORDED (evidence absent)",
               source="fallback literal in scripts/decide_emission_width.py")
    p = ROOT / INDEPENDENT_VERIFICATION
    if p.exists():
        try:
            st = json.loads(p.read_text()).get("competition_standing") or {}
            rec = dict(dti=float(st.get("top_dti", LEADERBOARD_TOP_FALLBACK)),
                       observed_utc=st.get("observed_utc", "NOT RECORDED"),
                       source=str(INDEPENDENT_VERIFICATION))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return rec


def provided(res: dict) -> dict:
    """The score of the raster handed to --pred, under either spelling.

    Evidence written after 2026-09-16 calls it `as_provided` (the old name, `as_submitted`, called
    every --pred raster a submission, which was wrong in the shaping sweep); earlier evidence files
    still carry the old key, so read both.
    """
    return res.get("as_provided") or res["as_submitted"]


def policy_row(name: str, truth_px: int, res: dict, note: str = "") -> dict:
    """Turn a measured score into the two parameters that control its scaling."""
    tp, fp = float(res["TP_w"]), float(res["FP_w"])
    return {"policy": name, "measured_dti": round(float(res["dti"]), 6),
            "coverage_fraction": round(tp / float(truth_px), 6), "wrong_mass_FP_w": round(fp, 3),
            "emission_px": int(res.get("emission_px") or 0), "note": note}


def project(row: dict, truth_px: float) -> float:
    """DTI for a hypothetical scored truth of `truth_px`, from a measured (coverage, FP) pair."""
    c, f = row["coverage_fraction"], row["wrong_mass_FP_w"]
    tp = c * float(truth_px)
    return float(tp / (ALPHA * (tp + f) + BETA * float(truth_px) + 1e-12))


def crossover(p1: dict, p2: dict) -> dict:
    """Truth size at which two policies' projected curves cross (or None if they do not).

    Solving c1|G|/(a(c1|G|+F1)+b|G|) = c2|G|/(a(c2|G|+F2)+b|G|) for |G| gives
        G* = a*(c2*F1 - c1*F2) / (b*(c1 - c2))
    A positive G* means the lower-coverage policy is better below it and the higher-coverage one
    above it (when c1 < c2); a negative or undefined G* means one policy dominates everywhere.
    """
    a, b = ALPHA, BETA
    c1, f1 = p1["coverage_fraction"], p1["wrong_mass_FP_w"]
    c2, f2 = p2["coverage_fraction"], p2["wrong_mass_FP_w"]
    if abs(c1 - c2) < 1e-12:
        return {"between": [p1["policy"], p2["policy"]], "crossover_px": None,
                "reason": "identical coverage: the policy with less wrong mass wins everywhere"}
    g = a * (c2 * f1 - c1 * f2) / (b * (c1 - c2))
    out = {"between": [p1["policy"], p2["policy"]], "crossover_px": round(float(g), 1)}
    if g <= 0:
        winner = p1["policy"] if c1 > c2 else p2["policy"]
        out["reason"] = f"no crossing at a positive truth size: {winner} wins at every size"
    else:
        wider = p2["policy"] if c2 > c1 else p1["policy"]
        cleaner = p1["policy"] if c2 > c1 else p2["policy"]
        out["reason"] = (f"{cleaner} wins below {round(float(g), 1)} px "
                         f"({round(float(g) * PX_KM, 1)} km); {wider} wins above")
    return out


def width_gain_table(sweep: dict) -> dict:
    """Per-floor DTI contrast between the widest swept band and the pure skeleton.

    This is the FLOOR-CONTROLLED form of the width question, and the reason it is computed inside
    one sweep file rather than across sweeps: absolute proxy DTI differs between ensembles (each
    fold set produces a different probability field), so comparing ensemble 2's number with
    ensemble 1's number would confound the width with the model.  For every swept floor ``t0``
    that has both ``dilate=0`` and the widest ``dilate``, the contrast is taken at the SAME floor,
    so the sign and the size of the gain are attributable to the emission width alone.

    The repository's pre-registered acceptance rule for the width change is "same sign and
    > 0.01" - evaluated here as: every swept floor gains, and the mean gain exceeds 0.01.
    """
    # HARD rows only: a ramp candidate has the same support as the hard band of its width but
    # different values, so it is a different policy - and, critically, its presence in this table
    # would define "the widest band" by a row that is not a band width at all.
    rows = [r for r in sweep["results"]["shaping_sweep"] if r["thin"] and not r.get("soft")]
    widths = sorted({int(r["dilate"]) for r in rows})
    if not widths:
        return dict(widest_px=None, per_floor=[], n_floors=0, mean_gain=None, min_gain=None)
    widest = max(widths)
    per_floor = []
    for t in sorted({float(r["t0"]) for r in rows}):
        d0 = next((float(r["dti"]) for r in rows
                   if float(r["t0"]) == t and int(r["dilate"]) == 0), None)
        dw = next((float(r["dti"]) for r in rows
                   if float(r["t0"]) == t and int(r["dilate"]) == widest), None)
        if d0 is None or dw is None:
            continue
        per_floor.append(dict(t0=t, dti_width0=round(d0, 6), dti_widest=round(dw, 6),
                              gain=round(dw - d0, 6)))
    gains = [p["gain"] for p in per_floor]
    return dict(widest_px=widest, per_floor=per_floor, n_floors=len(gains),
                mean_gain=(round(sum(gains) / len(gains), 6) if gains else None),
                min_gain=(min(gains) if gains else None),
                max_gain=(max(gains) if gains else None),
                all_floors_positive=bool(gains) and all(g > 0 for g in gains))


def policy_reproduction(sweep: dict, t0: float, width: int) -> dict | None:
    """Look up the SAME hard policy in another ensemble's sweep and contrast it with THAT sweep's
    own reference policy.

    Absolute proxy DTI is not comparable across ensembles - each fold set produces a different
    probability field - so a candidate is only "reproduced" if the contrast against the reference
    policy of the same sweep keeps its sign and size.  This is the floor-change counterpart of
    ``width_gain_table``: the width table controls the floor and varies the width, this controls
    nothing and varies the whole policy, which is what a candidate that changes BOTH needs.
    """
    rows = [r for r in sweep["results"]["shaping_sweep"] if r["thin"] and not r.get("soft")]
    target = next((r for r in rows if float(r["t0"]) == float(t0) and int(r["dilate"]) == int(width)),
                  None)
    ref = (sweep["results"].get("sweep_verdict") or {}).get("current_policy_dti")
    if target is None or ref is None:
        return None
    return {"t0": float(t0), "width_px": int(width),
            "measured_dti": round(float(target["dti"]), 6),
            "reference_policy_dti": round(float(ref), 6),
            "contrast_vs_reference": round(float(target["dti"]) - float(ref), 6),
            "reference_is": "sweep_verdict.current_policy_dti of that same sweep"}


def cross_ensemble_ranking(sweeps: list[tuple[str, dict]], limit: int = 10) -> dict:
    """Rank every hard (floor, width) candidate by its WORST contrast taken over every sweep.

    ``best_measured_candidate`` is the argmax of the FIRST ensemble's sweep, which is a maximum of
    one field.  A policy is only worth shipping if it is not: this ranks the candidates that exist
    in EVERY sweep by the smallest contrast they earn against each sweep's own reference policy, so
    the record shows whether the shipped candidate is the robust one or merely the first field's
    favourite.  (It is also the cheapest check of the pre-registered rule: a candidate whose worst
    contrast is <= 0.01 cannot pass condition 3 on the sweeps supplied.)
    """
    per: list[tuple[str, float, dict]] = []
    common: set | None = None
    for name, sw in sweeps:
        rows = [r for r in sw["results"]["shaping_sweep"]
                if r["thin"] and not r.get("soft") and float(r.get("gamma", 1.0)) == 1.0]
        ref = (sw["results"].get("sweep_verdict") or {}).get("current_policy_dti")
        if ref is None:
            continue
        table = {(round(float(r["t0"]), 6), int(r["dilate"])): r for r in rows}
        per.append((name, float(ref), table))
        common = set(table) if common is None else (common & set(table))
    if not per or not common:
        return dict(n_ensembles=len(per), n_candidates=0, ranked=[],
                    meaning="no candidate is present in every supplied sweep")
    ranked = []
    for t0, w in sorted(common):
        contrasts = {name: table[(t0, w)]["dti"] - ref for name, ref, table in per}
        ranked.append({"floor_t0": t0, "width_px": w,
                       "contrast_by_ensemble": {k: round(v, 6) for k, v in contrasts.items()},
                       "worst_contrast": round(min(contrasts.values()), 6),
                       "mean_contrast": round(sum(contrasts.values()) / len(contrasts), 6),
                       "dti_first_ensemble": round(per[0][2][(t0, w)]["dti"], 6),
                       "emission_px_first_ensemble": int(per[0][2][(t0, w)].get("emission_px", -1))})
    ranked.sort(key=lambda r: (-r["worst_contrast"], -r["mean_contrast"]))
    return {"n_ensembles": len(per), "n_candidates": len(ranked), "ranked": ranked[:limit],
            "meaning": ("ranked by the worst contrast against each sweep's own reference policy; "
                        "a candidate that only wins on one field is a field-specific maximum")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-eval", default="data/evidence/proxy/eval_submission.json")
    ap.add_argument("--reblend-eval", default="data/evidence/proxy/eval_reblend_submission.json")
    ap.add_argument("--sweep", default="data/evidence/proxy/eval_sweep.json")
    ap.add_argument("--miss-distance", default="data/evidence/proxy/miss_distance-ensemble1.json",
                    help="scripts/measure_miss_distance.py evidence: the exact metric as a function "
                         "of the emitted band width, plus the localization/detection split")
    ap.add_argument("--second-sweep", default=None, action="append",
                    help="sweep of a SECOND, independently trained ensemble (same recipe, different "
                         "seed/folds), or a COMMA-SEPARATED LIST of them; the flag may be REPEATED and "
                         "the lists are concatenated (a repeated flag that silently kept only the last "
                         "list would let condition 3 pass on LESS evidence than the command line asked "
                         "for). Evaluated as condition 3: "
                         "the policy this record proposes must keep the sign and size of its contrast "
                         "against each sweep's own reference policy. Written by the proxy-eval "
                         "workflow as data/evidence/proxy/eval_sweep-<label>.json")
    ap.add_argument("--in-domain",
                    default="data/evidence/runs/35042805806-experiment/blend_report.json")
    ap.add_argument("--out", default="data/evidence/emission_decision.json")
    a = ap.parse_args()

    leaderboard_top = _leaderboard_top()

    assert abs((ALPHA + BETA) - 1.0) < 1e-12, "the projection identity requires alpha + beta == 1"
    # A reproduction is per ENSEMBLE, so the flag takes a list: two independent sweeps that agree
    # are a stronger claim than one, and the record has to be able to hold both.  The flag is
    # repeatable as well as comma-joined, and BOTH forms are concatenated - a repeated flag that
    # quietly kept only the last list would shrink condition 3 without any visible symptom.
    second_sweeps = [q.strip() for part in (a.second_sweep or [])
                     for q in part.split(",") if q.strip()]

    # ------------------------------------------------------------------ measured policies (C)
    rows: list[dict] = []
    ev = load(Path(a.proxy_eval))
    gp = int(ev["truth"]["px"])
    rows.append(policy_row("shipped_skeleton_dilate0", gp, provided(ev["results"]),
                           "the committed submission: t0=0.470, thin, width 0"))
    blanks = ev["results"]["baselines"]
    rows.append(policy_row("baseline_blanket_ones", gp, blanks["blanket_ones"],
                           "fill the data footprint with 1.0 - no skill, maximum recall"))
    rows.append(policy_row("baseline_catalogue_plus_submission", gp,
                           blanks["catalogue_plus_submission"], "the catalogue unioned with ours"))
    if "blanket_ones_whole_grid" in blanks:
        rows.append(policy_row("baseline_blanket_ones_whole_grid", gp,
                               blanks["blanket_ones_whole_grid"],
                               "illegal submission (must be NaN outside the footprint), "
                               "recorded for scale"))
    rb = load(Path(a.reblend_eval))
    rows.append(policy_row("reblend_hard_39517px", gp, provided(rb["results"]),
                           "the fold pair re-blended without the pre-shaping floor"))
    sw = load(Path(a.sweep))
    swept: list[tuple[float, int, dict]] = []
    ramp: list[dict] = []
    for row in sw["results"]["shaping_sweep"]:
        if not row["thin"]:
            continue
        # HARD candidates only in the width machinery below: a ramp candidate has its own support
        # and its own emitted values, so f"width {d} px" does not identify it, and mixing the two
        # would put a policy into the width curve that the width curve cannot describe.
        if row.get("soft"):
            ramp.append(row)
            continue
        swept.append((float(row["t0"]), int(row["dilate"]), row))
        if row["dilate"] not in (0, 3, 6) or row["t0"] not in (0.0, 0.0432675):
            continue
        rows.append(policy_row(f"sweep_t0_{row['t0']:g}_width{row['dilate']}px", gp, row,
                               f"floor {row['t0']:g}, thinning on, width {row['dilate']} px"))
    if not swept:
        raise SystemExit("the sweep evidence carries no thinned rows - nothing to decide from")
    # The widest candidate actually measured, whatever the grid was: the workflow's dilate grid is
    # an input, so hard-coding "6 px" here would silently compare against nothing when it changes.
    widest = max(d for _, d, _ in swept)
    # Among the widest band, the floor that scores best on this population is the fair wide
    # candidate (picking the highest floor would pick the collapse case, which scores ~0).
    # The floor axis is a candidate axis in its own right.  The session-12 extended grid showed the
    # best HARD candidate can be a floor change rather than a width change, and a ranking that
    # admitted only floors {0, 0.0433} would then decide a width question while silently excluding
    # what the search actually found.  Every swept floor contributes its single best width.
    best_per_floor: dict[float, dict] = {}
    for t, d, r in swept:
        if t not in best_per_floor or float(r["dti"]) > float(best_per_floor[t]["dti"]):
            best_per_floor[t] = r
    existing_names = {r["policy"] for r in rows}
    for t, r in sorted(best_per_floor.items()):
        name = f"sweep_best_t0_{t:g}_width{int(r['dilate'])}px"
        if name in existing_names:
            continue
        rows.append(policy_row(name, gp, r, f"best hard candidate at floor {t:g} "
                                            f"(thinning on): width {int(r['dilate'])} px"))
    wide_t0, _, wide_row = max([(t, d, r) for t, d, r in swept if d == widest],
                               key=lambda x: (x[2]["dti"], -x[0]))
    wide_name = f"sweep_t0_{wide_t0:g}_width{widest}px"
    if wide_name not in {r["policy"] for r in rows}:
        rows.append(policy_row(wide_name, gp, wide_row,
                               f"floor {wide_t0:g}, thinning on, width {widest} px (widest swept)"))
    print(f"widest swept emission: {widest} px at floor {wide_t0:g}")
    rows.append(policy_row("ensemble_soft_map", gp, provided(sw["results"]),
                           "the raw pre-shaping ensemble mean (not a legal submission: values "
                           "outside the footprint are not NaN)"))
    # Ramp emission (session 11): identical support to the hard band of the same width, values
    # decaying to 0 at the band edge.  The best ramp per width is a policy in the same ranking as
    # the hard candidates, so a decision that ignores it would be a decision made on a subset.
    ramp_best: dict[int, dict] = {}
    for row in ramp:
        d = int(row["dilate"])
        if d not in ramp_best or float(row["dti"]) > float(ramp_best[d]["dti"]):
            ramp_best[d] = row
    for d, row in sorted(ramp_best.items()):
        rows.append(policy_row(
            f"sweep_ramp_t0_{float(row['t0']):g}_width{d}px_gamma{float(row.get('gamma', 1.0)):g}",
            gp, row,
            f"ramp values (gamma {float(row.get('gamma', 1.0)):g}) on the width-{d} px support, "
            f"floor {float(row['t0']):g}, thinning on"))
    if ramp:
        print(f"ramp emission candidates scored: {len(ramp)}; best per width: "
              f"{ {d: round(float(r['dti']), 6) for d, r in sorted(ramp_best.items())} }")

    # ------------------------------------------------------------------ plausible |G|
    # Two anchors, both stated as assumptions.  1) Density scaling: the proxy catalogue has a
    # measured fault density over this AOI; the scored faults are drawn from the GeoDAWN blocks,
    # which cover a known area of it.  2) The catalogue itself (labels.tif) is the other known
    # population inside the same AOI.
    density_km_per_km2 = (gp * PX_KM) / AOI_AREA_KM2
    plausible = {
        "assumption_density_km_per_km2": round(density_km_per_km2, 5),
        "assumption_geodawn_area_km2": GEODAWN_AREA_KM2,
        "assumption_aoi_area_km2": round(AOI_AREA_KM2, 1),
        "anchors": [
            {"name": "same density over the GeoDAWN blocks",
             "truth_px": int(round(density_km_per_km2 * GEODAWN_AREA_KM2 / PX_KM)),
             "why": "the scored faults come from the blocks the geophysics was flown over"},
            {"name": "same density over the whole AOI",
             "truth_px": gp, "why": "the measured population itself"},
            {"name": "the public catalogue (labels.tif) alone",
             "truth_px": 60988, "why": "known: 6,099 km, the faults the model trained on"},
        ],
    }
    sizes = sorted({500, 1000, 2500, 5000, 10000, 20000, 30000, 50000, 61664, 100000, 250000}
                   | {x["truth_px"] for x in plausible["anchors"]})

    # ------------------------------------------------------------------ in-domain (A), for contrast
    idr = load(Path(a.in_domain))
    ct = idr["shaping"]["calibration_table"]
    in_domain = [
        {"t0": r["t0"], "thin": r["thin"], "dilate": r["dilate"], "mean_heldout_dti": r["mean_dti"]}
        for r in ct if r.get("t0") in (0.0, 0.469674) and r.get("thin") and
        r.get("dilate") in (0, 1, 3, 6)]
    in_domain_rows = sorted(in_domain, key=lambda r: (r["t0"], r["dilate"]))

    # ------------------------------------------------------------------ verdict
    # The rule is evaluated candidate-wise, so the best measured sweep candidate has to be known
    # before any verdict text is written.
    best_cand = max((r for r in rows if r["policy"].startswith("sweep_")),
                    key=lambda r: r["measured_dti"])
    shipped = next(r for r in rows if r["policy"] == "shipped_skeleton_dilate0")
    wide = next(r for r in rows if r["policy"] == wide_name)
    blanket = next(r for r in rows if r["policy"] == "baseline_blanket_ones")
    v = {
        "pre_registered_rule": sw["results"]["sweep_verdict"]["acceptance_rule"],
        "proxy_population_ranking": [r["policy"] for r in
                                     sorted(rows, key=lambda r: -r["measured_dti"])],
        "wide_vs_shipped": crossover(shipped, wide),
        "blanket_vs_shipped": crossover(shipped, blanket),
        "blanket_vs_wide": crossover(wide, blanket),
        "in_domain_contrast": ("on the population the model trained on, the same widening costs "
                               "0.1903 -> 0.0908 (skeleton -> 6 px): the sign of the effect flips "
                               "with the population, which is why neither number alone can decide it"),
        "conditions": [],
        "conclusion": "",
        # Computed, not typed: the blanket baseline's own number moved between evidence versions
        # (0.0585 over the footprint-clipped support, 0.0249 over the whole footprint), and a
        # hard-coded ratio then contradicted the table beside it.
        "top_priority": ("detection, not the emission policy, is the binding constraint: the best "
                         "measured candidate (%.4f, %s) %s the constant-ones baseline (%.4f) on the "
                         "new-fault-like population it was measured on, while the leaderboard's top "
                         "score (%.4f, read %s) is %.1fx that baseline; the skeleton the candidate "
                         "replaces scores %.4f there"
                         % (best_cand["measured_dti"], best_cand["policy"],
                            "beats" if best_cand["measured_dti"] > blanket["measured_dti"]
                            else "trails",
                            blanket["measured_dti"], leaderboard_top["dti"],
                            leaderboard_top["observed_utc"],
                            leaderboard_top["dti"] / max(blanket["measured_dti"], 1e-9),
                            shipped["measured_dti"])),
    }
    # Sweep policies only: the baselines (blanket, catalogue copy) are ranked in the same table but
    # they are controls, not candidates the shaping pipeline can produce (best_cand is computed
    # above, before the verdict text that names it).
    v["best_measured_candidate"] = {
        "policy": best_cand["policy"],
        "measured_dti": best_cand["measured_dti"],
        "note": best_cand["note"],
        "contrast_vs_shipped": round(best_cand["measured_dti"] - shipped["measured_dti"], 6),
        "crosses_shipped": crossover(shipped, best_cand),
        "reproduced_on_second_ensemble": None,
        "why_this_is_recorded": ("the pre-registered rule's candidate set has to be stated, not "
                                 "assumed: the best measured candidate on this population is "
                                 "ranked here, so a verdict cannot be reached with it excluded"),
    }
    m_best = re.match(r"sweep_best_t0_([0-9.]+)_width([0-9]+)px$", best_cand["policy"]) or \
        re.match(r"sweep_t0_([0-9.]+)_width([0-9]+)px$", best_cand["policy"])
    if m_best:
        cand_floor = float(m_best.group(1))
        # The floor and the width are alternative policies, not additive knobs: at the best floor the
        # best width may be 0 px.  Recording the whole width curve of the winning floor makes that
        # visible in the record instead of leaving it to a reader to reconstruct from the sweep file.
        v["best_measured_candidate"]["widths_at_that_floor"] = {
            str(int(r["dilate"])): round(float(r["dti"]), 6)
            for t, d, r in sorted(swept, key=lambda x: x[1]) if t == cand_floor}
        w_at = v["best_measured_candidate"]["widths_at_that_floor"]
        v["best_measured_candidate"]["width_optimum_at_that_floor_px"] = (
            int(max(w_at, key=lambda k: w_at[k])) if w_at else None)
    # ---------------------------------------------------------------- reproduction (condition 3)
    # The candidate this record proposes is a JOINT (floor, width) policy, so condition 3 asks about
    # that policy - not about the width axis at a fixed floor.  Each sweep supplies its own reference
    # policy, because absolute proxy DTI is not comparable across ensembles (different fold sets
    # produce different fields); what must survive is the CONTRAST.
    repros: list[dict] = []
    for path_str in second_sweeps:
        path = Path(path_str)
        if not path.exists():
            continue
        rep = policy_reproduction(load(path), float(m_best.group(1)), int(m_best.group(2))) \
            if m_best else None
        if rep is not None:
            rep["sweep_file"] = path_str
            rep["reproduced"] = bool(rep["measured_dti"] > rep["reference_policy_dti"]
                                     and rep["contrast_vs_reference"] > 0.01)
            repros.append(rep)
    if repros:
        v["best_measured_candidate"]["reproduced_on_second_ensemble"] = repros[0]
        v["best_measured_candidate"]["reproduction_across_ensembles"] = repros

    # ---------------------------------------------------------------- cross-ensemble robustness
    # The candidate was picked as the argmax of ONE field.  Before the verdict is stated, every
    # candidate that exists in every supplied sweep is ranked by its worst contrast, and the record
    # carries both the ranking and where the shipped candidate sits in it.  A policy that only wins
    # on the first ensemble would be reported here, loudly, instead of shipping as a maximum of one
    # field - which is exactly what a single-ensemble sweep cannot see.
    all_sweeps = [("eval_sweep.json", sw)]
    for path_str in second_sweeps:
        if Path(path_str).exists():
            all_sweeps.append((path_str, load(Path(path_str))))
    if len(all_sweeps) > 1:
        rank = cross_ensemble_ranking(all_sweeps)
        cand_t0 = float(m_best.group(1)) if m_best else None
        cand_w = int(m_best.group(2)) if m_best else None
        cand_key = (round(cand_t0, 6), cand_w) if m_best else None
        pos = next((i + 1 for i, r in enumerate(rank["ranked"])
                    if (r["floor_t0"], r["width_px"]) == cand_key), None)
        rank["shipped_candidate"] = (f"floor {cand_t0:g}, width {cand_w} px "
                                     f"ranks {pos} of {rank['n_candidates']}"
                                     if pos else "the shipped candidate is not in the ranking")
        rank["shipped_candidate_rank"] = pos
        if pos and pos > 1:
            best_of = rank["ranked"][0]
            rank["warning"] = (f"the shipped candidate is NOT the worst-case best: floor "
                               f"{best_of['floor_t0']}g, width {best_of['width_px']} px has the "
                               f"higher worst-case contrast ({best_of['worst_contrast']:+.4f} vs "
                               f"{rank['ranked'][pos - 1]['worst_contrast']:+.4f})")
        v["robustness_across_ensembles"] = rank

    # ---------------------------------------------------------------- conditions, candidate-wise
    gain = round(best_cand["measured_dti"] - shipped["measured_dti"], 6)
    v["conditions"].append({
        "condition": "the best measured candidate beats the shipped policy on the new-fault-like "
                     "population by > 0.01",
        "required": "> 0.01 measured proxy DTI", "measured": gain, "passes": bool(gain > 0.01)})
    cw = crossover(shipped, best_cand)["crossover_px"]
    anchor_rows = [(x["name"], project(shipped, x["truth_px"]), project(best_cand, x["truth_px"]))
                   for x in plausible["anchors"]]
    wins = sum(1 for _, s_, w in anchor_rows if w > s_)
    v["anchor_projection"] = [
        {"anchor": n, "truth_px": int(next(x["truth_px"] for x in plausible["anchors"]
                                           if x["name"] == n)),
         "shipped_dti": round(s_, 4), "candidate_dti": round(w, 4), "candidate_wins": bool(w > s_)}
        for n, s_, w in anchor_rows]
    if cw is None or cw <= 0:
        v["conditions"].append({"condition": "dominates the shipped policy across plausible "
                                             "scored-truth sizes", "required": True,
                                "measured": "no crossing: it wins everywhere", "passes": True})
    else:
        v["conditions"].append({
            "condition": "wins at the plausible scored-truth sizes (not only in the large-|G| limit)",
            "required": "every anchor favours the candidate",
            "measured": f"{wins}/{len(anchor_rows)} anchors favour it; crossover {cw:,.0f} px "
                        f"({cw * PX_KM:,.0f} km)",
            "passes": bool(wins == len(anchor_rows))})
    reproduced = bool(repros) and all(r["reproduced"] for r in repros)
    v["conditions"].append({
        "condition": "reproduced on a second (and every further) independently trained ensemble",
        "required": "same sign and > 0.01 contrast against each sweep's own reference policy",
        "measured": (
            "; ".join(f"{r['sweep_file']}: DTI {r['measured_dti']:.4f} vs its own reference "
                      f"{r['reference_policy_dti']:.4f} -> {r['contrast_vs_reference']:+.4f} "
                      f"({'reproduced' if r['reproduced'] else 'NOT reproduced'})" for r in repros)
            if repros else
            "NOT MEASURED: needs a second, independently trained ensemble sweep "
            "(see SUGGESTIONS.md)"),
        "passes": reproduced})

    # The width axis stays measured and recorded - it is how the record shows that "lower the floor"
    # and "widen the band" are alternatives rather than two knobs of one policy.
    v["width_axis"] = {"first_ensemble": width_gain_table(sw)}
    for path_str in second_sweeps:
        if Path(path_str).exists():
            v["width_axis"].setdefault("by_ensemble", {})[path_str] = \
                width_gain_table(load(Path(path_str)))
    v["width_gain_first_ensemble"] = v["width_axis"]["first_ensemble"]
    if "by_ensemble" in v["width_axis"]:
        first_key = next(iter(v["width_axis"]["by_ensemble"]))
        v["width_gain_second_ensemble"] = v["width_axis"]["by_ensemble"][first_key]

    # ---------------------------------------------------------------- conclusion (derived, never typed)
    cand_txt = (f"{best_cand['policy']} (floor {float(m_best.group(1)):g}, thin, "
                f"width {int(m_best.group(2))} px)") if m_best else best_cand["policy"]
    repro_txt = ", ".join(f"{r['sweep_file']}: {r['contrast_vs_reference']:+.4f}"
                           for r in repros)
    cond1 = bool(gain > 0.01)
    cond2 = bool(wins == len(anchor_rows))
    cond3 = reproduced
    failed = [n for n, ok in (("1", cond1), ("2", cond2), ("3", cond3)) if not ok]
    if not failed:
        v["conclusion"] = (
            f"SHIP the measured policy: {cand_txt} beats the shipped policy on the new-fault-like "
            f"population by {gain:+.4f} (condition 1), at {wins}/{len(anchor_rows)} plausible "
            f"scored-truth anchors with a crossover of {cw:,.0f} px ({cw * PX_KM:,.0f} km) "
            f"(condition 2), and the contrast reproduces on {len(repros)} independent ensemble(s) "
            f"({repro_txt}) (condition 3). Ship it through the re-blend workflow's "
            f"SHAPING_T0/SHAPING_DILATE, which is the only path that writes an adopted policy.")
    elif not repros:
        v["conclusion"] = (
            f"measured, not yet reproduced: {cand_txt} beats the shipped policy on the "
            f"new-fault-like population by {gain:+.4f} (condition 1) and at {wins}/"
            f"{len(anchor_rows)} plausible scored-truth anchors (condition 2), while a scored truth "
            f"below {(cw or 0):,.0f} px ({(cw or 0) * PX_KM:,.0f} km) would still favour the shipped "
            f"policy. Condition 3 - reproduction on a second, independently trained ensemble - is "
            f"unmet, so the shipped default is unchanged and this file is the record of why")
    else:
        bad = [r for r in repros if not r["reproduced"]]
        v["conclusion"] = (
            f"measured, not shipped: {cand_txt} was ranked first on this population, but "
            f"condition(s) {'/'.join(failed)} fail - "
            + (f"contrast vs the shipped policy {gain:+.4f} (condition 1); " if not cond1 else "")
            + (f"{wins}/{len(anchor_rows)} plausible scored-truth anchors favour it "
               f"(condition 2); " if not cond2 else "")
            + (f"{len(bad)} of {len(repros)} independent ensemble(s) do not reproduce the contrast "
               f"({'; '.join(r['sweep_file'] for r in bad)}, condition 3). "
               if not cond3 else "")
            + "No policy change is shipped, and this file is the record of why")

    # ------------------------------------------------------------------ measured width curve
    # The sweep searches floor x width on the ensemble probability map; this is the independent,
    # floor-free measurement on the SHIPPED hard submission (scripts/measure_miss_distance.py):
    # dilate the emitted set by k px and score with the same metric.  It answers what the sweep
    # could not - how far the misses actually are - and gives the projected optimum width per
    # assumed scored-truth size.
    md = None
    if a.miss_distance and Path(a.miss_distance).exists():
        m = load(Path(a.miss_distance))
        md = {"source": a.miss_distance,
              "truth_km": m.get("truth_km"), "emitted_km": m.get("emitted_km"),
              "miss_distance_percentiles_px": m["miss_geometry"]["percentiles_px"],
              "detection_failure_share_beyond_12px": m["verdict"]["detection_failure_share"],
              "best_width_px": m["verdict"]["best_width_px"],
              "gain_from_widening": m["verdict"]["gain_from_widening"],
              "widening_starts_winning_above_px": m.get("widening_starts_winning_above_px"),
              "projection": m.get("projection_over_scored_truth_size")}

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/decide_emission_width.py",
        "purpose": ("reconcile the three disagreeing emission-width measurements and decide the "
                    "shipped shaping policy on the record, using the metric's own scaling in |G|"),
        "inputs": {"proxy_eval": a.proxy_eval, "reblend_eval": a.reblend_eval, "sweep": a.sweep,
                   "second_sweep": ",".join(second_sweeps) or None,
                   "second_sweeps": second_sweeps,
                   "miss_distance": a.miss_distance,
                   "in_domain": a.in_domain,
                   "metric": {"alpha": ALPHA, "beta": BETA, "R_pixels": 3, "R_meters": 300}},
        "proxy_truth_px": gp,
        "policies": rows,
        "width_vs_scored_truth_size": md,
        "crossovers": {"wide_vs_shipped": v["wide_vs_shipped"],
                       "candidate_vs_shipped": crossover(shipped, best_cand),
                       "widest_swept_px": widest,
                       "blanket_vs_shipped": v["blanket_vs_shipped"],
                       "blanket_vs_wide": v["blanket_vs_wide"]},
        "projection_table": [
            {"truth_px": int(g), "truth_km": round(g * PX_KM, 1),
             **{r["policy"]: round(project(r, g), 5) for r in rows}} for g in sizes],
        "plausible_scored_truth": plausible,
        "in_domain_measurement": {"rows": in_domain_rows,
                                  "measures": "held-out crops of the public catalogue",
                                  "caveat": "upper bound on plumbing, not on skill"},
        "verdict": v,
        "caveats": [
            "The projection holds coverage and wrong mass fixed while |G| changes; a model that "
            "would be retrained for a different-sized truth would not hold either.",
            "Population C is USGS state-map surface mapping, not the expert interpretation of "
            "GeoDAWN geophysics that the prize scores. Policy comparisons only.",
            "The scored truth size is unknowable from here; the anchors are assumptions printed "
            "as assumptions, not estimates with confidence intervals.",
        ],
    }
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"proxy truth |G_proxy| = {gp:,} px ({gp * PX_KM:,.0f} km)\n")
    print("measured policies on the new-fault-like population (population C):")
    for r in sorted(rows, key=lambda r: -r["measured_dti"]):
        print("  %-34s DTI %.4f  coverage %6.2f%%  wrong mass FP_w %12.1f  emitted %9d px"
              % (r["policy"], r["measured_dti"], 100 * r["coverage_fraction"],
                 r["wrong_mass_FP_w"], r["emission_px"]))
    print("\ncrossovers (the truth size at which the wider/higher-coverage policy takes over):")
    for k, c in out["crossovers"].items():
        # `crossovers` also carries scalar context (`widest_swept_px`); subscripting every value
        # as a dict crashed this print AFTER the JSON was written (found 2026-09-17: the step
        # failed while its evidence landed, so a red step looked like a completed reconciliation).
        print("  %-22s %s" % (k, c["reason"] if isinstance(c, dict) else c))
    print("\nprojected DTI at the plausible anchors (best measured candidate vs the shipped policy):")
    for x in v.get("anchor_projection", []):
        print("  %-40s shipped %.4f  candidate %.4f  %s"
              % (x["anchor"], x["shipped_dti"], x["candidate_dti"],
                 "CANDIDATE WINS" if x["candidate_wins"] else "shipped wins"))
    bc = v["best_measured_candidate"]
    print("\nbest measured candidate on this population: %s" % bc["policy"])
    print("  DTI %.4f (%+.4f vs the shipped policy); %s"
          % (bc["measured_dti"], bc["contrast_vs_shipped"], bc["note"]))
    for r2 in bc.get("reproduction_across_ensembles", []):
        print("  reproduction on %s: DTI %.4f vs its own reference %.4f -> %+.4f  %s"
              % (r2["sweep_file"], r2["measured_dti"], r2["reference_policy_dti"],
                 r2["contrast_vs_reference"], "REPRODUCED" if r2["reproduced"] else "NOT reproduced"))
    rob = v.get("robustness_across_ensembles")
    if rob and rob.get("ranked"):
        print("\ncross-ensemble robustness (worst contrast against each sweep's own reference):")
        for i, r in enumerate(rob["ranked"][:6], 1):
            print("  %2d. floor %-8s width %2d px  worst %+.4f  mean %+.4f  per-ensemble %s"
                  % (i, r["floor_t0"], r["width_px"], r["worst_contrast"], r["mean_contrast"],
                     ", ".join(f"{k.split('/')[-1]}: {v2:+.4f}"
                               for k, v2 in r["contrast_by_ensemble"].items())))
        print("  %s" % rob.get("shipped_candidate"))
        if rob.get("warning"):
            print("  WARNING: %s" % rob["warning"])
    print("\nconclusion: %s" % v["conclusion"])
    print("top priority: %s" % v["top_priority"])
    print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
