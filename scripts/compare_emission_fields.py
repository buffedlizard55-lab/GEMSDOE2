#!/usr/bin/env python3
"""Compare the ENSEMBLE FIELDS at matched emission support - the axis no rule currently covers.

WHAT IS ALREADY DECIDED, AND WHAT IS NOT
----------------------------------------
`data/evidence/emission_decision.json` selects the shaping POLICY (floor x thinning x emission
width) by a pre-registered rule: the candidate must beat each field's own reference policy by
> 0.01 proxy DTI, and the contrast must reproduce on every independently trained ensemble - ranked
by WORST-CASE contrast across the four committed sweeps.  Floor 0.1 / thin / width 0 px wins that
ranking and is what `data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif` ships.

That rule ranks POLICIES.  It says nothing about which FIELD the policy is applied to, and the
field is not a free parameter in disguise:

    a floor is a threshold on a field whose SCALE CHANGES WITH THE NUMBER OF AVERAGED FOLDS.

Averaging more independent folds pulls probability mass toward the mean, so fewer pixels clear a
fixed floor.  Measured from the four committed sweeps at the SAME adopted policy (t0=0.1, thin,
width 0):

    ensemble 1        (6 folds)   464,736 px   proxy DTI 0.136452
    ensembles 1+2    (11 folds)   172,974 px   proxy DTI 0.099859   <- the shipped artifact
    ensembles 1+2+3  (16 folds)   144,738 px   proxy DTI 0.084962
    ensemble 2        (5 folds)   131,117 px   proxy DTI 0.077671

So "the same policy" is 3.6x more emission on one field than another, and the DTI ranking follows
the support, not the field's quality.  Comparing fields at a fixed floor therefore measures the
floor, not the field - which is also why policy condition 3 ("reproduced on every ensemble") is
confounded: floor 0.05 looks irreproducible partly because ensemble 1's field carries a different
value distribution, not because the policy is unstable.

WHAT THIS SCRIPT DOES
---------------------
A scale-free comparison: for every committed sweep it finds the HARD candidate whose emission
support is closest to the shipped support (172,974 px - cross-checked against the block-stratified
report when present) and reports that candidate's DTI.  At matched support the floor has been
re-tuned per field, so the remaining difference between fields is the field.

It then states, as a DERIVED verdict rather than prose, whether the surrogate prefers a different
field than the one shipped, and records that the field axis is not covered by any pre-registered
rule - so acting on this needs either a rule committed BEFORE the next re-blend, or a scored
submission.  Nothing here rewrites the adopted policy.

USAGE
    python scripts/compare_emission_fields.py
    python scripts/compare_emission_fields.py --out data/evidence/emission_field_axis.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ADOPTED = dict(t0=0.1, thin=True, dilate=0, soft=False)
ADOPTED_LABEL = "floor 0.1 / thin / width 0 px"
SHIPPED_FIELD_SWEEP = "eval_sweep-mean12.json"     # the field the shipped artifact was blended from
DEFAULT_SUPPORT_PX = 172_974                       # measured: block-stratified report + mean12 sweep
# A floor grid is discrete, so "the candidate at exactly the shipped support" does not exist for
# most fields.  Three nested windows are reported and the ranking must hold in ALL of them before
# the verdict calls a field better: with a single window the representative candidate can flip on a
# knife edge (measured 2026-09-18: at +-15 % the ens123 field is represented by floor 0.8 / 12 px,
# at +-25 % by floor 0.1 / 0 px - different candidates, different DTI, same field).
SUPPORT_WINDOWS = (0.15, 0.25, 0.40)
PRIMARY_WINDOW = 0.25


def is_hard(row: dict) -> bool:
    """A hard-band candidate on the swept grid (not a ramp, not an unthinned variant)."""
    return bool(row.get("thin", False)) and not bool(row.get("soft", False))


def is_adopted(row: dict) -> bool:
    return (abs(float(row.get("t0", -1)) - ADOPTED["t0"]) < 1e-9
            and int(row.get("dilate", -1)) == ADOPTED["dilate"] and is_hard(row))


def load_sweeps(proxy_dir: Path) -> list[dict]:
    """Every committed sweep, joined with the run ids that produced its field."""
    out = []
    for path in sorted(glob.glob(str(proxy_dir / "eval_sweep*.json"))):
        p = Path(path)
        try:
            doc = json.loads(p.read_text())
        except Exception as exc:                                          # noqa: BLE001
            print(f"::warning:: {p.name} is unreadable ({type(exc).__name__}: {exc}) - skipped")
            continue
        rows = ((doc.get("results") or {}).get("shaping_sweep")) or []
        hard = [r for r in rows if is_hard(r)]
        if not hard:
            print(f"::warning:: {p.name} has no hard candidates - skipped")
            continue
        stem = p.name[len("eval_sweep"):-len(".json")].lstrip("-")
        src_path = proxy_dir / f"sweep_source{('-' + stem) if stem else ''}.json"
        src = json.loads(src_path.read_text()) if src_path.exists() else {}
        out.append(dict(file=p.name, path=str(p), label=(stem or "ensemble1"),
                        run_ids=src.get("swept_run_id"),
                        pred_sha256=(doc.get("inputs") or {}).get("pred_sha256"),
                        truth=(doc.get("truth") or {}),
                        rows=hard, n_rows=len(hard), doc=doc))
    if not out:
        raise SystemExit(f"no committed sweeps with hard candidates under {proxy_dir} - run "
                         ".github/workflows/proxy-eval.yml first (it re-blends the saved fold "
                         "artifacts and sweeps the field)")
    return out


def shipped_support(sweeps: list[dict], override: Optional[int],
                    block_report: Optional[Path]) -> dict:
    """The support the shipped artifact actually writes, cross-checked two ways."""
    sources = {}
    if override:
        sources["cli_override"] = int(override)
    mean12 = next((s for s in sweeps if s["file"] == SHIPPED_FIELD_SWEEP), None)
    if mean12:
        row = next((r for r in mean12["rows"] if is_adopted(r)), None)
        if row:
            sources["adopted_policy_row_of_" + SHIPPED_FIELD_SWEEP] = int(row["emission_px"])
    if block_report and block_report.exists():
        try:
            rep = json.loads(block_report.read_text())
            ref = rep["verdict"]["reference_candidate"]
            cand = next(c for c in rep["populations"]["proxy_only"]["candidates"]
                        if c["label"] == ref)
            sources["block_stratified_report"] = int(cand["pred_px"])
        except Exception as exc:                                          # noqa: BLE001
            sources["block_stratified_report"] = f"unreadable: {type(exc).__name__}"
    numeric = [v for v in sources.values() if isinstance(v, int)]
    agreed = len(set(numeric)) <= 1
    support = numeric[0] if numeric else DEFAULT_SUPPORT_PX
    return dict(support_px=support, sources=sources, sources_agree=agreed,
                fallback_used=(not numeric),
                note=("the shipped artifact is a hard band, so its written support equals the "
                      "adopted-policy row of the sweep that measured the field it was blended "
                      "from; the block-stratified report counts the same pixels independently"))


def _row_brief(r: dict, support: int) -> dict:
    return dict(t0=r["t0"], width_px=int(r["dilate"]), dti=float(r["dti"]),
                emission_px=int(r["emission_px"]),
                support_deviation=round((int(r["emission_px"]) - support) / support, 6),
                TP_w=r["TP_w"], FP_w=r["FP_w"], FN_w=r["FN_w"])


def matched_support_rows(rows: list[dict], support: int) -> dict:
    """Support-matched candidates for one field: best-in-window (primary) + closest (secondary).

    `best_in_window[tol]` is the highest-DTI hard candidate whose support is within +-tol of the
    shipped support.  The same rule is applied to every field, so the max-over-window bias is
    common to all of them; reporting three nested windows shows whether the ranking survives the
    window choice.  `closest` is the single candidate nearest the shipped support, kept because it
    is the strictest reading - and because it is the one that flips on a knife edge.
    """
    out = dict(best_in_window={}, closest=None, n_candidates_in_window={})
    for tol in SUPPORT_WINDOWS:
        cands = [r for r in rows if abs(int(r["emission_px"]) - support) <= tol * support]
        out["n_candidates_in_window"][str(tol)] = len(cands)
        out["best_in_window"][str(tol)] = (_row_brief(max(cands, key=lambda r: float(r["dti"])), support)
                                           if cands else None)
    all_win = [r for r in rows if abs(int(r["emission_px"]) - support) <= SUPPORT_WINDOWS[-1] * support]
    if all_win:
        out["closest"] = _row_brief(min(all_win, key=lambda r: abs(int(r["emission_px"]) - support)),
                                    support)
    return out


def build_report(proxy_dir: Path, override_support: Optional[int],
                 block_report: Optional[Path]) -> dict:
    sweeps = load_sweeps(proxy_dir)
    ship = shipped_support(sweeps, override_support, block_report)
    support = ship["support_px"]

    fields = []
    for s in sweeps:
        adopted = next((r for r in s["rows"] if is_adopted(r)), None)
        matched = matched_support_rows(s["rows"], support)
        best = max(s["rows"], key=lambda r: float(r["dti"]))
        fields.append(dict(
            field=s["label"], sweep_file=s["file"], run_ids=s["run_ids"],
            pred_sha256=s["pred_sha256"], n_hard_candidates=s["n_rows"],
            adopted_policy=(dict(t0=adopted["t0"], width_px=adopted["dilate"],
                                 dti=adopted["dti"], emission_px=adopted["emission_px"],
                                 TP_w=adopted["TP_w"], FP_w=adopted["FP_w"], FN_w=adopted["FN_w"])
                            if adopted else None),
            matched_support=matched,
            unconstrained_best=dict(t0=best["t0"], width_px=best["dilate"], dti=best["dti"],
                                    emission_px=best["emission_px"]),
            is_shipped_field=(s["file"] == SHIPPED_FIELD_SWEEP),
        ))

    def dti_in(f, tol):
        row = (f["matched_support"]["best_in_window"].get(str(tol)) or {})
        return row.get("dti")

    ranked = {}
    for tol in SUPPORT_WINDOWS:
        ranked[str(tol)] = sorted([f for f in fields if dti_in(f, tol) is not None],
                                  key=lambda f: -dti_in(f, tol))
    primary = ranked[str(PRIMARY_WINDOW)]
    top_fields = {tol: (ranked[tol][0]["field"] if ranked[tol] else None) for tol in ranked}
    stable = len(set(v for v in top_fields.values() if v)) <= 1
    shipped = next((f for f in fields if f["is_shipped_field"]), None)
    shipped_dti = dti_in(shipped, PRIMARY_WINDOW) if shipped else None

    verdict = dict(
        shipped_field=(shipped["field"] if shipped else None),
        shipped_support_px=support,
        adopted_policy=ADOPTED_LABEL,
        primary_rule=(f"best hard candidate within +-{PRIMARY_WINDOW * 100:.0f} % of the shipped "
                      "support, the same rule applied to every field"),
        ranking_at_matched_support={tol: [(f["field"], dti_in(f, float(tol))) for f in ranked[tol]]
                                    for tol in ranked},
        ranking_at_the_adopted_floor=[(f["field"], f["adopted_policy"]["dti"])
                                      for f in sorted([x for x in fields if x["adopted_policy"]],
                                                      key=lambda x: -x["adopted_policy"]["dti"])],
        top_field_by_window=top_fields,
        ranking_stable_across_windows=stable,
        knife_edge_note=("the single nearest-support candidate is reported per field as "
                         "matched_support.closest; for fields whose value distribution differs from "
                         "the shipped one it can be a different (floor, width) than the "
                         "best-in-window candidate, which is why the verdict uses the window rule "
                         "and reports all three windows"),
    )
    if not primary:
        verdict["conclusion"] = (
            f"no field has a hard candidate within +-{SUPPORT_WINDOWS[-1] * 100:.0f} % of the "
            f"shipped support ({support:,} px), so no matched-support comparison is possible from "
            "the committed sweeps; widen the swept floor grid "
            "(.github/triggers/proxy-eval-params SHAPING_VALUES) and re-run proxy-eval.yml")
    elif shipped and primary[0]["field"] != shipped["field"] and stable:
        top = primary[0]
        verdict["surrogate_prefers_a_different_field"] = dict(
            field=top["field"], sweep_file=top["sweep_file"], run_ids=top["run_ids"],
            dti=dti_in(top, PRIMARY_WINDOW),
            policy=(top["matched_support"]["best_in_window"][str(PRIMARY_WINDOW)]),
            shipped_dti_at_matched_support=shipped_dti,
            gain=round(dti_in(top, PRIMARY_WINDOW) - (shipped_dti or 0.0), 6))
        verdict["conclusion"] = (
            f"At matched support the surrogate prefers the field swept in {top['sweep_file']} "
            f"(run ids {top['run_ids']}) over the shipped field "
            f"({dti_in(top, PRIMARY_WINDOW):.4f} vs {shipped_dti:.4f} proxy DTI), and that "
            "preference holds in every support window tested.  The pre-registered rule in "
            "data/evidence/emission_decision.json ranks the POLICY axis by worst-case contrast "
            "across fields and does NOT rank the FIELD axis, so this is an un-ruled decision: "
            "commit a field-selection rule FIRST, then re-blend through reblend.yml.")
    elif shipped and primary[0]["field"] != shipped["field"] and not stable:
        verdict["conclusion"] = (
            f"the surrogate's top field at matched support depends on the support window "
            f"({json.dumps(top_fields)}), so no field change is supportable: the ranking is not "
            "robust to the one free parameter this comparison introduces.")
    else:
        verdict["conclusion"] = (
            f"The shipped field ({shipped['field']}) is also the surrogate's best at matched "
            f"support ({shipped_dti:.4f} proxy DTI within +-{PRIMARY_WINDOW * 100:.0f} % of "
            f"{support:,} px)"
            + (", and it stays best in every window tested" if stable else
               " (note: the top field is not the same in every window)")
            + ".  The higher absolute DTI other fields reach at floor 0.1 is a SUPPORT effect - "
              "they emit several times more pixels at that floor - not better detection: at the "
              "shipped support they score lower.  No field change is indicated.")

    return dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/compare_emission_fields.py",
        purpose=("scale-free comparison of the ensemble FIELDS behind the committed sweeps: at a "
                 "fixed floor the support shrinks as more folds are averaged, so fields must be "
                 "compared at matched support, not at a shared threshold"),
        metric_source="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric",
        adopted_policy=ADOPTED,
        shipped_support=ship,
        support_windows=list(SUPPORT_WINDOWS), primary_window=PRIMARY_WINDOW,
        fields=fields,
        verdict=verdict,
        caveats=[
            "the new-fault-like proxy population (USGS SGMC structure absent from the labels) is a "
            "SURROGATE for the scored truth, not the scored truth; every ranking here inherits that "
            "limitation (data/evidence/emission_decision.json caveats)",
            "matched support equalises the NUMBER of emitted pixels, not their spatial distribution: "
            "two fields at equal support can still differ in how much of that support is wrong mass",
            "the unconstrained-best column is a maximum over ~130 candidates on the same population "
            "the comparison is judged by - it is upward biased and is reported for completeness, "
            "never as a shipping recommendation",
            "no rule in this repository currently selects the field; this script reports the "
            "measurement and says so rather than deciding",
        ],
    )


def print_report(rep: dict) -> None:
    ship, v = rep["shipped_support"], rep["verdict"]
    srcs = "; ".join(f"{k}={val}" for k, val in ship["sources"].items())
    print(f"shipped support: {ship['support_px']:,} px (sources agree: {ship['sources_agree']})")
    if srcs:
        print(f"  support sources: {srcs}")
    print(f"adopted policy: {v['adopted_policy']}   shipped field: {v['shipped_field']}\n")
    head = (f"  {'field':<11} {'run ids':<30} {'adopted DTI':>11} {'support px':>11} "
            f"{'matched DTI':>11} {'at (floor,w)':>13} {'closest DTI':>11}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    tol = str(rep["primary_window"])
    for f in rep["fields"]:
        ad, ms = f["adopted_policy"], f["matched_support"]
        runs = str(f["run_ids"] or "")
        if len(runs) > 30:
            runs = runs[:29] + "..."
        ad_dti = f"{ad['dti']:.4f}" if ad else "-"
        ad_px = f"{ad['emission_px']:,}" if ad else "-"
        bw = (ms or {}).get("best_in_window", {}).get(tol)
        ms_dti = f"{bw['dti']:.4f}" if bw else "-"
        ms_pol = f"({bw['t0']:g},{bw['width_px']})" if bw else "-"
        cl = (ms or {}).get("closest")
        cl_dti = f"{cl['dti']:.4f}" if cl else "-"
        tag = "   <- shipped" if f["is_shipped_field"] else ""
        print(f"  {f['field']:<11} {runs:<30} {ad_dti:>11} {ad_px:>11} "
              f"{ms_dti:>11} {ms_pol:>13} {cl_dti:>11}{tag}")
    print(f"  (matched DTI = best hard candidate within +-{float(tol) * 100:.0f} % of the shipped "
          f"support; windows tested: {', '.join(f'+-{t * 100:.0f}%' for t in rep['support_windows'])}; "
          f"top field per window: {json.dumps(v['top_field_by_window'])}; stable: "
          f"{v['ranking_stable_across_windows']})")
    print(f"\nverdict: {v['conclusion']}")
    pref = v.get("surrogate_prefers_a_different_field")
    if pref:
        print(f"  surrogate preference: field {pref['field']} (runs {pref['run_ids']}) at floor "
              f"{pref['floor_needed']:g} -> {pref['dti']:.4f} (gain {pref['gain']:+.4f} over the "
              "shipped field at matched support)")
    for c in rep["caveats"]:
        print(f"  caveat: {c}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", default="data/evidence/proxy")
    ap.add_argument("--shipped-support", type=int, default=None,
                    help="override the written support of the shipped artifact (px)")
    ap.add_argument("--block-report", default="data/evidence/block_holdout/block_stratified.json",
                    help="block-stratified report used to cross-check the shipped support")
    ap.add_argument("--out", default="data/evidence/emission_field_axis.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    rep = build_report(Path(a.proxy_dir), a.shipped_support,
                       Path(a.block_report) if a.block_report else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    if not a.quiet:
        print_report(rep)
        print(f"\nwrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
