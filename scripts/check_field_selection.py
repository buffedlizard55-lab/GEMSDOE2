#!/usr/bin/env python3
"""Check a proposed re-blend FIELD against the pre-registered FIELD-selection rule.

WHAT THIS IS
------------
`docs/FIELD_SELECTION_RULE.md` (v1, 2026-09-19) pre-registers the rule that decides which
ensemble field the adopted policy (floor 0.1 / thin / width 0 px) is applied to.  It was
committed BEFORE the next re-blend, which is what makes it a rule rather than a post-hoc
justification: the policy axis was already ruled by `data/evidence/emission_decision.json`,
and `scripts/compare_emission_fields.py` measured the field axis and said the decision
"needs either a rule committed BEFORE the next re-blend, or a scored submission".

This script evaluates the rule from the committed evidence and derives the verdict:

  eligibility  F1  fold completeness of the field's constituent runs
               F2  the field has its own committed proxy sweep (floor calibrated to its scale)
  conditions   R1  matched-support contrast: best-in-window(+-25%) proxy DTI beats the
                        shipped field by more than 0.010
               R2  ranked first at matched support in ALL THREE windows (+-15/25/40 %)
               R3  paired block bootstrap of the per-block DTI difference, P(F > S) >= 0.95;
                        NOT_MEASURABLE_FROM_COMMITTED_BYTES unless both fields' emission
                        rasters are supplied, and an unmeasured R3 can never pass
               R4  the adopted policy beats the field's OWN in-domain reference policy by
                        more than 0.010 (policy transfer, the policy rule's condition 3)
               R5  at least 3 hard candidates in the primary window (window actually populated)

  verdict      ADOPT F   - all of F1..F2 and R1..R5 pass (a field change is a claim)
               KEEP S    - anything else; the default is to do nothing

The verdict is COMPUTED from the condition table, never typed.  `.github/workflows/reblend.yml`
runs `--gate` as a pre-blend gate for any field other than the shipped one.

USAGE
    # evaluate every committed field against the shipped one (writes the evidence record)
    python scripts/check_field_selection.py

    # the pre-blend gate used by reblend.yml (exit 0 iff the field is allowed to blend)
    python scripts/check_field_selection.py --gate --field ens123
    python scripts/check_field_selection.py --gate --field mean12     # shipped field: no-op pass

    # R3 measurement: supply both fields' RAW probability rasters (the sweep inputs)
    python scripts/check_field_selection.py --pred-shipped ensemble_mean.tif \
        --pred ens123_field.tif --field ens123

Exit codes: 0 = ran (verdict in the report), 2 = missing/unreadable inputs, 1 = gate failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# the measurement machinery this rule builds on - same loaders, same window rule
_spec = importlib.util.spec_from_file_location(
    "compare_emission_fields", ROOT / "scripts" / "compare_emission_fields.py")
cef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cef)

# ---- rule constants (docs/FIELD_SELECTION_RULE.md is the canonical statement) ----
RULE_VERSION = "v1"
RULE_DOC = "docs/FIELD_SELECTION_RULE.md"
RULE_REGISTERED = "2026-09-19"                    # committed before any re-blend after this date
CONTRAST_MARGIN = 0.010                           # R1 + R4: the policy rule's pre-registered margin
BOOT_P_THRESHOLD = 0.95                           # R3: paired block-bootstrap acceptance
BOOTSTRAPS = 2000
MIN_CANDIDATES_IN_PRIMARY_WINDOW = 3              # R5
SHIPPED_FIELD_LABEL = "mean12"                    # the field the shipped artifact was blended from

# the official metric, as published on the problem page (same values as src/metrics.py defaults)
R_PIX, ALPHA, BETA, EPS = 3, 0.2, 0.8, 1.0e-7
PROXY_TIF = ROOT / "data" / "evidence" / "proxy" / "proxy_catalogue.tif"


def _parse_trigger_params(path: Path) -> dict:
    """KEY=VALUE lines from a .github/triggers/*-params file (comments and blanks skipped)."""
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _status(passed: Optional[bool], note: str = "") -> dict:
    return dict(status=("PASS" if passed else "FAIL"), note=note)


def _evaluate_sweep_conditions(f: dict, shipped: dict, support_px: int, all_fields: list,
                               trigger_params: dict) -> dict:
    """F1, F2 and R1/R2/R4/R5 for one field, from its committed sweep.

    All the numbers come from the same loaders and the same window rule as
    `scripts/compare_emission_fields.py`, so this cannot drift from the published
    field-axis measurement.
    """
    tol = str(cef.PRIMARY_WINDOW)
    bw = (f["matched_support"]["best_in_window"] or {}).get(tol)
    shipped_bw = (shipped["matched_support"]["best_in_window"] or {}).get(tol)

    # F2 - committed sweep with its own reference policy recorded
    inputs = f["sweep_doc"].get("inputs") or {}
    has_reference = inputs.get("reference_t0") is not None
    f2 = _status(
        has_reference and f["adopted_policy"] is not None,
        "own sweep committed, inputs.reference_t0 recorded, adopted-policy row present"
        if has_reference and f["adopted_policy"] is not None
        else "missing own sweep / reference policy / adopted-policy row - "
             "a floor is a threshold on the field, so a field without its own sweep "
             "has no calibrated floor and cannot be adopted")

    # F1 - fold completeness, from the committed provenance chain:
    #   sweep_source-<field>.json records the run ids -> .github/triggers/reblend-params pins
    #   the run ids of the currently re-blendable field (RUN_ID) and its evidence directory
    #   (EVIDENCE_DIR) with the live-fold floor (MIN_FOLDS) -> that directory's usable_folds.txt
    #   + blend_report.json are the auditable liveness evidence.  The re-blend workflow enforces
    #   MIN_FOLDS again at blend time, so a silently short download can never ship.
    run_ids = [r for r in str(f.get("run_ids") or "").split(",") if r.strip()]
    if not run_ids:
        f1 = dict(status="UNVERIFIED", note="sweep source does not record run ids")
        committed, missing = [], list(run_ids)
    else:
        committed, missing = [], []
        rb_runs = [r for r in str(trigger_params.get("RUN_ID", "")).split(",") if r.strip()]
        ev_dir = ROOT / trigger_params.get("EVIDENCE_DIR", "")
        min_folds = trigger_params.get("MIN_FOLDS", "")
        if rb_runs and sorted(run_ids) == sorted(rb_runs):
            committed, missing = list(rb_runs), []
            usable = ev_dir / "usable_folds.txt"
            report = ev_dir / "blend_report.json"
            n_live = (len([l for l in usable.read_text().splitlines() if l.strip()])
                      if usable.is_file() else None)
            n_report = None
            if report.is_file():
                try:
                    n_report = json.loads(report.read_text()).get("n_folds")
                except Exception:                                          # noqa: BLE001
                    n_report = None
            min_n = int(min_folds) if min_folds.isdigit() else None
            ok = (n_live is not None and (min_n is None or n_live >= min_n)
                  and (n_report is None or n_report == n_live))
            f1 = _status(
                ok,
                (f"field run ids == pinned re-blend RUN_ID {rb_runs}; evidence dir "
                 f"{ev_dir.relative_to(ROOT)}: usable_folds.txt {n_live} live folds, "
                 f"blend_report.n_folds {n_report}, MIN_FOLDS {min_folds or 'unpinned'}")
                if ok else
                (f"field run ids == pinned re-blend RUN_ID {rb_runs} but the evidence dir "
                 f"{ev_dir.relative_to(ROOT)} is missing usable_folds.txt / blend_report.json or "
                 f"the live-fold count disagrees with MIN_FOLDS {min_folds or 'unpinned'}"))
        else:
            committed = []
            for rid in run_ids:
                d = ROOT / "data" / "evidence" / "runs" / rid
                if (d / "usable_folds.txt").exists() or (d / "fold_provenance.txt").exists():
                    committed.append(rid)
            missing = [r for r in run_ids if r not in committed]
            if missing:
                f1 = dict(status="UNVERIFIED",
                          note=(f"runs {missing} are not the pinned re-blend and have no committed "
                               f"fold-liveness evidence (run dirs with usable_folds.txt / "
                               f"fold_provenance.txt); completeness is enforced by the re-blend "
                               f"MIN_FOLDS gate when this field is pinned and blended"))
            else:
                f1 = _status(True, f"all constituent runs have committed fold-liveness evidence: "
                                   f"{run_ids}")

    # R1 - matched-support contrast at the primary window
    if bw and shipped_bw:
        r1 = _status(bw["dti"] > shipped_bw["dti"] + CONTRAST_MARGIN,
                     f"best-in-window(+-{cef.PRIMARY_WINDOW * 100:.0f} %) "
                     f"{bw['dti']:.6f} vs shipped {shipped_bw['dti']:.6f} "
                     f"(margin {CONTRAST_MARGIN}); candidate ({bw['t0']:g}, w{bw['width_px']}px, "
                     f"{bw['emission_px']:,} px)")
    else:
        r1 = dict(status="FAIL", note="no hard candidate within the primary window")

    # R2 - first in ALL three windows (same window rule applied to every field)
    tops = {}
    for t in cef.SUPPORT_WINDOWS:
        rows = []
        for g in all_fields:
            row = (g["matched_support"]["best_in_window"] or {}).get(str(t))
            if row:
                rows.append((g["field"], row["dti"]))
        if rows:
            tops[str(t)] = max(rows, key=lambda x: x[1])
    r2 = _status(all(tops.get(str(t), (None,))[0] == f["field"] for t in cef.SUPPORT_WINDOWS),
                 f"top field per window: { {k: v[0] for k, v in tops.items()} }")

    # R4 - policy transfer on the field's own sweep
    sv = (f["sweep_doc"].get("results") or {}).get("sweep_verdict") or {}
    current = sv.get("current_policy_dti")
    if f["adopted_policy"] is not None and current is not None:
        transfer = f["adopted_policy"]["dti"] - float(current)
        r4 = _status(transfer > CONTRAST_MARGIN,
                     f"adopted policy {f['adopted_policy']['dti']:.6f} vs the field's own "
                     f"in-domain reference policy {float(current):.6f} "
                     f"(reference_t0={inputs.get('reference_t0')}); transfer {transfer:+.6f} "
                     f"vs margin {CONTRAST_MARGIN}")
    else:
        r4 = dict(status="FAIL", note="adopted-policy row or own reference policy missing")

    # R5 - the primary window is actually populated
    n_in = (f["matched_support"].get("n_candidates_in_window") or {}).get(tol, 0)
    r5 = _status(n_in >= MIN_CANDIDATES_IN_PRIMARY_WINDOW,
                 f"{n_in} hard candidates within +-{cef.PRIMARY_WINDOW * 100:.0f} % of "
                 f"{support_px:,} px (need >= {MIN_CANDIDATES_IN_PRIMARY_WINDOW})")

    return dict(
        run_ids=run_ids or None,
        run_evidence={"committed": committed, "not_committed": missing},
        eligibility={"F1_fold_completeness": f1, "F2_committed_own_sweep": f2},
        conditions={
            "R1_matched_support_contrast": r1,
            "R2_window_stability": r2,
            "R4_policy_transfer": r4,
            "R5_window_sampling": {"min_candidates": MIN_CANDIDATES_IN_PRIMARY_WINDOW, **r5},
        },
        matched_candidate=dict(bw) if bw else None,
        matched_dti=(bw["dti"] if bw else None),
    )


def _per_block_rows(pred, truth, t0: float, width: int,
                    blocks, n_blocks: int, R: int, alpha: float, beta: float,
                    eps: float, label: str) -> dict:
    """One sweep candidate of one field, scored per block (exact official-metric components)."""
    from src.metrics import GtContext, block_aggregate
    from src.submission_optim import optimize_submission

    emis = optimize_submission(pred, R=R, t0=float(t0), thin=True, dilate=int(width), soft=False)
    ctx = GtContext(truth, R_pixels=R)
    comp = block_aggregate(ctx, emis, blocks, n_blocks)
    per_block = []
    for b in range(n_blocks):
        ngt = int(comp["n_gt"][b])
        denom = comp["TP"][b] + alpha * comp["FP"][b] + beta * comp["FN"][b] + eps
        per_block.append(dict(block=int(b),
                              dti=(round(float(comp["TP"][b] / denom), 6) if ngt else None),
                              TP_w=float(comp["TP"][b]), FP_w=float(comp["FP"][b]),
                              FN_w=float(comp["FN"][b]), n_gt=ngt,
                              scoreable=bool(ngt > 0)))
    return dict(label=label, t0=float(t0), width_px=int(width),
                support_px=int(np.count_nonzero(emis > 0)), blocks=per_block)


def _measure_r3(pred_shipped: Path, pred_field: Path, cand_ship: dict, cand_field: dict,
                support_px: int) -> dict:
    """R3: paired block bootstrap of the per-block DTI difference (F - S).

    Both emissions are built with the SAME function the sweeps used
    (src.submission_optim.optimize_submission, thin=True, soft=False) on the fields' raw
    probability rasters, and are scored per 51.2 km block with the exact additive components
    of the official metric.  The bootstrap resamples the SAME block ids for both fields
    (src.metrics.bootstrap_from_blocks, row 0 = the shipped field = the reference).
    """
    import numpy as np
    import rasterio
    from src.blocks import block_id_map, DEFAULT_BLOCK_PX
    from src.metrics import bootstrap_from_blocks

    proxy = np.asarray(rasterio.open(PROXY_TIF).read(1), dtype=np.uint8)
    truth = (proxy == 2).astype(np.float32)      # code 2 = proxy-only (absent from labels)
    blocks = block_id_map(truth.shape, DEFAULT_BLOCK_PX)
    n_blocks = int(blocks.max()) + 1

    tol = float(cef.PRIMARY_WINDOW)
    for name, cand in (("shipped", cand_ship), ("proposed", cand_field)):
        dev = abs(cand["emission_px"] - support_px) / float(support_px)
        if dev > tol + 1e-9:
            raise ValueError(
                f"{name} candidate ({cand['t0']:g}, w{cand['width_px']}px, "
                f"{cand['emission_px']:,} px) is outside the primary support window "
                f"+-{tol * 100:.0f} % of {support_px:,} px - R3 is defined on best-in-window "
                f"candidates only")

    def read_field(p: Path):
        with rasterio.open(p) as src:
            if (src.width, src.height) != (3292, 3730):
                raise ValueError(f"{p}: grid {src.width}x{src.height} != competition grid")
            a = src.read(1).astype(np.float32)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
        a[~np.isfinite(a)] = 0.0
        return a

    row_s = _per_block_rows(read_field(pred_shipped), truth, cand_ship["t0"], cand_ship["width_px"],
                            blocks, n_blocks, R_PIX, ALPHA, BETA, EPS,
                            f"shipped:{cand_ship['t0']:g}_w{cand_ship['width_px']}px")
    row_f = _per_block_rows(read_field(pred_field), truth, cand_field["t0"], cand_field["width_px"],
                            blocks, n_blocks, R_PIX, ALPHA, BETA, EPS,
                            f"proposed:{cand_field['t0']:g}_w{cand_field['width_px']}px")
    boot = bootstrap_from_blocks([row_s, row_f], alpha=ALPHA, beta=BETA, eps=EPS,
                                 n_boot=BOOTSTRAPS, seed=0)
    p_beats = (boot.get(row_f["label"]) or {}).get("prob_beats_reference")
    return dict(
        measured=True,
        inputs={"shipped_raster": str(pred_shipped), "proposed_raster": str(pred_field)},
        shipped_candidate={k: row_s[k] for k in ("label", "t0", "width_px", "support_px")},
        proposed_candidate={k: row_f[k] for k in ("label", "t0", "width_px", "support_px")},
        n_blocks=n_blocks, bootstraps=BOOTSTRAPS,
        p_proposed_beats_shipped=p_beats,
        bootstrap={row_f["label"]: boot.get(row_f["label"])},
        threshold=BOOT_P_THRESHOLD,
        passed=(p_beats is not None and p_beats >= BOOT_P_THRESHOLD),
    )


def evaluate(proxy_dir: Path, block_report: Path, pred_shipped: Optional[Path] = None,
             pred_for: Optional[dict] = None, proposal: Optional[str] = None) -> dict:
    sweeps = cef.load_sweeps(proxy_dir)
    ship = cef.shipped_support(sweeps, None, block_report)
    support_px = ship["support_px"]

    fields = []
    for s in sweeps:
        adopted = next((r for r in s["rows"] if cef.is_adopted(r)), None)
        best = max(s["rows"], key=lambda r: float(r["dti"]))
        fields.append(dict(
            field=s["label"], sweep_file=s["file"], run_ids=s["run_ids"],
            pred_sha256=s["pred_sha256"], n_hard_candidates=s["n_rows"],
            sweep_doc=s["doc"],
            adopted_policy=(dict(t0=adopted["t0"], width_px=adopted["dilate"], dti=adopted["dti"],
                                 emission_px=adopted["emission_px"]) if adopted else None),
            matched_support=cef.matched_support_rows(s["rows"], support_px),
            unconstrained_best=dict(t0=best["t0"], width_px=best["dilate"], dti=best["dti"],
                                    emission_px=best["emission_px"]),
            is_shipped_field=(s["file"] == cef.SHIPPED_FIELD_SWEEP),
        ))
    shipped = next((f for f in fields if f["is_shipped_field"]), None)
    if shipped is None:
        raise SystemExit(2)
    if shipped["field"] != SHIPPED_FIELD_LABEL:
        print(f"::warning:: the shipped field is now {shipped['field']!r}, not "
              f"{SHIPPED_FIELD_LABEL!r}: the rule's reference field changed with the "
              f"shipped artifact - re-commit docs/FIELD_SELECTION_RULE.md with the new "
              f"shipped field before relying on this report", file=sys.stderr)

    trigger_params = _parse_trigger_params(ROOT / ".github" / "triggers" / "reblend-params")
    report = {}
    for f in fields:
        entry = _evaluate_sweep_conditions(f, shipped, support_px, fields, trigger_params)
        if f["is_shipped_field"]:
            entry["is_shipped_field"] = True
            entry["note"] = ("the current shipped field: re-blending it changes nothing about "
                             "which field ships, so it passes the re-blend gate without R3")
            entry["conditions"]["R1_matched_support_contrast"] = dict(
                status="NOT_APPLICABLE",
                note=("the shipped field IS the reference of the R1 contrast - it cannot beat "
                      "itself by the margin; KEEP of this field is the rule's default by "
                      "construction"))
            entry["conditions"]["R3_block_bootstrap"] = dict(
                status="NOT_APPLICABLE",
                note="the shipped field is the reference; R3 applies to a proposed NEW field")
        else:
            entry["is_shipped_field"] = False
            r3 = dict(status="NOT_MEASURABLE_FROM_COMMITTED_BYTES",
                      note=("paired block bootstrap of the per-block DTI difference needs both "
                            "fields' raw probability rasters (the sweep inputs); only the "
                            "shipped field's emission raster is committed.  Until measured, R3 "
                            "cannot pass and the verdict cannot be ADOPT - this is the rule "
                            "working as pre-registered, not a fallback"),
                      threshold=BOOT_P_THRESHOLD, bootstraps=BOOTSTRAPS)
            pf = (pred_for or {}).get(f["field"])
            if pred_shipped and pf:
                r3 = _measure_r3(pred_shipped, pf, shipped["matched_support"]
                                               ["best_in_window"][str(cef.PRIMARY_WINDOW)],
                                 f["matched_support"]["best_in_window"]
                                 [str(cef.PRIMARY_WINDOW)], support_px)
                r3["status"] = "PASS" if r3["passed"] else "FAIL"
            entry["conditions"]["R3_block_bootstrap"] = r3
        report[f["field"]] = entry

    # ---- derive the verdict; never type it ----
    def all_pass(entry: dict, ref_field: bool) -> bool:
        el = entry["eligibility"]
        conds = entry["conditions"]
        elig = el["F1_fold_completeness"]["status"] == "PASS" and el["F2_committed_own_sweep"]["status"] == "PASS"
        r = {k: v["status"] for k, v in conds.items()}
        if ref_field:
            r["R3_block_bootstrap"] = "PASS"        # reference field: R3 not applicable
        return elig and all(v == "PASS" for v in r.values())

    candidates = [f["field"] for f in fields
                  if not f["is_shipped_field"] and all_pass(report[f["field"]], False)]
    if len(candidates) == 1 and proposal is None:
        verdict, why = f"ADOPT {candidates[0]}", (
            "every eligibility test and adoption condition passed for exactly one field")
    elif len(candidates) > 1:
        verdict, why = "CONFLICT", (
            f"more than one field passed every condition ({candidates}); the rule does not rank "
            "among passing fields - that would be a new free parameter, so nothing ships")
    elif proposal is not None and proposal in candidates:
        verdict, why = f"ADOPT {proposal}", "the proposed field passed every condition"
    else:
        verdict, why = f"KEEP {shipped['field']}", (
            "no field passed all of F1-F2 + R1-R5 (or the passing field is not the one "
            "proposed); the default is to do nothing - a field change is a claim and requires "
            "positive evidence on every axis")

    out = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/check_field_selection.py",
        rule=dict(id=f"field-selection-rule-{RULE_VERSION}", doc=RULE_DOC,
                  registered=RULE_REGISTERED,
                  committed_before="any re-blend after 2026-09-19 (session 18)",
                  constants=dict(contrast_margin=CONTRAST_MARGIN,
                                 boot_p_threshold=BOOT_P_THRESHOLD,
                                 bootstraps=BOOTSTRAPS,
                                 min_candidates_primary_window=MIN_CANDIDATES_IN_PRIMARY_WINDOW,
                                 support_windows=list(cef.SUPPORT_WINDOWS),
                                 primary_window=cef.PRIMARY_WINDOW)),
        shipped_field=shipped["field"],
        shipped_support_px=support_px,
        shipped_support_sources=ship["sources"],
        adopted_policy=cef.ADOPTED,
        proxy_population="data/evidence/proxy/proxy_catalogue.tif code 2 (61,664 px SGMC trace "
                         "absent from labels.tif - the standing surrogate for the scored new faults)",
        proposal=proposal,
        fields=report,
        verdict=dict(decision=verdict, why=why),
        caveats=[
            "the proxy population is a surrogate for the scored truth (emission_decision.json caveats)",
            "matched support equalises the NUMBER of emitted pixels, not their spatial distribution",
            "R3 measured on the surrogate population only; the rule cannot see the scored truth "
            "either - that is what the public leaderboard upload is for",
        ],
    )
    return out


def _print(rep: dict) -> None:
    v = rep["verdict"]
    print(f"field-selection rule {rep['rule']['id']} (registered {rep['rule']['registered']}, "
          f"committed before any re-blend after that date)")
    print(f"shipped field: {rep['shipped_field']}   support {rep['shipped_support_px']:,} px   "
          f"adopted policy: {cef.ADOPTED_LABEL}\n")
    for name, f in rep["fields"].items():
        tag = "  <- shipped" if f.get("is_shipped_field") else ""
        print(f"{name}{tag}")
        for grp in ("eligibility", "conditions"):
            for cid, c in f[grp].items():
                note = c.get("note", "")
                if len(note) > 120:
                    note = note[:117] + "..."
                print(f"  {cid:<38} {c['status']:<32} {note}")
        print()
    print(f"VERDICT: {v['decision']} - {v['why']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", default="data/evidence/proxy")
    ap.add_argument("--block-report", default="data/evidence/block_holdout/block_stratified.json")
    ap.add_argument("--proposal", default=None,
                    help="field label being considered for adoption (default: any passing field)")
    ap.add_argument("--pred-shipped", default=None,
                    help="raw probability raster of the shipped field (enables R3)")
    ap.add_argument("--pred", action="append", default=[],
                    help="raw probability raster of a proposed field, '<field>=<path>' "
                         "(repeatable; enables R3 for that field)")
    ap.add_argument("--gate", action="store_true",
                    help="re-blend gate mode: exit 1 unless the field is allowed to blend")
    ap.add_argument("--field", default=None,
                    help="with --gate: the field the re-blend would produce")
    ap.add_argument("--runs", default=None,
                    help="with --gate: comma-separated RUN_IDs of the re-blend; the field is "
                         "identified by its run ids. RUN_IDs equal to the pinned re-blend "
                         "(.github/triggers/reblend-params) re-blend the SHIPPED field and pass "
                         "without measurement; any other run set must be ADOPTed by a committed "
                         "measurement (including R3) first")
    ap.add_argument("--out", default="data/evidence/field_selection.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    proxy_dir = Path(a.proxy_dir)
    if not proxy_dir.is_dir():
        print(f"::error:: {proxy_dir} does not exist - run proxy-eval first", file=sys.stderr)
        return 2
    pred_for = {}
    for spec in a.pred:
        if "=" not in spec:
            print(f"::error:: --pred takes '<field>=<path>', got {spec!r}", file=sys.stderr)
            return 2
        k, v = spec.split("=", 1)
        pred_for[k.strip()] = Path(v)

    try:
        rep = evaluate(proxy_dir, Path(a.block_report),
                       Path(a.pred_shipped) if a.pred_shipped else None,
                       pred_for, a.proposal)
    except FileNotFoundError as exc:
        print(f"::error:: missing input: {exc}", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))

    if a.gate:
        verdict = rep["verdict"]["decision"]
        if a.runs:
            # the workflow form: the field is identified by the re-blend's RUN_ID set
            pinned = [r.strip() for r in
                      str(_parse_trigger_params(ROOT / ".github" / "triggers" / "reblend-params")
                          .get("RUN_ID", "")).split(",") if r.strip()]
            runs = [r.strip() for r in a.runs.split(",") if r.strip()]
            if sorted(runs) == sorted(pinned) and pinned:
                ok, msg = True, (f"re-blend RUN_IDs {runs} == the pinned re-blend: this re-blends "
                                 f"the shipped field ({rep['shipped_field']}), which changes "
                                 f"nothing about which field ships - gate passes without R3")
            else:
                match = [name for name, e in rep["fields"].items()
                         if sorted(runs) == sorted(e.get("run_ids") or [])]
                if len(match) == 1 and verdict == f"ADOPT {match[0]}" and \
                        rep["fields"][match[0]]["conditions"]["R3_block_bootstrap"]["status"] == "PASS":
                    ok, msg = True, (f"gate passes: RUN_IDs {runs} identify field {match[0]} and "
                                     f"the committed rule measurement says ADOPT {match[0]} "
                                     f"with R3 measured")
                else:
                    ok = False
                    msg = (f"gate FAILS: RUN_IDs {runs} are not the pinned re-blend {pinned}, so "
                           f"this re-blend would ship a NEW field; {RULE_DOC} requires a committed "
                           f"field_selection.json that ADOPTs exactly that field with R3 measured "
                           f"(current verdict: {verdict!r}; fields with matching run ids: "
                           f"{match or 'none'}) - commit the passing measurement, including R3 on "
                           f"both fields' raw probability rasters, BEFORE re-blending")
        else:
            field = a.field or rep["shipped_field"]
            if field == rep["shipped_field"]:
                ok, msg = True, (f"field {field} is the current shipped field: re-blending it "
                                 f"changes nothing about which field ships - gate passes without R3")
            elif verdict == f"ADOPT {field}":
                ok, msg = True, f"gate passes: the rule says ADOPT {field} (all conditions measured)"
            else:
                ok = False
                entry = rep["fields"].get(field)
                fails = []
                if entry:
                    for grp in ("eligibility", "conditions"):
                        for cid, c in entry[grp].items():
                            if c["status"] != "PASS":
                                fails.append(f"{cid}={c['status']}")
                msg = (f"gate FAILS: re-blending field {field} is not allowed by "
                       f"{RULE_DOC} (verdict {verdict!r}; "
                       + (", ".join(fails) if fails else "field not in committed evidence")
                       + ") - commit the passing measurement (including R3 on both fields' raw "
                         "rasters) BEFORE re-blending")
        if not a.quiet:
            print(msg)
        if not ok:
            return 1

    if not a.quiet:
        _print(rep)
        print(f"\nwrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
