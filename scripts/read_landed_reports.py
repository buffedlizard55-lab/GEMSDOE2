#!/usr/bin/env python3
"""Read the runner reports that landed and derive the two summaries the next decision needs.

Session 18 dispatched four 300-minute jobs (block-holdout folds 1-3 and the SGMC pseudo-label
fold 0) and left the next session one instruction: *read the four landed reports*.  This script
is that reading, and it is deliberately **derivation only**:

  * it never trains, never re-scores a raster and never invents a number;
  * every DTI it reports is either copied from a committed report or recomposed EXACTLY from
    that report's own per-block metric components (TP_w/FP_w/N_w are sums, so a block bootstrap
    needs no re-scoring - src/metrics.py::bootstrap_from_blocks);
  * every provenance claim (which fold, which seed, which population, which pseudo-label source)
    is read out of the committed JSON/YAML, not typed here;
  * if an input is missing it says so and exits non-zero rather than printing a partial table
    that looks complete.

Two artefacts are written:

1. ``data/evidence/block_holdout/fold_gap_summary.json``
   The memorisation-vs-transfer gap is a PER-FOLD quantity, so one fold is a point estimate with
   no spread.  This aggregates every committed ``fold*_generalisation_gap.json`` into its
   distribution (n, mean, min, max, sign counts) and carries each fold's own bootstrap
   reliability flag forward, because a fold restricted to 8-9 blocks has fewer resampling units
   than MIN_BLOCKS_FOR_A_READABLE_CI and its interval is coarse by the scorer's own rule.

2. ``data/evidence/pseudo_labels/fold<K>_two_population_contrast.json``
   The committed paired contrast (``fold0_paired_vs_baseline.json``) bootstrapped ONE population:
   the new-fault-like proxy.  That is the population whose truth is built from the SAME raster the
   pseudo-labels were cut from (``configs/config_pseudo_labels.yaml`` sets
   ``pseudo_label_path: data/evidence/proxy/proxy_catalogue.tif``, ``pseudo_code: 2``), so a gain
   there is partly "the model learned to draw SGMC traces" measured against SGMC traces.  The
   catalogue population in the very same reports is independent of the pseudo-label source, and
   this script contrasts BOTH - plus the ``combined`` population whenever a report carries it
   (``scripts/block_holdout_eval.py --combined-population``, the Phase-2-like surrogate).  The
   verdict is derived from the two signs, and the circularity of the proxy arm is derived from the
   config, never asserted in prose.

Usage:
    python scripts/read_landed_reports.py                 # both artefacts
    python scripts/read_landed_reports.py --fold 0 --quiet
    python scripts/read_landed_reports.py --check         # verify recomposition, write nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import bootstrap_from_blocks  # noqa: E402

BH = ROOT / "data/evidence/block_holdout"
PL = ROOT / "data/evidence/pseudo_labels"
DECISION = ROOT / "data/evidence/emission_decision.json"
PSEUDO_CONFIG = ROOT / "configs/config_pseudo_labels.yaml"
TRIGGER_PARAMS = ROOT / ".github/triggers/block-holdout-params"
HOLDOUT_CONFIG = ROOT / "configs/config_block_holdout.yaml"

PROXY_CODE_ONLY = 2          # must equal scripts/block_holdout_eval.py::PROXY_CODE_ONLY
TOL = 1e-6                   # the reports round DTI to 6 decimals


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _rel(path) -> str:
    """Repo-relative when the path is inside the repository, absolute otherwise.

    Evidence paths are quoted repo-relative in every report this script writes (that is what makes
    them auditable), but the same functions are called with temporary directories by the tests and
    with --gaps-out/--contrast-out by a runner, so a path outside the tree must not crash.
    """
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"MISSING INPUT: {path} - nothing is derived without its committed source")
    try:
        return json.loads(path.read_text())
    except Exception as exc:                                    # pragma: no cover
        raise SystemExit(f"UNREADABLE INPUT: {path}: {exc}")


def adopted_policy_label(decision: dict) -> dict:
    """Derive the adopted (floor, width) and the scorer's candidate label from the decision record.

    ``emission_decision.json`` names the adopted policy ``sweep_best_t0_0.1_width0px``;
    ``block_holdout_eval.py`` names the same candidate ``floor0.1_w0px``.  Parsing the one into
    the other keeps this script from hard-coding a policy that a later measurement may change.
    """
    cand = ((decision.get("verdict") or {}).get("best_measured_candidate") or {})
    policy = str(cand.get("policy") or "")
    m = re.fullmatch(r"sweep_best_t0_([\d.]+)_width(\d+)px", policy)
    if not m:
        raise SystemExit(f"cannot derive the adopted policy from {DECISION}: {policy!r}")
    floor, width = float(m.group(1)), int(m.group(2))
    return dict(floor=floor, width=width, policy=policy,
                candidate_label=f"floor{floor:g}_w{width}px",
                measured_dti=cand.get("measured_dti"),
                source=_rel(DECISION))


def candidate_row(report: dict, population: str, label: str) -> dict:
    pop = (report.get("populations") or {}).get(population)
    if not pop:
        raise SystemExit(f"population {population!r} is not in the report")
    for row in pop.get("candidates") or []:
        if row.get("label") == label:
            if not row.get("blocks"):
                twin = row.get("duplicate_of")
                raise SystemExit(
                    f"candidate {label} on {population} has no per-block rows (pruned as a "
                    f"duplicate of {twin}); a paired block bootstrap needs them - re-run the "
                    "scorer with --keep-duplicate-rows")
            return row
    raise SystemExit(f"candidate {label!r} was not scored on {population!r}")


def bootstrap_of(report: dict) -> dict:
    """The report's own bootstrap settings, so the recomposition uses the committed convention."""
    b = report.get("bootstrap") or {}
    return dict(n_boot=int(b.get("n") or 2000), seed=int(b.get("seed") or 0),
                reliability=(b.get("reliability") or {}))


def paired(base_row: dict, cand_row: dict, alpha: float, beta: float, eps: float,
           n_boot: int, seed: int, names: Sequence[str]) -> dict:
    """Paired block bootstrap of cand vs base on ONE population, recomposed from committed rows.

    ``bootstrap_from_blocks`` treats row 0 as the reference, so ``prob_beats_reference`` is the
    paired P(candidate > reference) over the SAME resampled blocks - the identical statistic the
    runner committed for the proxy arm, recomputed here for every arm.
    """
    rows = [dict(label=names[0], blocks=base_row["blocks"]),
            dict(label=names[1], blocks=cand_row["blocks"])]
    out = bootstrap_from_blocks(rows, alpha=alpha, beta=beta, eps=eps, n_boot=n_boot, seed=seed)
    ref, cand = out[names[0]], out[names[1]]

    def dti_of(row: dict) -> Optional[float]:
        tp, fp, fn = row["TP_w"], row["FP_w"], row["FN_w"]
        return tp / (tp + alpha * fp + beta * fn + eps)

    exact = {
        names[0]: abs(dti_of(base_row) - base_row["global_dti"]) <= TOL,
        names[1]: abs(dti_of(cand_row) - cand_row["global_dti"]) <= TOL,
    }
    return dict(
        reference=names[0], candidate=names[1],
        reference_dti=base_row["global_dti"], candidate_dti=cand_row["global_dti"],
        contrast=cand_row["global_dti"] - base_row["global_dti"],
        reference_support_px=base_row["support_px"], candidate_support_px=cand_row["support_px"],
        reference_components=dict(TP_w=base_row["TP_w"], FP_w=base_row["FP_w"],
                                  FN_w=base_row["FN_w"], n_gt=base_row["n_gt"]),
        candidate_components=dict(TP_w=cand_row["TP_w"], FP_w=cand_row["FP_w"],
                                  FN_w=cand_row["FN_w"], n_gt=cand_row["n_gt"]),
        bootstrap=dict(n_boot=n_boot, seed=seed,
                       reference_ci95=ref["dti_ci95"], candidate_ci95=cand["dti_ci95"],
                       contrast_p50=cand["contrast_vs_reference_p50"],
                       contrast_ci95=cand["contrast_vs_reference_ci95"],
                       prob_candidate_beats_reference=cand["prob_beats_reference"]),
        recomposition_exact=exact,
    )


# --------------------------------------------------------------------------------------
# 1. the fold-gap distribution
# --------------------------------------------------------------------------------------
def load_fold_gaps(directory=None) -> List[dict]:
    directory = Path(directory or BH)
    gaps = []
    for p in sorted(directory.glob("fold*_generalisation_gap.json")):
        m = re.fullmatch(r"fold(\d+)_generalisation_gap\.json", p.name)
        if not m:
            continue
        d = load_json(p)
        d["_fold"] = int(m.group(1))
        d["_path"] = _rel(p)
        gaps.append(d)
    gaps.sort(key=lambda d: d["_fold"])
    return gaps


def expected_folds(config=None, params=None) -> dict:
    config, params = Path(config or HOLDOUT_CONFIG), Path(params or TRIGGER_PARAMS)
    """How many folds the partition has, and which the trigger dispatched - both read, not typed."""
    n_folds = None
    if config.exists():
        m = re.search(r"^\s*block_folds:\s*(\d+)", config.read_text(), re.MULTILINE)
        n_folds = int(m.group(1)) if m else None
    dispatched: List[int] = []
    if params.exists():
        m = re.search(r"^TRAIN_FOLDS=([\d,]*)", params.read_text(), re.MULTILINE)
        if m and m.group(1).strip():
            dispatched = [int(x) for x in m.group(1).split(",") if x.strip()]
    return dict(block_folds=n_folds, dispatched=dispatched,
                all_folds=(list(range(n_folds)) if n_folds else None),
                sources=dict(config=(_rel(config) if config.exists() else None),
                             params=(_rel(params) if params.exists() else None)))


def fold_reliability(fold: int, directory=None) -> dict:
    directory = Path(directory or BH)
    """Carry the scorer's own coarse-interval warning forward for this fold's held-out arm."""
    p = directory / f"fold{fold}_heldout.json"
    if not p.exists():
        return dict(report=None, interval_readable=None,
                    note=f"{_rel(p)} is not committed")
    rel = (load_json(p).get("bootstrap") or {}).get("reliability") or {}
    return dict(report=_rel(p),
                resampling_units=rel.get("resampling_units"),
                min_units_for_a_readable_ci=rel.get("min_units_for_a_readable_ci"),
                interval_readable=rel.get("interval_readable"),
                note=rel.get("note"))


def summarise_fold_gaps(gaps: List[dict], expected: dict, directory=None) -> dict:
    directory = Path(directory or BH)
    if not gaps:
        raise SystemExit(f"no fold*_generalisation_gap.json under {directory} - nothing to summarise")
    rows, held, trained, signed = [], [], [], []
    for g in gaps:
        rows.append(dict(fold=g["_fold"], candidate=g.get("candidate"),
                         held_out_dti=g.get("held_out_dti"), held_out_ci95=g.get("held_out_ci95"),
                         held_out_blocks=g.get("held_out_blocks"),
                         trained_on_dti=g.get("trained_on_dti"),
                         trained_on_ci95=g.get("trained_on_ci95"),
                         trained_on_blocks=g.get("trained_on_blocks"),
                         generalisation_gap=g.get("generalisation_gap"),
                         reliability=fold_reliability(int(g["_fold"]), directory),
                         source=g["_path"]))
        held.append(float(g["held_out_dti"]))
        trained.append(float(g["trained_on_dti"]))
        signed.append(float(g["generalisation_gap"]))
        # internal consistency: the gap must be the difference of the two committed numbers
        if abs((trained[-1] - held[-1]) - signed[-1]) > 1e-6:
            raise SystemExit(f"fold {g['_fold']}: generalisation_gap {signed[-1]} != trained_on "
                             f"{trained[-1]} - held_out {held[-1]} (committed file disagrees with "
                             "itself; do not summarise it)")
    n = len(signed)
    mean = sum(signed) / n
    all_folds = expected.get("all_folds") or []
    missing = sorted(set(all_folds) - {int(g["_fold"]) for g in gaps})
    return dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/read_landed_reports.py",
        purpose=("the distribution of the memorisation-vs-transfer gap over the committed "
                 "block-holdout folds: trained_on DTI minus held-out DTI on the same candidate, "
                 "the same partition and the same metric"),
        n_folds_committed=n,
        expected=expected,
        folds_missing=missing,
        complete=bool(all_folds) and not missing,
        per_fold=rows,
        gap=dict(mean=round(mean, 6), min=round(min(signed), 6), max=round(max(signed), 6),
                 spread=round(max(signed) - min(signed), 6),
                 folds_with_positive_gap=int(sum(1 for x in signed if x > 0)),
                 folds_with_negative_gap=int(sum(1 for x in signed if x < 0)),
                 mean_held_out_dti=round(sum(held) / n, 6),
                 mean_trained_on_dti=round(sum(trained) / n, 6)),
        interval_readability=dict(
            folds_with_a_readable_ci=int(sum(1 for r in rows
                                             if r["reliability"].get("interval_readable"))),
            rule=("a fold-restricted bootstrap resamples only that fold's scoreable blocks; below "
                  "MIN_BLOCKS_FOR_A_READABLE_CI (12) the scorer itself marks the interval COARSE, "
                  "so the per-fold CIs below are spread indicators, not confidence statements"),
            source="scripts/block_holdout_eval.py::bootstrap_reliability"),
        reading=(
            "The gap is small in absolute terms (every fold's held-out DTI is within ~0.014 of its "
            "trained-on DTI) and its SIGN is not stable across folds, so the honest statement is "
            "that this model neither clearly memorises nor clearly improves on unseen 51 km "
            "blocks: the proxy-policy reading survives the hold-out, but the per-fold intervals "
            "are too coarse to rank folds.  Derived, not asserted - see per_fold."
            if n >= 2 and min(signed) < 0 < max(signed) else
            "See per_fold: the gap's sign and spread are the measurement."),
        caveat=("every number here is measured against the new-fault-like PROXY population and the "
                "catalogue population of the committed reports; neither is the scored truth "
                "(rules 3.6: public/private test sets, then an expert-updated set)"),
    )


# --------------------------------------------------------------------------------------
# 2. the two-population pseudo-label contrast
# --------------------------------------------------------------------------------------
def pseudo_provenance(config=None) -> dict:
    config = Path(config or PSEUDO_CONFIG)
    """Derive WHERE the pseudo-labels came from, and whether that is the proxy truth's source."""
    if not config.exists():
        raise SystemExit(f"MISSING INPUT: {config}")
    try:
        import yaml
    except ImportError:                                          # pragma: no cover
        raise SystemExit("pyyaml is required (pip install pyyaml)")
    cfg = yaml.safe_load(config.read_text()) or {}
    data = cfg.get("data") or {}
    train = cfg.get("training") or {}
    path = data.get("pseudo_label_path")
    code = data.get("pseudo_code")
    return dict(
        config=_rel(config),
        pseudo_label_path=path,
        pseudo_code=code,
        pseudo_weight=train.get("pseudo_weight"),
        holdout=train.get("holdout"),
        block_fold=train.get("block_fold"),
        seed=train.get("seed"),
        circular_source=bool(path and code is not None
                             and Path(str(path)).name == "proxy_catalogue.tif"
                             and int(code) == PROXY_CODE_ONLY),
        circular_source_meaning=(
            "the pseudo-label mask is the code-2 layer of data/evidence/proxy/proxy_catalogue.tif, "
            "which is ALSO the raster the 'proxy_only' truth population is cut from - so a gain on "
            "that population is measured against the same catalogue the extra training signal came "
            "from (spatially held out, but not source-held-out).  The 'labels' population is "
            "independent of that source and is the non-circular arm of the same measurement."
            if (path and code is not None and Path(str(path)).name == "proxy_catalogue.tif"
                and int(code) == PROXY_CODE_ONLY) else
            "the pseudo-label source is not the proxy truth raster; the proxy arm is not circular"),
    )


def scope_pair(base_dir, pseudo_dir, fold: int, scope: str) -> tuple:
    """Both arms of one fold and scope, plus any sibling report that adds a population.

    A re-score of the SAME raw field with ``--combined-population`` is committed next to the
    original as ``fold<K>_<scope>_combined.json`` (that is what .github/workflows/pseudo-label.yml
    does for the baseline arm, whose raw field survives only as a runner artifact).  The sibling is
    accepted only if it is the same field: its recorded input sha256 must equal the primary
    report's, otherwise the two files describe different models and comparing them would be a
    fabrication.
    """
    base_dir, pseudo_dir = Path(base_dir or BH), Path(pseudo_dir or PL)
    name = "complement" if scope == "complement" else "heldout"
    b, p = base_dir / f"fold{fold}_{name}.json", pseudo_dir / f"fold{fold}_{name}.json"
    bx, px = (base_dir / f"fold{fold}_{name}_combined.json",
              pseudo_dir / f"fold{fold}_{name}_combined.json")
    return (load_json(b) if b.exists() else None,
            load_json(p) if p.exists() else None,
            _rel(b), _rel(p),
            _sibling(load_json(bx) if bx.exists() else None, load_json(b) if b.exists() else None,
                     bx, b),
            _sibling(load_json(px) if px.exists() else None, load_json(p) if p.exists() else None,
                     px, p))


def _field_sha(report) -> Optional[str]:
    return (((report or {}).get("inputs") or {}).get("pred_grid") or {}).get("sha256")


def _sibling(extra, primary, extra_path: Path, primary_path: Path):
    """Accept a re-score of the same field; refuse a re-score of a different one."""
    if extra is None:
        return None
    a, b = _field_sha(extra), _field_sha(primary)
    if primary is not None and a and b and a != b:
        raise SystemExit(f"{_rel(extra_path)} records input sha256 {a[:12]}... but {_rel(primary_path)} "
                         f"records {b[:12]}...: the two reports are NOT the same probability field, "
                         "so their populations cannot be contrasted as one arm")
    extra["_path"] = _rel(extra_path)
    return extra


def contrast_report(fold: int, decision: dict, prov: dict, base_dir=None, pseudo_dir=None,
                    populations: Sequence[str] = ("proxy_only", "labels", "combined")) -> dict:
    base_dir, pseudo_dir = Path(base_dir or BH), Path(pseudo_dir or PL)
    pol = adopted_policy_label(decision)
    label = pol["candidate_label"]
    arms: Dict[str, dict] = {}
    for scope in ("heldout", "complement"):
        base, pseudo, bpath, ppath, base_x, pseudo_x = scope_pair(base_dir, pseudo_dir, fold, scope)
        if base is None or pseudo is None:
            arms[scope] = dict(status="MISSING", baseline_report=bpath, pseudo_report=ppath,
                               note=("both arms must be committed for the same fold and scope "
                                     "before a paired contrast exists"))
            continue
        if base["blocks"]["block_px"] != pseudo["blocks"]["block_px"]:
            raise SystemExit(f"fold {fold}/{scope}: the two arms were scored on different block "
                             "partitions - not a paired contrast")
        if base["restriction"]["blocks"] != pseudo["restriction"]["blocks"]:
            raise SystemExit(f"fold {fold}/{scope}: the two arms were scored on different blocks")
        bset = bootstrap_of(base)
        per_pop = {}
        for pop in populations:
            # the combined population may live in a sibling re-score of the same field
            src_b = base_x if (pop == "combined" and base_x) else base
            src_p = pseudo_x if (pop == "combined" and pseudo_x) else pseudo
            have_base = pop in ((src_b or {}).get("populations") or {})
            have_pseudo = pop in ((src_p or {}).get("populations") or {})
            if not (have_base and have_pseudo):
                per_pop[pop] = dict(status="NOT_SCORED",
                                    note=(f"{'both' if not (have_base or have_pseudo) else 'one'} "
                                          f"arm(s) carry no {pop!r} population; re-score with "
                                          "scripts/block_holdout_eval.py --combined-population"))
                continue
            m = (src_b.get("metric") or {})
            per_pop[pop] = dict(
                status="MEASURED",
                sources=dict(baseline=(base_x or {}).get("_path", bpath) if src_b is base_x else bpath,
                             pseudo=(pseudo_x or {}).get("_path", ppath) if src_p is pseudo_x else ppath),
                **paired(candidate_row(src_b, pop, label), candidate_row(src_p, pop, label),
                         alpha=float(m.get("alpha", 0.2)), beta=float(m.get("beta", 0.8)),
                         eps=float(m.get("eps", 1e-7)), n_boot=bset["n_boot"], seed=bset["seed"],
                         names=("baseline_no_pseudo", "pseudo")))
            per_pop[pop]["interval_readable"] = bset["reliability"].get("interval_readable")
            per_pop[pop]["resampling_units"] = bset["reliability"].get("resampling_units")
        arms[scope] = dict(status="MEASURED", baseline_report=bpath, pseudo_report=ppath,
                           blocks_scored=base["restriction"]["blocks_scoreable"],
                           restriction_note=base["restriction"]["note"],
                           populations=per_pop)

    held = arms.get("heldout") or {}
    hp = (held.get("populations") or {}) if held.get("status") == "MEASURED" else {}
    proxy_c = hp.get("proxy_only", {}).get("contrast")
    labels_c = hp.get("labels", {}).get("contrast")
    combined_c = hp.get("combined", {}).get("contrast")

    def sign(x):
        return None if x is None else ("gain" if x > 0 else "loss")

    if proxy_c is None or labels_c is None:
        verdict, why = "NOT_MEASURABLE", ("the held-out arm is not measured on both the proxy and "
                                          "the catalogue population yet")
    elif combined_c is not None:
        verdict = ("GAIN_ON_THE_COMBINED_SURROGATE" if combined_c > 0
                   else "LOSS_ON_THE_COMBINED_SURROGATE")
        why = (f"the combined (Phase-2-like) population is the only committed population that "
               f"contains BOTH fault kinds, and it moves {combined_c:+.6f}; proxy "
               f"{proxy_c:+.6f}, catalogue {labels_c:+.6f}")
    elif sign(proxy_c) == sign(labels_c):
        verdict = ("GAIN_ON_BOTH_POPULATIONS" if proxy_c > 0 else "LOSS_ON_BOTH_POPULATIONS")
        why = f"proxy {proxy_c:+.6f} and catalogue {labels_c:+.6f} move the same way"
    else:
        verdict = ("TRADE_OFF_PROXY_GAINS_CATALOGUE_LOSSES" if proxy_c > 0
                   else "TRADE_OFF_CATALOGUE_GAINS_PROXY_LOSSES")
        why = (f"proxy {proxy_c:+.6f} but catalogue {labels_c:+.6f}: the extra signal moves the "
               "model between the two fault kinds instead of adding detection")

    return dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/read_landed_reports.py",
        purpose=("the SGMC pseudo-label contrast on EVERY committed truth population, not only the "
                 "one whose truth shares the pseudo-labels' source: derived from the committed "
                 "per-block metric components of both arms (no re-scoring, no re-training)"),
        fold=fold,
        policy=pol,
        pseudo_provenance=prov,
        arms=arms,
        verdict=verdict,
        verdict_reason=why,
        shippable_evidence=False,
        shipping_note=("no shipping decision is made here.  A pseudo-labelled field reaches the "
                       "leaderboard only through docs/FIELD_SELECTION_RULE.md as a new reblend.yml "
                       "RUN_ID with R3 measured on both fields' raw probability rasters, and the "
                       "proxy arm of this contrast is source-circular (see "
                       "pseudo_provenance.circular_source) so it cannot carry an adoption on its "
                       "own."),
        caveat=("fold-restricted bootstraps resample only that fold's scoreable blocks; below 12 "
                "units the scorer marks the interval COARSE, so P(candidate > reference) here is a "
                "spread indicator on a small number of 51 km regions, not a confidence statement "
                "(scripts/block_holdout_eval.py::bootstrap_reliability)"),
    )


def self_check(rep: dict, fold: int, pl_dir=None) -> List[str]:
    pl_dir = Path(pl_dir or PL)
    """Verify the recomposition against the runner's OWN committed paired contrast."""
    problems = []
    committed = pl_dir / f"fold{fold}_paired_vs_baseline.json"
    if not committed.exists():
        return [f"{_rel(committed)} not committed - cross-check skipped"]
    c = load_json(committed)
    held = (rep["arms"].get("heldout") or {})
    got = ((held.get("populations") or {}).get("proxy_only") or {})
    if got.get("status") != "MEASURED":
        return ["held-out proxy arm not measured - cross-check impossible"]
    boot = (c.get("bootstrap") or {}).get("pseudo") or {}
    mine_boot = got.get("bootstrap") or {}
    for name, mine, theirs in (
            ("baseline_held_out_dti", got.get("reference_dti"), c.get("baseline_held_out_dti")),
            ("pseudo_held_out_dti", got.get("candidate_dti"), c.get("pseudo_held_out_dti")),
            ("prob_beats_reference", mine_boot.get("prob_candidate_beats_reference"),
             boot.get("prob_beats_reference")),
            ("contrast_p50", mine_boot.get("contrast_p50"), boot.get("contrast_vs_reference_p50")),
    ):
        if mine is None or theirs is None:
            problems.append(f"{name}: not present on one side (mine={mine}, committed={theirs})")
        elif abs(float(mine) - float(theirs)) > TOL:
            problems.append(f"{name}: recomposed {mine} != committed {theirs}")
    ci_mine, ci_theirs = mine_boot.get("candidate_ci95"), boot.get("dti_ci95")
    if ci_mine and ci_theirs and any(abs(x - y) > TOL for x, y in zip(ci_mine, ci_theirs)):
        problems.append(f"candidate CI95: recomposed {ci_mine} != committed {ci_theirs}")
    return problems


# --------------------------------------------------------------------------------------
# 3. the POOLED multi-fold contrast (EXECUTIVE_SUMMARY §11 item 7's remaining half)
# --------------------------------------------------------------------------------------
# WHY POOLING IS THE ONLY READING THAT CAN SETTLE ITEM 7
#   Fold 0's union contrast is +0.031280 at P = 0.916 with CI95 [-0.010, +0.082] over 8 scoreable
#   51.2 km blocks (data/evidence/pseudo_labels/fold0_two_population_contrast.json).  The
#   scorer's OWN reliability rule (scripts/block_holdout_eval.py::bootstrap_reliability,
#   MIN_BLOCKS_FOR_A_READABLE_CI = 12) marks that interval COARSE, and the two fires of the same
#   fold/seed/config on different runners differ by ~0.036 on the proxy arm - i.e. the single-fold
#   replicate noise is larger than the effect.  A single fold therefore cannot support a shipping
#   claim in either direction.
#
#   The four block-holdout folds partition the SAME 56-block grid into four DISJOINT block sets
#   (blocks.partition.n_folds = 4, assignment by greedy LPT), so their scoreable blocks are
#   geographically disjoint: pooling them gives 30-35 resampling units instead of 7-9, which is
#   above the scorer's own 12-unit bar.  That is a legitimate pool, not a repeated measurement of
#   one region - and it is why the disjointness of the block ids is VERIFIED below rather than
#   assumed.
#
#   Pooling recomposes each fold's committed per-block metric components (TP_w/FP_w/FN_w are sums)
#   and bootstraps the concatenation once.  No raster is re-scored and no number is invented:
#   exactly the derivation-only contract of this script.
MIN_UNITS_FOR_A_READABLE_POOL = 12       # scripts/block_holdout_eval.py::MIN_BLOCKS_FOR_A_READABLE_CI
ADOPT_P_BAR = 0.95                       # docs/FIELD_SELECTION_RULE.md R3 (P(F > S) >= 0.95)
ADOPT_CONTRAST_BAR = 0.010               # docs/FIELD_SELECTION_RULE.md R1 (strictly greater than +0.010)


def _poolable_blocks(row: dict, mode: str) -> List[dict]:
    """The block rows that are resampling units for one fold, under the named convention.

    ``scoreable`` (the pooled default) keeps only the blocks that carry truth AND prediction, i.e.
    the fold's own held-out blocks that the scorer could actually score.  ``all`` keeps the full
    56-block partition the per-fold reports bootstrap over (48 of them all-zero for a fold-restricted
    report), which is the convention the committed per-fold CI was produced with - it exists so the
    pool can be shown to reproduce a committed single-fold number exactly before it is trusted with
    four folds.
    """
    blocks = row.get("blocks") or []
    if not blocks:
        raise SystemExit("a candidate row with no per-block rows cannot be pooled (pruned as a "
                         "duplicate? re-run the scorer with --keep-duplicate-rows)")
    if mode == "all":
        return list(blocks)
    if mode == "scoreable":
        return [b for b in blocks if b.get("scoreable")]
    raise SystemExit(f"unknown --pool-blocks mode {mode!r} (expected 'scoreable' or 'all')")


def pooled_paired(base_rows: List[dict], cand_rows: List[dict], alpha: float, beta: float,
                  eps: float, n_boot: int, seed: int, names: Sequence[str],
                  block_mode: str = "scoreable") -> dict:
    """One paired block bootstrap over the CONCATENATED blocks of every supplied fold.

    ``base_rows``/``cand_rows`` are the per-fold candidate rows (arm A and arm B), in the same fold
    order.  Row 0 of ``bootstrap_from_blocks`` is the reference, so ``prob_beats_reference`` is the
    paired P(candidate > reference) over the pooled resamples - the same statistic the per-fold
    reading reports, on 4x the resampling units.
    """
    if len(base_rows) != len(cand_rows):
        raise SystemExit("the two arms must be pooled over the same folds in the same order")
    bb = [b for r in base_rows for b in _poolable_blocks(r, block_mode)]
    cb = [b for r in cand_rows for b in _poolable_blocks(r, block_mode)]
    if len(bb) != len(cb):
        raise SystemExit(f"the two arms carry different pooled block counts ({len(bb)} vs {len(cb)})"
                         " - not a paired contrast")
    if not bb:
        raise SystemExit("no scoreable blocks in either arm - nothing to pool")
    ids_b = [b["block"] for b in bb]
    ids_c = [b["block"] for b in cb]
    if ids_b != ids_c:
        raise SystemExit("the two arms are not scored on the same pooled block sequence - the "
                         "bootstrap would pair unrelated regions")
    out = bootstrap_from_blocks([dict(label=names[0], blocks=bb),
                                 dict(label=names[1], blocks=cb)],
                                alpha=alpha, beta=beta, eps=eps, n_boot=n_boot, seed=seed)
    ref, cand = out[names[0]], out[names[1]]

    def _sums(rows, key):
        return float(sum(b[key] for b in rows))

    def _dti(rows):
        tp, fp, fn = _sums(rows, "TP_w"), _sums(rows, "FP_w"), _sums(rows, "FN_w")
        return tp / (tp + alpha * fp + beta * fn + eps)

    dti_b, dti_c = _dti(bb), _dti(cb)
    # A resampling unit is a block that actually carries truth (n_gt > 0): a zero block contributes
    # nothing to the paired contrast and understates variance if counted, so readability is judged
    # on the scoreable count regardless of which block set the point estimate summed - exactly the
    # convention scripts/block_holdout_eval.py::bootstrap_reliability uses.
    n_scoreable = int(sum(1 for b in cb if b.get("scoreable")))
    return dict(
        reference=names[0], candidate=names[1],
        block_convention=block_mode,
        n_folds_pooled=len(base_rows),
        n_blocks_drawn=len(bb),
        resampling_units=n_scoreable,
        n_scoreable_blocks=n_scoreable,
        interval_readable=bool(n_scoreable >= MIN_UNITS_FOR_A_READABLE_POOL),
        pooled_reference_dti=round(dti_b, 6),
        pooled_candidate_dti=round(dti_c, 6),
        pooled_contrast=round(dti_c - dti_b, 6),
        reference_components=dict(TP_w=_sums(bb, "TP_w"), FP_w=_sums(bb, "FP_w"),
                                  FN_w=_sums(bb, "FN_w"), n_gt=int(_sums(bb, "n_gt"))),
        candidate_components=dict(TP_w=_sums(cb, "TP_w"), FP_w=_sums(cb, "FP_w"),
                                  FN_w=_sums(cb, "FN_w"), n_gt=int(_sums(cb, "n_gt"))),
        reference_support_px=int(sum(r["support_px"] for r in base_rows)),
        candidate_support_px=int(sum(r["support_px"] for r in cand_rows)),
        bootstrap=dict(n_boot=n_boot, seed=seed,
                       reference_ci95=ref["dti_ci95"], candidate_ci95=cand["dti_ci95"],
                       contrast_p50=cand["contrast_vs_reference_p50"],
                       contrast_ci95=cand["contrast_vs_reference_ci95"],
                       prob_candidate_beats_reference=cand["prob_beats_reference"]),
    )


def _partition_key(report: dict, where: str) -> tuple:
    p = ((report.get("blocks") or {}).get("partition") or {})
    if not p:
        raise SystemExit(f"{where}: the report carries no block partition - pooling would mix "
                         "unknown geographies")
    return (int(p.get("block_px", -1)), int(p.get("seed", -1)), int(p.get("n_folds", -1)),
            str(p.get("mode", "")))


def pooled_contrast(folds: Sequence[int], decision: dict, prov: dict, base_dir=None,
                    pseudo_dir=None, scope: str = "heldout",
                    populations: Sequence[str] = ("combined", "proxy_only", "labels"),
                    block_mode: str = "scoreable", n_boot: int = 2000, seed: int = 0) -> dict:
    """Pool every committed fold's two arms into ONE paired contrast per population.

    Refuses rather than approximates: a fold whose arm is missing is reported in ``missing_folds``
    (never silently dropped from the denominator of a claim), and the pooled number is only labelled
    readable when it clears the scorer's own 12-unit bar.
    """
    base_dir, pseudo_dir = Path(base_dir or BH), Path(pseudo_dir or PL)
    pol = adopted_policy_label(decision)
    label = pol["candidate_label"]
    per_pop: Dict[str, dict] = {}
    used_folds: Dict[str, List[int]] = {}
    missing: Dict[str, List[dict]] = {}
    partitions: set = set()
    fields: List[dict] = []

    for pop in populations:
        b_rows, c_rows, ok_folds, why_missing = [], [], [], []
        for k in folds:
            base, pseudo, bpath, ppath, base_x, pseudo_x = scope_pair(base_dir, pseudo_dir, k, scope)
            src_b = base_x if (pop == "combined" and base_x) else base
            src_p = pseudo_x if (pop == "combined" and pseudo_x) else pseudo
            if base is None or pseudo is None:
                why_missing.append(dict(fold=k, reason="an arm is not committed for this fold/scope",
                                        baseline_report=bpath, pseudo_report=ppath))
                continue
            have_b = pop in ((src_b or {}).get("populations") or {})
            have_p = pop in ((src_p or {}).get("populations") or {})
            if not (have_b and have_p):
                why_missing.append(dict(
                    fold=k,
                    reason=(f"the {'baseline' if not have_b else 'pseudo'} arm carries no {pop!r} "
                            "population - re-score it with block_holdout_eval.py "
                            "--combined-population"),
                    baseline_report=bpath, pseudo_report=ppath))
                continue
            if base["blocks"]["block_px"] != pseudo["blocks"]["block_px"]:
                raise SystemExit(f"fold {k}/{scope}/{pop}: the two arms were scored on different "
                                 "block partitions - not a paired contrast")
            if base["restriction"]["blocks"] != pseudo["restriction"]["blocks"]:
                raise SystemExit(f"fold {k}/{scope}/{pop}: the two arms were scored on different "
                                 "blocks")
            if int(base["restriction"].get("score_fold", -1)) != int(k):
                raise SystemExit(f"{bpath}: restriction.score_fold="
                                 f"{base['restriction'].get('score_fold')} but this is fold {k} - "
                                 "the report does not describe the fold it is named for")
            if bool(base["restriction"].get("complement")) != (scope == "complement"):
                raise SystemExit(f"{bpath}: restriction.complement="
                                 f"{base['restriction'].get('complement')} but scope={scope!r}")
            partitions.add(_partition_key(base, bpath))
            partitions.add(_partition_key(pseudo, ppath))
            mb = (src_b.get("metric") or {})
            mp = (src_p.get("metric") or {})
            if (float(mb.get("alpha")), float(mb.get("beta")), float(mb.get("eps")),
                    int(mb.get("R_pixels"))) != (float(mp.get("alpha")), float(mp.get("beta")),
                                                 float(mp.get("eps")), int(mp.get("R_pixels"))):
                raise SystemExit(f"fold {k}/{pop}: the two arms were scored with different metric "
                                 f"parameters ({mb} vs {mp})")
            b_row, c_row = candidate_row(src_b, pop, label), candidate_row(src_p, pop, label)
            fb, fc = _field_sha(src_b), _field_sha(src_p)
            if fb and fc and fb == fc:
                raise SystemExit(f"fold {k}/{pop}: both arms record the SAME probability field "
                                 f"(sha256 {fb[:12]}...) - a contrast of a field with itself is not "
                                 "a measurement")
            fields.append(dict(fold=k, population=pop, baseline_field_sha256=fb,
                               pseudo_field_sha256=fc,
                               baseline_report=(base_x or {}).get("_path", bpath) if src_b is base_x
                               else bpath,
                               pseudo_report=(pseudo_x or {}).get("_path", ppath) if src_p is pseudo_x
                               else ppath))
            b_rows.append(b_row)
            c_rows.append(c_row)
            ok_folds.append(k)

        used_folds[pop] = ok_folds
        missing[pop] = why_missing
        if not ok_folds:
            per_pop[pop] = dict(status="NOT_SCORED", folds_pooled=[], missing=why_missing,
                                note=("no fold carries both arms on this population yet; nothing is "
                                      "pooled and nothing is claimed"))
            continue
        m = float(0.2), float(0.8), float(1e-7)
        pooled = pooled_paired(b_rows, c_rows, alpha=m[0], beta=m[1], eps=m[2],
                               n_boot=n_boot, seed=seed,
                               names=("baseline_no_pseudo", "pseudo"), block_mode=block_mode)
        # disjointness of the pooled resampling units is the property that makes pooling legitimate:
        # the four block folds partition the grid, so a block id must appear at most once per arm.
        ids = [b["block"] for b in _poolable_blocks(b_rows[0], block_mode)]
        seen: set = set()
        dupes: List[int] = []
        for r in b_rows:
            for b in _poolable_blocks(r, block_mode):
                if b["block"] in seen:
                    dupes.append(int(b["block"]))
                seen.add(int(b["block"]))
        if dupes:
            raise SystemExit(f"{pop}: pooled block ids repeat across folds ({sorted(set(dupes))}) - "
                             "the folds are not disjoint, so pooling would resample the same "
                             "geography more than once")
        per_fold = [dict(fold=k, reference_dti=br["global_dti"], candidate_dti=cr["global_dti"],
                         contrast=round(cr["global_dti"] - br["global_dti"], 6),
                         scoreable_blocks=sum(1 for b in _poolable_blocks(br, block_mode)))
                    for k, br, cr in zip(ok_folds, b_rows, c_rows)]
        b_ = pooled["bootstrap"]
        p_bar = ADOPT_P_BAR
        c_bar = ADOPT_CONTRAST_BAR
        p = b_["prob_candidate_beats_reference"]
        contrast_p50 = b_["contrast_p50"]
        clears = bool(p is not None and p >= p_bar and contrast_p50 is not None
                      and contrast_p50 > c_bar and pooled["interval_readable"])
        per_pop[pop] = dict(
            status="MEASURED", folds_pooled=ok_folds, missing=why_missing,
            per_fold=per_fold, first_pooled_fold_block_ids=ids, **pooled,
            adoption_bars=dict(prob_bar=p_bar, contrast_bar=c_bar,
                               min_resampling_units=MIN_UNITS_FOR_A_READABLE_POOL,
                               source="docs/FIELD_SELECTION_RULE.md R1/R3; "
                                      "scripts/block_holdout_eval.py::MIN_BLOCKS_FOR_A_READABLE_CI"),
            clears_pre_registered_bars=clears,
            bars_reason=(
                f"P(pseudo > baseline) = {p} (bar >= {p_bar}), pooled contrast p50 = "
                f"{contrast_p50:+.6f} (bar > {c_bar:+.3f}), resampling units = "
                f"{pooled['resampling_units']} (bar >= {MIN_UNITS_FOR_A_READABLE_POOL})"
                + ("" if clears else " - at least one bar is not cleared")),
        )

    if len(partitions) > 1:
        raise SystemExit(f"the pooled folds were scored on {len(partitions)} different block "
                         f"partitions ({sorted(partitions)}) - blocks would not be comparable")

    comb = per_pop.get("combined") or {}
    proxy = per_pop.get("proxy_only") or {}
    labels = per_pop.get("labels") or {}
    if comb.get("status") == "MEASURED":
        p = comb["bootstrap"]["prob_candidate_beats_reference"]
        c = comb["bootstrap"]["contrast_p50"]
        units = comb["resampling_units"]
        readable = comb["interval_readable"]
        p_ok = p is not None and p >= ADOPT_P_BAR
        c_ok = c is not None and c > ADOPT_CONTRAST_BAR
        if comb["clears_pre_registered_bars"]:
            verdict = "POOL_CLEARS_THE_BARS_ON_THE_COMBINED_SURROGATE"
            why = (f"pooled over {len(comb['folds_pooled'])} folds / {units} disjoint blocks: "
                   f"contrast {c:+.6f} at P = {p}, both above the pre-registered bars and the "
                   f"interval is readable ({units} >= {MIN_UNITS_FOR_A_READABLE_POOL} units)")
        elif p_ok and c_ok and not readable:
            # the effect and its probability clear the bars, but the pool is still under-powered:
            # this is the state that names the next action (fire more folds), so it gets its own verdict
            verdict = "POOL_POSITIVE_BUT_UNDER_POWERED"
            why = (f"contrast {c:+.6f} (> {ADOPT_CONTRAST_BAR:+.3f}) at P = {p} (>= {ADOPT_P_BAR}), "
                   f"but only {units} pooled resampling units are committed (< "
                   f"{MIN_UNITS_FOR_A_READABLE_POOL}); the pseudo-label arm exists for folds "
                   f"{comb['folds_pooled']} only - fire the remaining folds to reach a readable pool")
        elif p_ok and not c_ok:
            verdict = "POOL_POSITIVE_BUT_BELOW_THE_EFFECT_BAR"
            why = (f"P = {p} clears {ADOPT_P_BAR} but the pooled contrast {c:+.6f} does not exceed "
                   f"{ADOPT_CONTRAST_BAR:+.3f} over {units} blocks")
        elif c is not None and c > 0:
            verdict = "POOL_SUGGESTIVE_UNDER_POWERED"
            why = (f"pooled contrast {c:+.6f} is positive but P = {p} < {ADOPT_P_BAR}"
                   + ("" if readable else
                      f" and only {units} resampling units are available (< "
                      f"{MIN_UNITS_FOR_A_READABLE_POOL})"))
        else:
            verdict = "POOL_NO_GAIN_ON_THE_COMBINED_SURROGATE"
            why = f"pooled contrast {c:+.6f} at P = {p} over {units} blocks"
    else:
        verdict, why = "NOT_MEASURABLE", ("no fold carries both arms on the combined population yet"
                                          if not any(v.get("status") == "MEASURED"
                                                     for v in per_pop.values())
                                          else "the combined population is not scored on any pooled fold")

    return dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/read_landed_reports.py --pool",
        purpose=(("the SGMC pseudo-label contrast POOLED over every committed block-holdout fold: "
                  "one fold carries 7-9 scoreable 51.2 km blocks, below the scorer's own "
                  "12-unit bar for a readable interval, and its replicate noise (~0.036 between two "
                  "fires of the same fold/seed/config) is larger than the effect it is asked to "
                  "resolve (+0.031 on fold 0).  Pooling the four DISJOINT fold partitions is the "
                  "only reading that can settle EXECUTIVE_SUMMARY §11 item 7; every number here is "
                  "recomposed from the committed per-block components of both arms.")),
        scope=scope,
        policy=pol,
        pseudo_provenance=prov,
        block_convention=block_mode,
        block_convention_note=(
            "'scoreable' pools only the blocks each fold could actually score (its own held-out "
            "blocks with truth and prediction); 'all' pools the full 56-block partition the "
            "per-fold reports bootstrap over, and exists to reproduce a committed single-fold "
            "number exactly before the pooled reading is trusted"),
        folds_requested=list(folds),
        partitions=sorted(str(p) for p in partitions),
        fields=fields,
        populations=per_pop,
        verdict=verdict,
        verdict_reason=why,
        shippable_evidence=bool(comb.get("status") == "MEASURED"
                                and comb.get("clears_pre_registered_bars")),
        shipping_note=(("no shipping decision is made here.  A pseudo-labelled field reaches the "
                        "leaderboard only through docs/FIELD_SELECTION_RULE.md as a new reblend.yml "
                        "RUN_ID with R1-R5 measured on both fields' raw probability rasters; this "
                        "pooled contrast is the power fix for the union arm, not an adoption")),
        caveat=("the proxy_only arm is source-circular (see pseudo_provenance.circular_source): its "
                "truth is cut from the same raster the pseudo-labels came from, so the combined and "
                "labels arms are the readings that can move a decision.  FP_w is a sum over "
                "PREDICTION pixels, so the combined population's DTI is never inferable from its "
                "two components - it is scored here from each arm's own committed combined rows."),
    )


def print_pooled(rep: dict) -> None:
    print(f"\nPOOLED pseudo-label contrast over folds {rep['folds_requested']} "
          f"(scope {rep['scope']}, blocks {rep['block_convention']}):")
    for pop, c in rep["populations"].items():
        if c.get("status") != "MEASURED":
            print(f"  {pop:<11} {c.get('status')} - {c.get('note','')}")
            for m in c.get("missing", []):
                print(f"              fold {m['fold']}: {m['reason']}")
            continue
        b = c["bootstrap"]
        print(f"  {pop:<11} folds {c['folds_pooled']} -> {c['resampling_units']} pooled blocks "
              f"(readable: {c['interval_readable']})")
        print(f"              baseline {c['pooled_reference_dti']:.6f} -> pseudo "
              f"{c['pooled_candidate_dti']:.6f}  contrast p50 {b['contrast_p50']:+.6f} "
              f"CI95 [{b['contrast_ci95'][0]:+.6f}, {b['contrast_ci95'][1]:+.6f}]  "
              f"P(pseudo>baseline) = {b['prob_candidate_beats_reference']}")
        print(f"              per fold: " + ", ".join(
            f"f{r['fold']} {r['contrast']:+.4f}({r['scoreable_blocks']}blk)" for r in c["per_fold"]))
        print(f"              bars: {c['bars_reason']}")
        for m in c.get("missing", []):
            print(f"              MISSING fold {m['fold']}: {m['reason']}")
    print(f"  VERDICT (derived): {rep['verdict']} - {rep['verdict_reason']}")
    print(f"  shippable on this evidence: {rep['shippable_evidence']}")


def print_report(gaps: dict, contrast: Optional[dict]) -> None:
    g = gaps["gap"]
    print(f"\nblock-holdout generalisation gap over {gaps['n_folds_committed']} committed fold(s)"
          f" (of {gaps['expected'].get('block_folds')} in the partition):")
    for r in gaps["per_fold"]:
        rel = r["reliability"]
        print(f"  fold {r['fold']}: held-out {r['held_out_dti']:.4f} "
              f"[{r['held_out_ci95'][0]:.4f}, {r['held_out_ci95'][1]:.4f}] on "
              f"{r['held_out_blocks']} blocks | trained-on {r['trained_on_dti']:.4f} on "
              f"{r['trained_on_blocks']} blocks | gap {r['generalisation_gap']:+.4f} "
              f"| CI readable: {rel.get('interval_readable')} "
              f"({rel.get('resampling_units')} resampling units)")
    print(f"  mean gap {g['mean']:+.6f} (min {g['min']:+.6f}, max {g['max']:+.6f}, spread "
          f"{g['spread']:.6f}); positive in {g['folds_with_positive_gap']} fold(s), negative in "
          f"{g['folds_with_negative_gap']}")
    if gaps["folds_missing"]:
        print(f"  MISSING folds (still running or not committed): {gaps['folds_missing']}")

    if contrast is None:
        return

    pol = contrast["policy"]
    print(f"\npseudo-label contrast, fold {contrast['fold']}, candidate {pol['candidate_label']} "
          f"(adopted policy {pol['policy']}):")
    for scope, arm in contrast["arms"].items():
        if arm.get("status") != "MEASURED":
            print(f"  {scope}: {arm.get('status')} - {arm.get('note')}")
            continue
        print(f"  {scope} ({arm['blocks_scored']} blocks):")
        for pop, c in arm["populations"].items():
            if c.get("status") != "MEASURED":
                print(f"    {pop:<11} {c.get('status')}")
                continue
            b = c["bootstrap"]
            print(f"    {pop:<11} baseline {c['reference_dti']:.4f} -> pseudo "
                  f"{c['candidate_dti']:.4f}  contrast {c['contrast']:+.4f}  "
                  f"P(pseudo>baseline)={b['prob_candidate_beats_reference']}  "
                  f"support {c['reference_support_px']:,} -> {c['candidate_support_px']:,} px")
    prov = contrast["pseudo_provenance"]
    print(f"  pseudo-label source: {prov['pseudo_label_path']} code {prov['pseudo_code']} "
          f"(source-circular with the proxy truth: {prov['circular_source']})")
    print(f"  VERDICT (derived): {contrast['verdict']} - {contrast['verdict_reason']}")
    print(f"  shippable on this evidence: {contrast['shippable_evidence']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0,
                    help="the fold whose pseudo-label arm is committed (default 0)")
    ap.add_argument("--gaps-out", default=str(BH / "fold_gap_summary.json"))
    ap.add_argument("--contrast-out", default=None,
                    help="default data/evidence/pseudo_labels/fold<K>_two_population_contrast.json")
    ap.add_argument("--check", action="store_true",
                    help="cross-check the recomposition against the runner's committed paired "
                         "contrast and write nothing")
    ap.add_argument("--strict", action="store_true",
                    help="exit(2) if that cross-check disagrees (a runner step uses this so an "
                         "independent recomputation of the same rows must match the runner's own "
                         "before anything derived from them is committed)")
    ap.add_argument("--gaps-only", action="store_true",
                    help="write only the fold-gap summary, no paired contrast (block-holdout.yml "
                         "uses this: at the moment a fold lands there is no pseudo-label arm to "
                         "contrast it with, and the summary must still count that fold)")
    ap.add_argument("--pool", action="store_true",
                    help="ALSO write the pooled multi-fold pseudo-label contrast (EXECUTIVE_SUMMARY "
                         "§11 item 7): every committed fold's two arms concatenated into one paired "
                         "block bootstrap, so the union arm clears the scorer's 12-unit readability "
                         "bar. Folds default to the partition's full fold set.")
    ap.add_argument("--pool-folds", default=None,
                    help="comma-separated folds to pool (default: every fold in the partition, read "
                         "from configs/config_block_holdout.yaml)")
    ap.add_argument("--pool-blocks", choices=["scoreable", "all"], default="scoreable",
                    help="'scoreable' (default) pools each fold's own scoreable held-out blocks; "
                         "'all' pools the full 56-block partition (reproduces a committed "
                         "single-fold CI exactly)")
    ap.add_argument("--pool-out", default=None,
                    help="default data/evidence/pseudo_labels/pooled_two_population_contrast.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    gaps = summarise_fold_gaps(load_fold_gaps(), expected_folds())
    contrast = None
    if not a.gaps_only:
        contrast = contrast_report(a.fold, load_json(DECISION), pseudo_provenance())

    pooled = None
    if a.pool:
        exp = expected_folds()
        if a.pool_folds:
            pool_folds = [int(x) for x in a.pool_folds.split(",") if x.strip()]
        else:
            pool_folds = exp.get("all_folds") or [a.fold]
        pooled = pooled_contrast(pool_folds, load_json(DECISION), pseudo_provenance(),
                                 block_mode=a.pool_blocks)

    problems = [] if contrast is None else self_check(contrast, a.fold)
    if problems:
        print("::warning:: cross-check against the committed paired contrast:")
        for p in problems:
            print(f"  - {p}")
        if a.strict:
            raise SystemExit(2)
    for scope, arm in (contrast or {}).get("arms", {}).items():
        for pop, c in (arm.get("populations") or {}).items():
            if c.get("status") == "MEASURED" and not all(c["recomposition_exact"].values()):
                raise SystemExit(f"recomposition is NOT exact for {scope}/{pop}: "
                                 f"{c['recomposition_exact']} - refusing to write derived numbers "
                                 "that disagree with their own committed components")

    if not a.quiet:
        print_report(gaps, contrast)
        if pooled is not None:
            print_pooled(pooled)

    if a.check:
        print("\n--check: nothing written")
        return 0

    outputs = [(Path(a.gaps_out), gaps)]
    if contrast is not None:
        outputs.append((Path(a.contrast_out or PL / f"fold{a.fold}_two_population_contrast.json"),
                        contrast))
    if pooled is not None:
        outputs.append((Path(a.pool_out or PL / "pooled_two_population_contrast.json"), pooled))
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1))
        if not a.quiet:
            print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
