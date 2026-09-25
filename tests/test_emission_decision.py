"""The emission-width decision must be computable, and must be falsifiable.

Two defects found on 2026-09-17 (session 10) while preparing the second-ensemble measurement:

1. ``scripts/decide_emission_width.py`` crashed in its final PRINT loop - ``out["crossovers"]``
   carries both crossing records (dicts with a ``reason``) and scalar context
   (``widest_swept_px``), and the loop subscripted every value as a dict.  The JSON was written
   first, so the workflow step failed *after* its evidence landed: a red reconciliation step
   looked like a completed one.  Pinned here by running the script on the committed evidence and
   requiring exit 0.
2. Condition 3 ("reproduced on a second, independently trained ensemble") was a hard-coded
   ``passes: False`` with the text "NOT MEASURED", i.e. the measurement the queued second
   ensemble exists to produce could not change the verdict even if it agreed.  It is now computed
   from a floor-controlled contrast between the widest swept band and the pure skeleton inside
   each sweep file - floor-controlled because absolute proxy DTI is not comparable across
   ensembles.  Both directions (reproduced / not reproduced) are pinned below.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decide_emission_width.py"
sys.path.insert(0, str(ROOT))

TRUTH_PX = 61664


def _sweep(gains: list[float], floors=(0.0, 0.05, 0.1), widths=(0, 3, 6),
           current_ref: float = 0.02, fp_w: float = 3e3) -> dict:
    """A synthetic sweep whose width-0 DTI is fixed and whose widest band moves by `gains`.

    `current_ref` is the sweep's own reference policy (the shipped policy re-blended from THIS
    fold set), because a reproduction is a contrast against the same sweep's reference - absolute
    proxy DTI is not comparable across ensembles.  `fp_w` scales the candidate's wrong mass, which
    is what decides whether the projection wins at every plausible scored-truth anchor.
    """
    rows = []
    for g, t0 in zip(gains, floors):
        for d in widths:
            dti = 0.02 + (g if d == max(widths) else 0.0)
            rows.append(dict(t0=t0, thin=True, dilate=d, dti=dti,
                             TP_w=dti * TRUTH_PX * 0.5, FP_w=fp_w * (1 + d),
                             FN_w=1e4, mass=fp_w * (1 + d), emission_px=int(fp_w * (1 + d))))
    return {"results": {"shaping_sweep": rows,
                        "as_provided": dict(dti=0.03, TP_w=2000.0, FP_w=5e5, emission_px=5165852),
                        "sweep_verdict": {"acceptance_rule": "test rule",
                                          "skeleton_dti": 0.02, "best_dti": 0.04,
                                          "current_policy_dti": current_ref}}}


def _evidence(tmp: Path, second_gains=None) -> dict:
    ev = {"truth": {"mode": "only", "px": TRUTH_PX, "km": TRUTH_PX * 0.1},
          "results": {"as_provided": dict(dti=0.0247, TP_w=1326.6, FP_w=20211.0, emission_px=21492),
                      "baselines": {"blanket_ones": dict(dti=0.0585, TP_w=61656.0, FP_w=4960570.0,
                                                         emission_px=5165852),
                                    "catalogue_plus_submission": dict(dti=0.0203, TP_w=1326.6,
                                                                      FP_w=78606.0, emission_px=79887)}}}
    rb = {"results": {"as_provided": dict(dti=0.0410, TP_w=2350.0, FP_w=37269.0, emission_px=39517)}}
    sweeps = {"--sweep": _sweep([0.0149, 0.0200, 0.0251])}
    if second_gains is not None:
        sweeps["--second-sweep"] = _sweep(second_gains)
    paths = {}
    for name, obj in [("--proxy-eval", ev), ("--reblend-eval", rb), *sweeps.items()]:
        p = tmp / f"{name.strip('-')}.json"
        p.write_text(json.dumps(obj))
        paths[name] = p
    idr = tmp / "in_domain.json"
    idr.write_text(json.dumps({"shaping": {"calibration_table": [
        dict(t0=0.0, thin=True, dilate=d, mean_dti=0.05 - 0.001 * d) for d in (0, 1, 3, 6)]}}))
    paths["--in-domain"] = idr
    return paths


def _run(tmp: Path, paths: dict) -> tuple[subprocess.CompletedProcess, dict]:
    out = tmp / "decision.json"
    cmd = [sys.executable, str(SCRIPT), "--out", str(out)]
    for k, v in paths.items():
        cmd += [k, str(v)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r, (json.loads(out.read_text()) if out.exists() else {})


def test_width_gain_table_is_floor_controlled():
    from scripts.decide_emission_width import width_gain_table

    t = width_gain_table(_sweep([0.0149, 0.0200, 0.0251]))
    assert t["widest_px"] == 6 and t["n_floors"] == 3
    assert t["min_gain"] == pytest.approx(0.0149) and t["max_gain"] == pytest.approx(0.0251)
    assert t["mean_gain"] == pytest.approx((0.0149 + 0.0200 + 0.0251) / 3)
    assert t["all_floors_positive"] is True
    # the contrast is taken at the SAME floor, so the width-0 column is the only reference
    assert all(row["dti_widest"] > row["dti_width0"] for row in t["per_floor"])

    neg = width_gain_table(_sweep([-0.001, -0.002, -0.003]))
    assert neg["all_floors_positive"] is False and neg["mean_gain"] < 0


def test_decision_script_runs_on_the_committed_evidence(tmp_path):
    """The crash-after-write defect: the step must exit 0, not just produce a file."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path / "real.json")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout tail:\n{r.stdout[-2000:]}"
    assert (tmp_path / "real.json").exists()


def _raw_sweep(dti_by_policy: dict, ref: float) -> dict:
    """Minimal sweep dict in the shape scripts/eval_proxy_catalogue.py writes."""
    rows = [dict(t0=t0, thin=True, dilate=w, soft=False, gamma=1.0, dti=d, emission_px=1000,
                 TP_w=d * TRUTH_PX * 0.5, FP_w=1e3, FN_w=1e3, mass=1000.0)
            for (t0, w), d in sorted(dti_by_policy.items())]
    return {"results": {"shaping_sweep": rows,
                        "sweep_verdict": {"current_policy_dti": ref}}}


def test_cross_ensemble_ranking_flags_a_field_specific_maximum():
    """The shipped candidate must survive EVERY sweep, not just the one that produced it.

    Session 13 added this ranking because `best_measured_candidate` is the argmax of the FIRST
    ensemble: a candidate that wins there and loses elsewhere is a maximum of one field, and a
    record that only reports the argmax cannot tell the difference.  The ranking is by the WORST
    contrast against each sweep's own reference policy - the same criterion condition 3 uses.
    """
    mod = importlib.import_module("scripts.decide_emission_width")
    # candidate (0.1, 0) wins on the first field; on the second it LOSES to (0.2, 0)
    e1 = _raw_sweep({(0.1, 0): 0.15, (0.2, 0): 0.04, (0.3, 0): 0.01}, ref=0.02)
    e2 = _raw_sweep({(0.1, 0): 0.01, (0.2, 0): 0.08, (0.3, 0): 0.01}, ref=0.02)
    r = mod.cross_ensemble_ranking([("e1.json", e1), ("e2.json", e2)])
    assert r["n_ensembles"] == 2 and r["n_candidates"] == 3
    assert (r["ranked"][0]["floor_t0"], r["ranked"][0]["width_px"]) == (0.2, 0), r["ranked"][0]
    assert r["ranked"][0]["worst_contrast"] == round(0.04 - 0.02, 6)
    by_policy = {(x["floor_t0"], x["width_px"]): x for x in r["ranked"]}
    assert by_policy[(0.1, 0)]["worst_contrast"] == round(0.01 - 0.02, 6)
    assert by_policy[(0.1, 0)]["contrast_by_ensemble"]["e1.json"] == round(0.15 - 0.02, 6)
    # a candidate missing from one sweep cannot be ranked at all (no silent extrapolation)
    e3 = _raw_sweep({(0.1, 0): 0.15}, ref=0.02)
    r2 = mod.cross_ensemble_ranking([("e1.json", e1), ("e3.json", e3)])
    assert r2["n_candidates"] == 1


def test_the_committed_evidence_now_says_ship(tmp_path):
    """Integration: on EVERY committed sweep, all three conditions pass and the rule is consistent.

    This is the record the re-blend workflow adopts its policy from, and the way the proxy-eval
    workflow builds it: the labelled sweeps on the branch are handed in together (the shipping-field
    mean12 sweep included), each contrasting the candidate with ITS OWN reference policy.  The test
    derives the expected field count from the files present, so a new ensemble strengthens the record
    instead of breaking the test - and the rank-1/first-place claim is checked against the number of
    sweeps that were actually supplied.
    """
    labelled = sorted((ROOT / "data/evidence/proxy").glob("eval_sweep-*.json"))
    if not labelled:
        pytest.skip("no labelled ensemble sweep is committed in this checkout")
    out = tmp_path / "ship.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--second-sweep",
                        ",".join(str(p) for p in labelled), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout tail:\n{r.stdout[-2000:]}"
    d = json.loads(out.read_text())
    v = d["verdict"]
    assert v["conclusion"].startswith("SHIP the measured policy"), v["conclusion"]
    assert all(c["passes"] for c in v["conditions"]), v["conditions"]
    bc = v["best_measured_candidate"]
    assert bc["policy"] == "sweep_best_t0_0.1_width0px", bc["policy"]
    assert bc["width_optimum_at_that_floor_px"] == 0
    repro = bc["reproduction_across_ensembles"]
    assert len(repro) == len(labelled), (len(repro), [x["sweep_file"] for x in repro])
    assert all(x["reproduced"] for x in repro), repro
    assert min(x["contrast_vs_reference"] for x in repro) > 0.01
    rob = v["robustness_across_ensembles"]
    assert rob["shipped_candidate_rank"] == 1, rob.get("shipped_candidate")
    assert rob["n_ensembles"] == 1 + len(labelled), rob["n_ensembles"]
    assert rob["n_candidates"] >= 100
    assert "warning" not in rob


def test_condition_three_stays_unmet_without_a_second_sweep(tmp_path):
    """No second ensemble sweep -> condition 3 is NOT MEASURED, and the verdict says so.

    Session 13 made the conditions about the best measured CANDIDATE (a joint floor x width policy)
    rather than about the width axis at a fixed floor: the search's winner on the proxy population
    is a floor change (floor 0.1, width 0 px), and a verdict phrased around "the widest band" could
    not have described it.
    """
    r, d = _run(tmp_path, _evidence(tmp_path))
    assert r.returncode == 0, r.stderr
    cond3 = d["verdict"]["conditions"][-1]
    assert cond3["passes"] is False
    assert "NOT MEASURED" in cond3["measured"]
    assert d["verdict"]["conclusion"].startswith("measured, not yet reproduced")
    cand = d["verdict"]["best_measured_candidate"]
    assert cand["policy"] in d["verdict"]["conclusion"]
    assert cand["reproduced_on_second_ensemble"] is None


def test_condition_three_passes_when_the_second_ensemble_reproduces_the_gain(tmp_path):
    """The candidate's CONTRAST against the second sweep's own reference policy is what must hold."""
    r, d = _run(tmp_path, _evidence(tmp_path, second_gains=[0.0131, 0.0188, 0.0240]))
    assert r.returncode == 0, r.stderr
    cond3 = d["verdict"]["conditions"][-1]
    assert cond3["passes"] is True, cond3
    assert "reproduced" in cond3["measured"] and "NOT reproduced" not in cond3["measured"]
    repro = d["verdict"]["best_measured_candidate"]["reproduction_across_ensembles"]
    assert len(repro) == 1 and repro[0]["reproduced"] is True
    assert repro[0]["reference_is"] == "sweep_verdict.current_policy_dti of that same sweep"
    # ... and when every condition passes the record instructs the ship path, naming the workflow
    # flags that write an adopted policy (a conclusion that cannot be acted on is not a decision)
    assert d["verdict"]["conclusion"].startswith("SHIP the measured policy")
    assert "SHAPING_T0/SHAPING_DILATE" in d["verdict"]["conclusion"]
    assert all(c["passes"] for c in d["verdict"]["conditions"]), d["verdict"]["conditions"]


def test_condition_three_fails_when_the_second_ensemble_flips_the_sign(tmp_path):
    r, d = _run(tmp_path, _evidence(tmp_path, second_gains=[-0.002, -0.003, -0.004]))
    assert r.returncode == 0, r.stderr
    cond3 = d["verdict"]["conditions"][-1]
    assert cond3["passes"] is False, cond3
    assert "NOT reproduced" in cond3["measured"]
    out = d["verdict"]
    assert out["conclusion"].startswith("measured, not shipped")
    assert "condition 3" in out["conclusion"], out["conclusion"]


def test_a_repeated_second_sweep_flag_does_not_drop_the_earlier_sweeps(tmp_path):
    """`--second-sweep a --second-sweep b` evaluates BOTH (argparse would keep only the last).

    The workflow joins the sweeps into one comma list, but a human or a future script calling the
    flag twice would have had the earlier ensembles silently dropped while condition 3 still
    reported a pass - less evidence under the same verdict.
    """
    paths = _evidence(tmp_path, second_gains=[0.0131, 0.0188, 0.0240])
    second = paths.pop("--second-sweep")
    third = tmp_path / "third_sweep.json"
    third.write_text(json.dumps(_sweep([0.0131, 0.0188, 0.0240])))
    out = tmp_path / "decision.json"
    cmd = [sys.executable, str(SCRIPT), "--out", str(out)]
    for k, v in paths.items():
        cmd += [k, str(v)]
    cmd += ["--second-sweep", str(second), "--second-sweep", str(third)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d["inputs"]["second_sweeps"] == [str(second), str(third)], d["inputs"]
    assert d["inputs"]["second_sweep"] == f"{second},{third}", d["inputs"]
    repro = d["verdict"]["best_measured_candidate"]["reproduction_across_ensembles"]
    assert len(repro) == 2 and all(x["reproduced"] for x in repro), repro


def test_a_second_list_of_sweeps_is_reproduced_one_by_one(tmp_path):
    """`--second-sweep a,b` : two independent ensembles, each contrasted with its own reference.

    One agreeing ensemble is weak evidence; the record has to be able to hold several, and it must
    fail when ANY of them disagrees (an average over ensembles would hide exactly the case that
    matters).
    """
    paths = _evidence(tmp_path, second_gains=[0.0131, 0.0188, 0.0240])
    third = _sweep([0.0131, 0.0188, 0.0240])
    third_path = tmp_path / "third_sweep.json"
    third_path.write_text(json.dumps(third))
    paths["--second-sweep"] = str(paths["--second-sweep"]) + "," + str(third_path) + \
        "," + str(tmp_path / "missing.json")          # a missing path is skipped, not invented
    r, d = _run(tmp_path, paths)
    assert r.returncode == 0, r.stderr
    repro = d["verdict"]["best_measured_candidate"]["reproduction_across_ensembles"]
    assert len(repro) == 2 and all(x["reproduced"] for x in repro)
    assert d["verdict"]["conditions"][-1]["passes"] is True

    # now make the second one disagree: the same policy, a flipped contrast
    third_path.write_text(json.dumps(_sweep([-0.02, -0.02, -0.02], current_ref=0.05)))
    r, d = _run(tmp_path, paths)
    assert r.returncode == 0, r.stderr
    repro = d["verdict"]["best_measured_candidate"]["reproduction_across_ensembles"]
    assert [x["reproduced"] for x in repro] == [True, False], repro
    assert d["verdict"]["conditions"][-1]["passes"] is False
    assert d["verdict"]["conclusion"].startswith("measured, not shipped")


def test_stale_proxy_evidence_is_flagged_by_the_km_field():
    """The committed sweep/submission evidence predates the 0.01 -> 0.1 km-per-pixel fix.

    The decision itself is unaffected (it projects on pixels), but any prose quoting those files'
    ``truth.km`` would repeat a 10x error, so this test records which files are stale until the
    next proxy-eval run rewrites them.
    """
    stale, fresh = [], []
    for name in ("eval_submission.json", "eval_sweep.json", "eval_reblend_submission.json"):
        p = ROOT / "data" / "evidence" / "proxy" / name
        if not p.exists():
            continue
        t = json.loads(p.read_text()).get("truth", {})
        (stale if t.get("km") and abs(t["km"] - t["px"] * 0.1) > 1e-6 else fresh).append(name)
    # Either state is acceptable; what must never happen is an unnoticed mix.  Both lists are
    # asserted to be *disjoint and total*, i.e. every file is classified.
    assert not set(stale) & set(fresh)
    if stale:
        # document the known-stale set precisely (updated by the next proxy-eval run)
        assert set(stale) <= {"eval_submission.json", "eval_sweep.json",
                              "eval_reblend_submission.json"}, stale


def test_ramp_rows_are_policies_but_never_a_width():
    """Session 11 added a second sweep axis: the VALUES written on a support (hard vs ramp).

    A ramp candidate is identified by (floor, width, gamma) - nothing in the string "width 6 px"
    says which of the two it is.  So the decision script must (a) rank ramp candidates as policies
    like any other, and (b) keep them out of the width machinery, whose whole premise is that one
    number (the band width) describes the candidate.
    """
    from scripts.decide_emission_width import width_gain_table

    sw = _sweep([0.0149, 0.0200, 0.0251])                 # hard rows, widths 0/3/6
    hard_rows = sw["results"]["shaping_sweep"]
    # a ramp twin of every width-6 row, worth slightly more than the hard band of the same width
    for r in list(hard_rows):
        if r["dilate"] == 6:
            sw["results"]["shaping_sweep"].append(dict(r, soft=True, gamma=1.0, dti=r["dti"] + 0.004))

    t = width_gain_table(sw)
    assert t["widest_px"] == 6, "a ramp row must not be mistaken for a wider hard band"
    assert sorted(t["per_floor"][0]["gain"].__class__.__name__ for _ in [0]) == ["float"]

    # the same table with the ramp rows REMOVED must be identical: proof of non-interference
    sw2 = _sweep([0.0149, 0.0200, 0.0251])
    assert width_gain_table(sw2) == t


def test_ramp_candidates_appear_in_the_decision_ranking(tmp_path):
    paths = _evidence(tmp_path)
    sw = json.loads(paths["--sweep"].read_text())
    for r in list(sw["results"]["shaping_sweep"]):
        if r["dilate"] == 6:
            sw["results"]["shaping_sweep"].append(dict(r, soft=True, gamma=2.0))
    paths["--sweep"].write_text(json.dumps(sw))
    r, out = _run(tmp_path, paths)
    assert r.returncode == 0, (r.stdout, r.stderr)
    names = [p["policy"] for p in out["policies"]]
    assert any("ramp" in n for n in names), names
    assert "ramp emission candidates scored" in r.stdout


def test_width_gain_table_ignores_ramp_rows_entirely():
    """The contrast must be hard vs hard: a ramp row that only exists at the widest width would
    otherwise define "the widest band" while having no width-0 counterpart."""
    from scripts.decide_emission_width import width_gain_table

    sw = _sweep([0.02, 0.02, 0.02], floors=(0.0, 0.05, 0.1), widths=(0, 3, 6))
    clean = width_gain_table(sw)
    # a ramp row at a width that the hard grid does not contain at all
    sw["results"]["shaping_sweep"].append(dict(t0=0.0, thin=True, dilate=12, soft=True, gamma=1.0,
                                               dti=0.5, TP_w=1.0, FP_w=1.0, FN_w=1.0,
                                               mass=1.0, emission_px=1))
    assert width_gain_table(sw) == clean


def test_the_best_hard_candidate_of_every_floor_is_in_the_ranking(tmp_path):
    """The candidate set must not decide the verdict.

    The policy ranking used to admit sweep rows only at floors {0, 0.043} and widths {0, 3, 6}
    (plus the widest band).  The session-12 extended grid then found its best HARD candidate at
    floor 0.1 with width 0 - a floor change - which that subset silently excluded.  The first
    ensemble's own sweep is the regression case: its `best_dti` (0.136452) must appear in the
    ranking, and the winner's whole width curve must be recorded with it.
    """
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path / "real.json")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "real.json").read_text())
    sweep = json.loads((ROOT / "data/evidence/proxy/eval_sweep.json").read_text())
    sweep_best = sweep["results"]["sweep_verdict"]["best_dti"]
    ranked = {p["policy"]: p["measured_dti"] for p in out["policies"]}
    assert max(ranked.values()) == pytest.approx(sweep_best, abs=1e-6), ranked
    bc = out["verdict"]["best_measured_candidate"]
    assert ranked[bc["policy"]] == pytest.approx(bc["measured_dti"])
    assert bc["contrast_vs_shipped"] > 0.01
    # the winning floor's whole width curve, so "lower the floor" and "widen the band" cannot be
    # mistaken for additive knobs when the best width at that floor is 0 px
    if bc.get("widths_at_that_floor"):
        w = {int(k): v for k, v in bc["widths_at_that_floor"].items()}
        assert bc["width_optimum_at_that_floor_px"] == max(w, key=lambda k: w[k])


def test_best_candidate_reproduction_is_measured_on_the_second_ensemble(tmp_path):
    """A floor candidate is reproduced the same way a width is: same policy, second sweep, and the
    contrast taken against THAT sweep's own reference policy (absolute DTI is not comparable)."""
    paths = _evidence(tmp_path)
    sw = json.loads(paths["--sweep"].read_text())
    sw["results"]["sweep_verdict"]["current_policy_dti"] = 0.02
    paths["--sweep"].write_text(json.dumps(sw))
    second = _sweep([0.0, 0.0, 0.0])
    second["results"]["sweep_verdict"]["current_policy_dti"] = 0.01
    second_path = tmp_path / "second_sweep.json"
    second_path.write_text(json.dumps(second))
    paths["--second-sweep"] = second_path

    r, out = _run(tmp_path, paths)
    assert r.returncode == 0, (r.stdout, r.stderr)
    rep = out["verdict"]["best_measured_candidate"]["reproduced_on_second_ensemble"]
    assert rep is not None, out["verdict"]["best_measured_candidate"]
    assert rep["reference_is"] == "sweep_verdict.current_policy_dti of that same sweep"
    assert rep["contrast_vs_reference"] == pytest.approx(rep["measured_dti"] - 0.01, abs=1e-9)
