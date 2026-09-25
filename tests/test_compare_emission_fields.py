"""Field-axis comparison (scripts/compare_emission_fields.py).

A floor is a threshold on a field whose scale changes with the number of averaged folds, so two
fields at "the same policy" do not emit the same pixels: measured from the committed sweeps,
floor 0.1 / thin / width 0 emits 464,736 px on the 6-fold field and 144,738 px on the 16-fold one.
Comparing fields at a fixed floor therefore measures the floor.  This script compares them at
MATCHED SUPPORT instead, and the tests below pin:

* the matched-support rule is applied identically to every field, over three nested windows, and
  the verdict requires the ranking to hold in all of them before it prefers another field;
* a field that looks better at the adopted floor purely because it emits more pixels is NOT
  declared better once support is matched (the real ensemble-1 case: 0.1365 at 464,736 px becomes
  0.0644 at the shipped support);
* the shipped support is cross-checked against two independent committed sources;
* the knife edge is visible: the nearest-support candidate and the best-in-window candidate are
  both reported, because for some fields they are different candidates;
* missing evidence fails loudly with the workflow that produces it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mod():
    spec = importlib.util.spec_from_file_location(
        "cmp_fields", ROOT / "scripts" / "compare_emission_fields.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(t0, dilate, dti, px, tp=None, fp=None, fn=None, thin=True, soft=False):
    return {"t0": t0, "thin": thin, "dilate": dilate, "soft": soft, "gamma": 1.0,
            "dti": dti, "TP_w": tp if tp is not None else dti * 1000.0,
            "FP_w": fp if fp is not None else px * 0.2,
            "FN_w": fn if fn is not None else 1000.0 - dti * 1000.0,
            "mass": float(px), "emission_px": px, "mean_kept_px": px}


def _write_sweep(d: Path, name: str, rows: list[dict], run_ids: str, label: str) -> Path:
    stem = name[len("eval_sweep"):-len(".json")].lstrip("-")
    (d / name).write_text(json.dumps({
        "inputs": {"pred": "ensemble_mean.tif", "pred_sha256": f"sha-{label}"},
        "truth": {"mode": "only", "px": 61664},
        "results": {"shaping_sweep": rows}}))
    src = f"sweep_source{('-' + stem) if stem else ''}.json"
    (d / src).write_text(json.dumps({"swept_run_id": run_ids, "sweep_file": name}))
    return d / name


SUPPORT = 172_974


@pytest.fixture()
def proxy_dir(tmp_path) -> Path:
    """Two fields: the shipped 11-fold mean, and a 6-fold field that emits 2.7x more at floor 0.1."""
    d = tmp_path / "proxy"
    d.mkdir()
    shipped = [
        _row(0.05, 0, 0.1260, 770_741),
        _row(0.1, 0, 0.0999, SUPPORT),          # <- the adopted policy, and the shipped support
        _row(0.1, 1, 0.0878, 205_000),
        _row(0.2, 0, 0.0860, 150_000),
        _row(0.362091, 0, 0.0304, 39_517),      # the in-domain reference policy
    ]
    six_fold = [
        _row(0.05, 0, 0.0144, 14_285),
        _row(0.1, 0, 0.1365, 464_736),          # higher DTI, but 2.7x the shipped support
        _row(0.2, 0, 0.0870, 260_000),
        _row(0.3, 1, 0.0644, 173_286),          # <- matched support: much worse
        _row(0.362091, 0, 0.0410, 39_517),
    ]
    _write_sweep(d, "eval_sweep-mean12.json", shipped, "1,2", "mean12")
    _write_sweep(d, "eval_sweep.json", six_fold, "1", "ensemble1")
    return d


def _rep(m, proxy_dir, tmp_path, **kw):
    out = tmp_path / "field_axis.json"
    argv = ["--proxy-dir", str(proxy_dir), "--block-report", kw.pop("block_report", ""),
            "--out", str(out), "--quiet"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    assert m.main(argv) == 0
    return json.loads(out.read_text())


# --------------------------------------------------------------------------------------
def test_matched_support_removes_the_support_effect(proxy_dir, tmp_path):
    m = _mod()
    rep = _rep(m, proxy_dir, tmp_path)
    by = {f["field"]: f for f in rep["fields"]}
    assert rep["shipped_support"]["support_px"] == SUPPORT
    # at the adopted floor the 6-fold field looks better...
    assert by["ensemble1"]["adopted_policy"]["dti"] == pytest.approx(0.1365)
    assert by["ensemble1"]["adopted_policy"]["emission_px"] == 464_736
    assert by["mean12"]["adopted_policy"]["dti"] == pytest.approx(0.0999)
    # ...and at matched support it is much worse
    tol = str(rep["primary_window"])
    assert by["ensemble1"]["matched_support"]["best_in_window"][tol]["dti"] == pytest.approx(0.0644)
    assert by["ensemble1"]["matched_support"]["best_in_window"][tol]["t0"] == 0.3
    assert by["mean12"]["matched_support"]["best_in_window"][tol]["dti"] == pytest.approx(0.0999)
    v = rep["verdict"]
    assert v["top_field_by_window"][tol] == "mean12"
    assert v["ranking_stable_across_windows"] is True
    assert v["conclusion"].startswith("The shipped field (mean12)")
    assert "SUPPORT effect" in v["conclusion"]
    assert "surrogate_prefers_a_different_field" not in v


def test_every_window_is_reported_and_the_rule_is_the_same_for_every_field(proxy_dir, tmp_path):
    m = _mod()
    rep = _rep(m, proxy_dir, tmp_path)
    assert rep["support_windows"] == [0.15, 0.25, 0.4]
    for f in rep["fields"]:
        win = f["matched_support"]["best_in_window"]
        assert set(win) == {"0.15", "0.25", "0.4"}
        for tol, row in win.items():
            if row is None:
                continue
            assert abs(row["emission_px"] - SUPPORT) <= float(tol) * SUPPORT + 1
            assert abs(row["support_deviation"]) <= float(tol) + 1e-6
        assert f["matched_support"]["n_candidates_in_window"]["0.15"] <= \
               f["matched_support"]["n_candidates_in_window"]["0.4"]


def test_a_field_that_is_genuinely_better_at_matched_support_is_reported(tmp_path):
    """Same rule, opposite outcome: the verdict must be derived, not hard-coded to 'keep shipping'."""
    m = _mod()
    d = tmp_path / "proxy2"
    d.mkdir()
    shipped = [_row(0.1, 0, 0.0500, SUPPORT), _row(0.362091, 0, 0.0200, 39_517)]
    better = [_row(0.2, 0, 0.1400, 170_000), _row(0.362091, 0, 0.0300, 40_000)]
    _write_sweep(d, "eval_sweep-mean12.json", shipped, "1,2", "mean12")
    _write_sweep(d, "eval_sweep.json", better, "1", "ensemble1")
    rep = _rep(m, d, tmp_path)
    v = rep["verdict"]
    pref = v["surrogate_prefers_a_different_field"]
    assert pref["field"] == "ensemble1" and pref["dti"] == pytest.approx(0.1400)
    assert pref["gain"] == pytest.approx(0.09, abs=1e-6)
    assert v["conclusion"].startswith("At matched support the surrogate prefers")
    assert "does NOT rank the FIELD axis" in v["conclusion"], \
        "the verdict must say which rule is missing before recommending a change"


def test_an_unstable_ranking_refuses_to_prefer_a_field(tmp_path):
    """If the top field depends on the window, the comparison must not recommend anything."""
    m = _mod()
    d = tmp_path / "proxy3"
    d.mkdir()
    # field B beats the shipped field only in the widest window (its good candidate is far off support)
    shipped = [_row(0.1, 0, 0.0900, SUPPORT), _row(0.362091, 0, 0.0200, 39_517)]
    other = [_row(0.1, 0, 0.0300, 168_000),          # inside every window, worse
             _row(0.05, 0, 0.2000, 230_000),         # only inside the +-40 % window
             _row(0.362091, 0, 0.0100, 40_000)]
    _write_sweep(d, "eval_sweep-mean12.json", shipped, "1,2", "mean12")
    _write_sweep(d, "eval_sweep.json", other, "1", "ensemble1")
    rep = _rep(m, d, tmp_path)
    v = rep["verdict"]
    assert v["ranking_stable_across_windows"] is False
    assert v["top_field_by_window"]["0.4"] == "ensemble1"
    assert v["top_field_by_window"][str(rep["primary_window"])] == "mean12"
    assert "surrogate_prefers_a_different_field" not in v


def test_the_knife_edge_is_visible_in_the_report(tmp_path):
    """closest-support and best-in-window can be different candidates; both must be recorded."""
    m = _mod()
    d = tmp_path / "proxy4"
    d.mkdir()
    shipped = [_row(0.1, 0, 0.0999, SUPPORT)]
    other = [_row(0.8, 12, 0.0440, 168_868),        # nearest support, poor DTI
             _row(0.1, 0, 0.0850, 144_738)]         # inside +-25 %, outside +-15 %
    _write_sweep(d, "eval_sweep-mean12.json", shipped, "1,2", "mean12")
    _write_sweep(d, "eval_sweep-ens123.json", other, "1,2,3", "ens123")
    rep = _rep(m, d, tmp_path)
    f = next(x for x in rep["fields"] if x["field"] == "ens123")
    ms = f["matched_support"]
    assert ms["closest"]["t0"] == 0.8 and ms["closest"]["dti"] == pytest.approx(0.0440)
    assert ms["best_in_window"]["0.15"] is None or ms["best_in_window"]["0.15"]["t0"] == 0.8
    assert ms["best_in_window"]["0.25"]["t0"] == 0.1
    assert ms["best_in_window"]["0.25"]["dti"] == pytest.approx(0.0850)
    assert "knife_edge_note" in rep["verdict"]


def test_shipped_support_is_cross_checked_against_the_block_report(proxy_dir, tmp_path):
    m = _mod()
    block = tmp_path / "block_stratified.json"
    block.write_text(json.dumps({
        "verdict": {"reference_candidate": "floor0.1_w0px"},
        "populations": {"proxy_only": {"candidates": [
            {"label": "floor0.1_w0px", "pred_px": SUPPORT}]}}}))
    rep = _rep(m, proxy_dir, tmp_path, block_report=str(block))
    src = rep["shipped_support"]["sources"]
    assert src["block_stratified_report"] == SUPPORT
    assert src["adopted_policy_row_of_eval_sweep-mean12.json"] == SUPPORT
    assert rep["shipped_support"]["sources_agree"] is True

    disagree = tmp_path / "disagree.json"
    disagree.write_text(json.dumps({
        "verdict": {"reference_candidate": "floor0.1_w0px"},
        "populations": {"proxy_only": {"candidates": [
            {"label": "floor0.1_w0px", "pred_px": 999_999}]}}}))
    rep2 = _rep(m, proxy_dir, tmp_path, block_report=str(disagree))
    assert rep2["shipped_support"]["sources_agree"] is False, \
        "a support that two committed sources disagree about must be flagged, not averaged away"


def test_unreadable_or_empty_evidence_fails_loudly(tmp_path):
    m = _mod()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        _rep(m, empty, tmp_path)
    assert "proxy-eval.yml" in str(exc.value)

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "eval_sweep.json").write_text("{not json")
    with pytest.raises(SystemExit):
        _rep(m, broken, tmp_path)


def test_no_candidate_near_the_shipped_support_says_so(tmp_path):
    """Only reachable when the shipped field's adopted-policy row is absent, so the support falls
    back to the documented default and no swept candidate lands near it."""
    m = _mod()
    d = tmp_path / "proxy5"
    d.mkdir()
    far = [_row(0.02, 0, 0.09, 4_900_000), _row(0.5, 0, 0.02, 20_000)]   # no t0=0.1 row at all
    _write_sweep(d, "eval_sweep-mean12.json", far, "1,2", "mean12")
    rep = _rep(m, d, tmp_path)
    assert rep["fields"][0]["adopted_policy"] is None
    assert rep["shipped_support"]["fallback_used"] is True
    assert rep["shipped_support"]["support_px"] == m.DEFAULT_SUPPORT_PX
    assert rep["fields"][0]["matched_support"]["best_in_window"]["0.4"] is None
    assert "no matched-support comparison is possible" in rep["verdict"]["conclusion"]
    assert "SHAPING_VALUES" in rep["verdict"]["conclusion"]


def test_the_support_source_is_named_when_it_falls_back(tmp_path):
    m = _mod()
    d = tmp_path / "proxy6"
    d.mkdir()
    _write_sweep(d, "eval_sweep-mean12.json", [_row(0.1, 0, 0.09, SUPPORT)], "1,2", "mean12")
    rep = _rep(m, d, tmp_path, shipped_support=200_000)
    assert rep["shipped_support"]["support_px"] == 200_000
    assert rep["shipped_support"]["sources"]["cli_override"] == 200_000
    assert rep["shipped_support"]["sources_agree"] is False, \
        "an override that disagrees with the committed row must be visible as a disagreement"


def test_committed_evidence_ranks_the_shipped_field_first():
    """The real committed sweeps: the 11-fold mean is the best field at matched support."""
    proxy = ROOT / "data/evidence/proxy"
    if not (proxy / "eval_sweep-mean12.json").exists():
        pytest.skip("committed sweeps not present")
    m = _mod()
    out = Path("/tmp/field_axis_from_committed.json")
    assert m.main(["--proxy-dir", str(proxy), "--out", str(out), "--quiet"]) == 0
    rep = json.loads(out.read_text())
    v = rep["verdict"]
    assert v["shipped_field"] == "mean12"
    assert rep["shipped_support"]["support_px"] == SUPPORT
    assert v["ranking_stable_across_windows"] is True
    assert all(f == "mean12" for f in v["top_field_by_window"].values()), v["top_field_by_window"]
    by = {f["field"]: f for f in rep["fields"]}
    tol = str(rep["primary_window"])
    assert by["mean12"]["matched_support"]["best_in_window"][tol]["dti"] == pytest.approx(0.099859)
    assert by["ensemble1"]["adopted_policy"]["emission_px"] > 2 * SUPPORT, \
        "the 6-fold field's advantage at floor 0.1 is a support effect"
    assert by["ensemble1"]["matched_support"]["best_in_window"][tol]["dti"] < 0.0999
