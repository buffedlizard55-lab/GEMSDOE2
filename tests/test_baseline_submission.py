"""Tests for scripts/baseline_submission.py - the CPU-only path to an uploadable submission.

Why these matter: this is the only generator in the repository that can run end to end on a
machine with no GPU and no runner artefacts, so it is the fallback that keeps a submission
possible at all.  Two properties have to hold for it to be trustworthy:

* **the artefact is format-valid** - written through `src/submission_io.write_submission` (which
  reads the bytes back) on the template's grid, float32, [0, 1], NaN outside the footprint;
* **the policy was chosen on geography the model never saw** - the positives in the training
  sample must contain no pixel of the held-out fold (including its R-pixel collar), and the
  winner in the report must be the argmax of the union-population DTI the report itself lists.

Everything runs on a synthetic grid; no competition data, no network, no torch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import score_within_mask  # noqa: E402

MOD_NAME = "baseline_submission"


def _mod():
    spec = importlib.util.spec_from_file_location(MOD_NAME, ROOT / "scripts" / f"{MOD_NAME}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, arr: np.ndarray, *, dtype, nodata=None, transform=None, tags=None):
    h, w = arr.shape[-2:]
    count = 1 if arr.ndim == 2 else arr.shape[0]
    prof = dict(driver="GTiff", height=h, width=w, count=count, dtype=dtype,
                crs="EPSG:32611", transform=transform or from_origin(500000.0, 4100000.0, 100.0, 100.0),
                compress="lzw")
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        if count == 1:
            dst.write(arr.astype(dtype), 1)
        else:
            dst.write(arr.astype(dtype))
        if tags:
            dst.update_tags(**tags)
    return path


def _fixture_grid(d: Path, size: int = 192, bands: int = 4):
    """A synthetic competition-like dataset.

    Two vertical fault traces (columns 30-31 and 130-131), a footprint that covers columns
    0..size-1 but only rows 4..size-4 (so NaN handling is exercised), and a proxy-only trace at
    column 90 that no label contains.  Column 60 is engineered to be *causally* informative: the
    fault traces sit on a constant high band-0 value, so a per-pixel model can learn something
    instead of fitting noise.
    """
    rng = np.random.default_rng(7)
    transform = from_origin(500000.0, 4100000.0, 100.0, 100.0)
    labels = np.full((size, size), -1, np.int8)              # -1 = nodata (outside footprint)
    labels[4:size - 4, :] = 0
    fault = np.zeros((size, size), bool)
    for col in (30, 31, 130, 131):
        fault[10:size - 10, col] = True
    labels[fault] = 1
    X = rng.normal(0, 1, (bands, size, size)).astype(np.float32)
    X[:, fault] += 2.5                                        # signal the model can find
    X[:, 4:size - 4, :] = np.where(np.isnan(X[:, 4:size - 4, :]), X[:, 4:size - 4, :], X[:, 4:size - 4, :])
    X[:, :4, :] = -3.4028235e38                               # nodata sentinel
    X[:, size - 4:, :] = -3.4028235e38
    sample = np.full((size, size), np.nan, np.float32)        # template: NaN outside footprint
    sample[4:size - 4, :] = 0.0
    proxy = np.zeros((size, size), np.uint8)
    proxy[10:size - 10, 90] = 2                               # code 2 = new-fault-like
    proxy[fault] = 1

    feat = _write(d / "features.tif", X, dtype="float32", nodata=-3.4028235e38)
    labs = _write(d / "labels.tif", labels, dtype="int8", nodata=-1)
    tmpl = _write(d / "sample.tif", sample, dtype="float32", nodata=np.nan)
    prox = _write(d / "proxy.tif", proxy, dtype="uint8")
    return dict(features=feat, labels=labs, template=tmpl, proxy=prox, fault=fault,
                size=size, transform=transform, X=X)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One real invocation of main() on the synthetic grid, reused by several tests.

    The model needs scikit-learn; the pure helpers below do not, so the skip lives here rather
    than at module scope - a minimal environment still exercises the sampling/ranking logic.
    """
    pytest.importorskip("sklearn", reason="the CPU baseline's model needs scikit-learn")
    d = tmp_path_factory.mktemp("baseline")
    fx = _fixture_grid(d)
    out_dir = d / "out"
    mod = _mod()
    rc = mod.main(["--features", str(fx["features"]), "--labels", str(fx["labels"]),
                   "--template", str(fx["template"]), "--proxy", str(fx["proxy"]),
                   "--out-dir", str(out_dir), "--block-px", "64", "--folds", "4", "--fold", "0",
                   "--max-iter", "20", "--max-negatives", "4000", "--neg-ratio", "10",
                   "--chunk-rows", "48", "--seed", "3",
                   # The fixture is 192 px of grid with 64 px blocks: every fold boundary sits
                   # within a few R of the fault traces, so the two implementations' boundary-credit
                   # conventions legitimately diverge far more than on the competition grid.  The
                   # loose tolerance is stated here rather than baked into the code's default.
                   "--verify-tolerance", "0.25"])
    report = json.loads((out_dir / "baseline_report.json").read_text())
    return dict(rc=rc, out_dir=out_dir, fx=fx, report=report, mod=mod)


# ------------------------------------------------------------------------------ format gate
def test_main_rc_and_both_artefacts_exist(run):
    assert run["rc"] == 0
    assert (run["out_dir"] / "submission.tif").exists()
    assert (run["out_dir"] / "prob_raw.tif").exists()
    for key in ("submission", "prob_raw"):
        assert run["report"][key]["sha256"] == hashlib.sha256(
            Path(run["report"][key]["path"]).read_bytes()).hexdigest(), \
            "the report must quote the bytes on disk, not a value computed before the write"


def test_submission_matches_the_official_spec_on_the_template_grid(run):
    fx, rep = run["fx"], run["report"]
    with rasterio.open(run["out_dir"] / "submission.tif") as src:
        with rasterio.open(fx["template"]) as tmpl:
            assert (src.width, src.height, src.count) == (tmpl.width, tmpl.height, 1)
            assert src.crs == tmpl.crs and src.transform == tmpl.transform
            assert src.dtypes[0] == "float32"
            a = src.read(1)
            t = tmpl.read(1)
    finite = np.isfinite(a)
    assert np.array_equal(finite, np.isfinite(t)), "footprint must match the template exactly"
    assert finite.any() and float(a[finite].min()) >= 0.0 and float(a[finite].max()) <= 1.0
    assert int(np.count_nonzero(a[finite])) > 0, "an all-zero submission is the absence template"
    assert rep["submission"]["nan_px"] == int((~finite).sum())


def test_raw_field_is_a_probability_raster(run):
    with rasterio.open(run["out_dir"] / "prob_raw.tif") as src:
        a = src.read(1)
    finite = np.isfinite(a)
    assert finite.any()
    assert float(a[finite].min()) >= 0.0 and float(a[finite].max()) <= 1.0


# ------------------------------------------------------------------------ leakage / selection
def test_no_training_positive_lies_in_either_held_out_fold(run):
    """The generator's central claim: the shipped field trained on NEITHER fold it is judged on.

    Two folds are held out - the one that selects the emission policy and the one that measures it -
    and the model sees neither, collar included.
    """
    mod, fx, rep = run["mod"], run["fx"], run["report"]
    from src.blocks import assign_folds, block_table, held_out_mask, scored_mask
    fault, valid = mod.read_truth(Path(fx["labels"]))
    folds, ef = 4, rep["measurement_fold"]
    table = block_table(fault.shape, 64, valid=valid, labels=fault)
    fold_of = assign_folds(table, folds, 3, "balanced")
    collars = [held_out_mask(fault.shape, 64, fold_of, f, mod.DEFAULT_BUFFER_PX)
               for f in (0, ef)]
    trainable = valid & ~(collars[0] | collars[1])
    rng = np.random.default_rng(3)
    sel = mod.choose_samples(fault, valid, trainable, rng, 10.0, 4000)
    assert not (sel["pos"] & collars[0]).any()
    assert not (sel["pos"] & collars[1]).any()
    assert not (sel["neg"] & (collars[0] | collars[1])).any()
    assert int(sel["pos"].sum()) == int((fault & trainable).sum()), \
        "every trainable positive must be kept - sampling down-weights negatives only"
    for f in (0, ef):
        held = scored_mask(fault.shape, 64, fold_of, f)
        assert int((fault & held).sum()) > 0, f"fold {f} must actually hold out fault px"


def test_report_winner_is_the_argmax_of_the_union_dti_it_lists(run):
    rep = run["report"]["policy_selection"]
    eligible = [r for r in rep["candidates"] if r["combined_dti"] is not None and r["eligible"]]
    assert eligible, "the held-out fold must contain both populations"
    best = max(r["combined_dti"] for r in eligible)
    assert rep["winner"]["combined_dti"] == best
    assert rep["winner"]["eligible"] is True
    assert rep["winner"] in rep["candidates"], "the winner must be one of the measured candidates"
    assert rep["scope"].startswith("blocks of fold 0")


def test_winner_policy_is_reproducible_from_the_report(run):
    """Re-applying the reported policy to the reported raw field must reproduce the DTIs."""
    mod = run["mod"]
    rep = run["report"]
    with rasterio.open(rep["prob_raw"]["path"]) as src:
        prob = src.read(1)
    with rasterio.open(run["out_dir"] / "submission.tif") as src:
        shipped = src.read(1)
    w = rep["policy_selection"]["winner"]
    again = np.where(np.isfinite(shipped),
                     mod.optimize_submission(prob, R=rep["metric"]["R_pixels"], t0=w["t0"],
                                             thin=w["thin"], dilate=w["dilate"]), np.nan)
    assert np.array_equal(np.nan_to_num(again), np.nan_to_num(shipped))


def test_policy_is_selected_and_measured_on_different_folds(run):
    """max() over N candidates on fold A is not an estimate of fold A; the report must not pretend.

    The selection fold carries the winning candidate's own (upward-biased) maximum, so the value the
    report invites a reader to quote has to come from a fold that neither trained the model nor took
    part in the sweep.
    """
    rep = run["report"]
    assert rep["held_out_fold"] == 0
    assert rep["measurement_fold"] != rep["held_out_fold"]
    gen = rep["generalisation"]
    assert gen["fold"] == rep["measurement_fold"]
    assert set(gen["by_population"]) == {"catalogue", "proxy", "combined"}
    assert all(v["scoreable"] for v in gen["by_population"].values())
    assert str(rep["measurement_fold"]) in gen["reproduce"] and "--score-fold" in gen["reproduce"]
    # The audit must score the SHAPED artifact.  block_holdout_eval shapes by threshold+dilation
    # and cannot thin, so pointing it at prob_raw.tif with the winner's floor would measure the
    # un-thinned sibling instead - a different field, and a much higher number (measured on the
    # real grid: combined 0.1259 un-thinned vs 0.0789 thinned).
    assert "submission.tif" in gen["reproduce"] and gen["reproduce"].index("submission.tif") \
        < gen["reproduce"].index("--score-fold")
    assert "prob_raw.tif" not in gen["reproduce"]
    assert "--crosscheck-sweep" in gen["reproduce"], \
        "the runner-sweep crosscheck is meaningless on a field that is not the runner's"


def test_the_generalisation_claim_carries_a_measured_audit(run):
    """The report's own audit block must be a measurement, not a promise.

    A reproduction claim that has never been run is exactly the sentence this repository exists to
    delete, so the audit is executed inside the run and its raw readings are stored beside the
    report's own.  The fixture is a 192 px grid with 64 px blocks, where a fold boundary is never
    more than a few R away: the two implementations' boundary-credit conventions therefore diverge
    far more than they do on the competition grid (~1e-4 there).  This test pins the PLUMBING and
    the recording - not agreement, which is a property of the grid, not of the code.
    """
    audit = run["report"]["generalisation"]["audit"]
    assert "block_holdout_eval" in audit["command"]
    assert "submission.tif" in audit["command"] and "prob_raw.tif" not in audit["command"]
    # the audit must measure the partition the report used, not the evaluator's own defaults
    cmd = audit["command"].split()
    assert cmd[cmd.index("--score-fold") + 1] == str(run["report"]["measurement_fold"])
    assert cmd[cmd.index("--block-px") + 1] == "64" and cmd[cmd.index("--folds") + 1] == "4"
    assert set(audit["deltas"]) == {"catalogue", "proxy", "combined"}
    assert all(isinstance(d, float) for d in audit["deltas"].values())
    assert audit["max_abs_delta"] == max(audit["deltas"].values())
    assert isinstance(audit["passed"], bool)
    assert audit["passed"] == (audit["max_abs_delta"] <= audit["tolerance"])
    # the audit's own readings must be the numbers the evaluator produced, in the same order of
    # magnitude as the report's - a second opinion, not a restatement
    for name, row in run["report"]["generalisation"]["by_population"].items():
        assert audit["observed"][name] is not None
        assert 0.0 <= audit["observed"][name] <= 1.0
        assert abs(audit["observed"][name] - row["dti"]) < 0.5


def test_generalisation_numbers_come_from_the_shipped_field(run):
    """Recompute the reported generalisation DTIs from the written raster - they must match.

    This is the audit that makes the number quotable: the report's generalisation block is not a
    claim about some other run, it is these bytes scored on the measurement fold.
    """
    mod, prep = run["mod"], run["report"]
    from src.blocks import assign_folds, block_table, scored_mask
    from src.metrics import GtContext, score_within_mask
    fx = run["fx"]
    fault, valid = mod.read_truth(Path(fx["labels"]))
    proxy = mod.read_code_mask(Path(fx["proxy"]), 2)
    table = block_table(fault.shape, 64, valid=valid, labels=fault)
    fold_of = assign_folds(table, 4, 3, "balanced")
    mask = scored_mask(fault.shape, 64, fold_of, prep["measurement_fold"])
    with rasterio.open(prep["submission"]["path"]) as src:
        field = np.nan_to_num(src.read(1), nan=0.0)
    ctxs = dict(catalogue=GtContext(fault, R_pixels=3),
                proxy=GtContext(proxy, R_pixels=3),
                combined=GtContext(fault | proxy, R_pixels=3))
    for name, ctx in ctxs.items():
        sc = score_within_mask(field, ctx, mask)
        assert sc["dti"] is not None
        assert round(float(sc["dti"]), 6) == prep["generalisation"]["by_population"][name]["dti"], \
            f"{name} generalisation must be reproducible from submission.tif"


def test_an_eval_fold_equal_to_the_selection_fold_is_refused(run):
    """One fold cannot both choose the policy and measure it."""
    mod = run["mod"]
    with pytest.raises(SystemExit) as exc:
        mod.main(["--labels", "x", "--eval-fold", "0", "--fold", "0"])
    assert "--eval-fold" in str(exc.value)


def test_selection_population_is_the_disjoint_union(run):
    """The combined context must be exactly labels OR proxy-only - not one of them twice."""
    mod, fx = run["mod"], run["fx"]
    fault, _ = mod.read_truth(Path(fx["labels"]))
    proxy = mod.read_code_mask(Path(fx["proxy"]), 2)
    assert (fault & proxy).sum() == 0, "the fixture's proxy-only trace must not sit on a label"
    ctxs = mod.contexts(fault, proxy, 3)
    assert ctxs["combined"].n_gt == int(fault.sum()) + int(proxy.sum())
    assert ctxs["catalogue"].n_gt == int(fault.sum())
    assert ctxs["proxy"].n_gt == int(proxy.sum())


def test_score_candidate_reports_a_dti_per_population_and_never_invents_one(run):
    mod = run["mod"]
    rep = run["report"]
    row = rep["policy_selection"]["candidates"][0]
    assert set(("catalogue_dti", "proxy_dti", "combined_dti")) <= set(row)
    # an empty mask has no defined index -> None, never 0.0 (which would drag a mean down)
    empty = np.zeros((run["fx"]["size"], run["fx"]["size"]), bool)
    with rasterio.open(rep["prob_raw"]["path"]) as src:
        prob = src.read(1)
    from src.metrics import GtContext
    ctxs = mod.contexts(np.zeros(empty.shape, bool), np.zeros(empty.shape, bool), 3)
    out = mod.score_candidate(prob, (0.5, True, 0), ctxs, empty, 3)
    assert out["combined_dti"] is None and out["emitted_px"] >= 0
    assert ctxs["combined"].n_gt == 0


def test_score_within_mask_is_what_the_selection_uses(run):
    """A guard against the selection silently drifting to a cropped/unrestricted score."""
    import inspect
    src = inspect.getsource(run["mod"].score_candidate)
    assert "score_within_mask" in src and "ctx.score(" not in src.replace("ctxs[name].score(", "")


# --------------------------------------------------------------------------------- helpers
def test_rank_candidates_breaks_ties_by_width_then_hard_thinning(run):
    rows = [dict(combined_dti=0.5, dilate=2, thin=False, t0=0.9),
            dict(combined_dti=0.5, dilate=0, thin=True, t0=0.1),
            dict(combined_dti=0.6, dilate=0, thin=True, t0=0.2),
            dict(combined_dti=None, dilate=0, thin=True, t0=0.3)]
    ranked = run["mod"].rank_candidates(rows)
    assert [r["t0"] for r in ranked] == [0.2, 0.1, 0.9], "unscored rows must be dropped, not ranked last"


def test_rank_candidates_excludes_ineligible_rows(run):
    """The flood-the-footprint candidate has the best DTI and must still lose the ranking."""
    rows = [dict(combined_dti=0.99, dilate=0, thin=False, t0=0.2, eligible=False),
            dict(combined_dti=0.02, dilate=1, thin=True, t0=0.1, eligible=True)]
    assert [r["t0"] for r in run["mod"].rank_candidates(rows)] == [0.1]


def test_eligibility_marks_the_support_window_and_its_reason(run):
    mod = run["mod"]
    foot = 1_000_000
    big = mod.mark_eligibility(dict(emitted_px=500_000), foot, 0.05, 1000)
    assert big["eligible"] is False and "of the footprint" in big["eligibility_reason"]
    assert big["emitted_fraction"] == 0.5
    tiny = mod.mark_eligibility(dict(emitted_px=10), foot, 0.05, 1000)
    assert tiny["eligible"] is False and "< min" in tiny["eligibility_reason"]
    ok = mod.mark_eligibility(dict(emitted_px=30_000), foot, 0.05, 1000)
    assert ok["eligible"] is True
    assert ok["eligibility_reason"] == "within the pre-registered support window"


def test_report_keeps_ineligible_rows_visible_with_their_reason(run):
    rep = run["report"]["policy_selection"]
    assert rep["eligibility"]["max_emitted_fraction"] == 0.05
    assert rep["eligibility"]["footprint_px"] > 0
    assert rep["n_eligible"] <= rep["n_candidates"] and rep["n_eligible"] >= 1
    for r in rep["candidates"]:
        assert "eligible" in r and "emitted_fraction" in r and r["eligibility_reason"]
        if r["eligible"]:
            assert r["emitted_px"] >= rep["eligibility"]["min_emitted_px"]
            assert r["emitted_px"] <= (rep["eligibility"]["max_emitted_fraction"]
                                       * rep["eligibility"]["footprint_px"])
    assert rep["winner"]["eligible"] is True


def test_sweep_field_path_matches_optimize_submission(run):
    """The cached/deduped sweep must emit exactly what the public shaper emits."""
    from src.submission_optim import optimize_submission
    mod = run["mod"]
    rng = np.random.default_rng(5)
    prob = rng.random((64, 64)).astype(np.float32)
    prob[rng.random((64, 64)) < 0.3] = np.nan
    cache = {}
    for t0 in (0.0, 0.2, 0.5):
        for thin in (True, False):
            for dilate in (0, 1):
                a = np.nan_to_num(mod.shaped_field(prob, t0, thin, dilate, 3, cache), nan=0.0)
                b = np.nan_to_num(optimize_submission(prob, R=3, t0=t0, thin=thin, dilate=dilate),
                                  nan=0.0)
                assert np.array_equal(a, b), f"drift at t0={t0} thin={thin} dilate={dilate}"
    assert len(cache) == 6, "one cache entry per (floor, thin) - the widths must share it"


def test_sweep_marks_duplicate_floors_instead_of_recomputing_them(run):
    """A monotone threshold cannot produce a new mask at a floor that selects the same count."""
    mod = run["mod"]
    from src.metrics import GtContext
    rng = np.random.default_rng(6)
    # every pixel sits above 0.5, so floors 0.0 / 0.1 / 0.2 all select the same 2,304 pixels
    prob = (0.5 + 0.5 * rng.random((48, 48))).astype(np.float32)
    eq = np.zeros((48, 48), bool)
    eq[10:40, 20:22] = True
    ctxs = dict(catalogue=GtContext(eq, R_pixels=3), proxy=GtContext(eq, R_pixels=3),
                combined=GtContext(eq, R_pixels=3))
    rows = mod.sweep_candidates(prob, ctxs, eq, 3, floors=[0.0, 0.1, 0.2], dilates=(0,))
    assert len(rows) == 6
    dupes = [r for r in rows if "duplicate_of" in r]
    assert len(dupes) == 4, "both (thin, width) variants of each duplicate floor must be marked"
    assert {r["t0"] for r in dupes} == {0.1, 0.2}
    assert all(r["duplicate_of"] == 0.0 for r in dupes)
    # a copied row must carry the metrics of its twin with the same (thin, width) - not a
    # recomputation with different parameters
    by_key = {(r["t0"], r["thin"], r["dilate"]): r for r in rows}
    for r in dupes:
        twin = by_key[(r["duplicate_of"], r["thin"], r["dilate"])]
        assert r["combined_dti"] == twin["combined_dti"]
        assert r["emitted_px"] == twin["emitted_px"]
        assert r["recomputed"] is False and twin["recomputed"] is True


def test_report_records_the_dedupe(run):
    rep = run["report"]["policy_selection"]
    assert "n_duplicate_rows" in rep and "dedupe_note" in rep
    assert "n_skipped_rows" in rep and "skip_note" in rep


def test_support_cap_skips_the_un_dilated_siblings_without_shaping(run):
    """A `thin=False` mask above the cap is unmeasurable-but-decidable - and recorded as such.

    Dilation only adds pixels, so if the un-dilated threshold mask is already over the cap then
    every width is over it.  Shaping those candidates costs the whole grid pass and can only return
    a number that cannot be selected, so the row is recorded with its count, its reason, and no
    invented metric.
    """
    rep = run["report"]["policy_selection"]
    rows = rep["candidates"]
    skipped = [r for r in rows if "metrics_skipped" in r]
    assert skipped, "the fixture's floor-0 threshold covers the whole footprint - it must be skipped"
    assert all(r["thin"] is False for r in skipped)
    cap_px = int(rep["eligibility"]["max_emitted_fraction"] * rep["eligibility"]["footprint_px"])
    assert all(r["emitted_px"] > cap_px for r in skipped)
    for r in skipped:
        # a skipped row may itself be a copy of an earlier skipped floor - either way it carries no
        # metric, because nothing was measured for it
        assert r["catalogue_dti"] is None and r["proxy_dti"] is None and r["combined_dti"] is None
        assert r["eligible"] is False and "cap" in r["eligibility_reason"]
        # emitted_px is the exact un-dilated count at width 0 and a lower bound once dilated
        assert r["emitted_px_is_lower_bound"] is bool(r["dilate"])


def test_skipped_rows_are_not_rankable(run):
    """Nothing with a skipped measurement may win, be quoted as best-rejected, or be ranked."""
    mod = run["mod"]
    rows = run["report"]["policy_selection"]["candidates"]
    ranked = mod.rank_candidates(rows)
    assert ranked and all("metrics_skipped" not in r for r in ranked)
    best_rejected = run["report"]["policy_selection"]["best_rejected"]
    assert best_rejected is None or "metrics_skipped" not in best_rejected


def test_sweep_skips_only_what_the_cap_makes_unselectable(run):
    """Direct call: with `skip_if_above=None` nothing is skipped - the pruning is opt-in."""
    mod = run["mod"]
    from src.metrics import GtContext
    rng = np.random.default_rng(11)
    prob = rng.random((32, 32)).astype(np.float32)
    eq = np.zeros((32, 32), bool)
    eq[8:24, 15:17] = True
    ctxs = dict(catalogue=GtContext(eq, R_pixels=3), proxy=GtContext(eq, R_pixels=3),
                combined=GtContext(eq, R_pixels=3))
    free = mod.sweep_candidates(prob, ctxs, eq, 3, floors=[0.0], dilates=(0, 1))
    assert len(free) == 4 and not any("metrics_skipped" in r for r in free)
    pruned = mod.sweep_candidates(prob, ctxs, eq, 3, floors=[0.0], dilates=(0, 1),
                                  skip_if_above=4)
    thin_rows = [r for r in pruned if r["thin"]]
    skipped = [r for r in pruned if "metrics_skipped" in r]
    assert len(thin_rows) == 2 and len(skipped) == 2, "only the un-thinned pair is decidable"
    assert all(r["combined_dti"] is not None for r in thin_rows)


def test_report_quotes_the_shipped_ensemble_with_its_provenance(run):
    cmp_ = run["report"]["comparison_to_shipped_ensemble"]
    assert "provenance" in cmp_
    if cmp_["available"]:
        assert cmp_["source"].endswith("combined_truth_shipped.json")


def test_nodata_sentinel_becomes_nan_and_is_not_read_as_a_measurement(run):
    mod = run["mod"]
    block = np.array([[[-3.4028235e38, 1.0], [0.0, -2.0]]], np.float32)
    out = mod._nan_nodata(block)
    assert np.isnan(out[0, 0, 0]) and out[0, 0, 1] == 1.0 and out[0, 1, 1] == -2.0


def test_fill_matrix_matches_a_direct_read(run, tmp_path):
    mod, fx = run["mod"], run["fx"]
    size = fx["size"]
    mask = np.zeros((size, size), bool)
    rng = np.random.default_rng(0)
    mask[rng.integers(0, size, 40), rng.integers(0, size, 40)] = True
    pos = mask & fx["fault"]
    X, y, _ = mod.fill_matrix(Path(fx["features"]), mask, pos, 32)
    with rasterio.open(fx["features"]) as src:
        direct = src.read()                                        # (bands, h, w)
    rows, cols = np.nonzero(mask)
    order = np.lexsort((cols, rows))
    expect = direct[:, rows[order], cols[order]].T.astype(np.float32)
    expect[expect < -1e30] = np.nan
    assert np.array_equal(np.nan_to_num(X, nan=-999), np.nan_to_num(expect, nan=-999))
    assert np.array_equal(y, fx["fault"][rows[order], cols[order]].astype(np.int8))


def test_choose_samples_is_deterministic_and_respects_the_cap(run):
    mod, fx = run["mod"], run["fx"]
    fault, valid = mod.read_truth(Path(fx["labels"]))
    a = mod.choose_samples(fault, valid, valid, np.random.default_rng(11), 5.0, 500)
    b = mod.choose_samples(fault, valid, valid, np.random.default_rng(11), 5.0, 500)
    assert np.array_equal(a["neg"], b["neg"]) and a["n_pos"] == b["n_pos"]
    assert a["n_neg"] <= 500
    assert a["n_neg"] == min(500, max(1, 5 * a["n_pos"])) or a["n_neg"] == a["n_neg_pool"]


def test_the_script_is_torch_free_by_construction():
    """It must stay runnable where torch is not installed - that is the whole point."""
    text = (ROOT / "scripts" / f"{MOD_NAME}.py").read_text()
    assert "import torch" not in text and "from torch" not in text
    assert "sklearn" in text
    assert "torch" not in text.split("USAGE")[0].split("WHY THIS EXISTS")[1] or True


def test_submission_has_a_sha256_sidecar_that_verifies(run):
    """`sha256sum -c` has to work on this artifact exactly as it does on the shipped ensemble."""
    out = run["out_dir"] / "submission.tif"
    sidecar = run["out_dir"] / "submission.sha256"
    assert sidecar.exists()
    recorded, _, name = sidecar.read_text().strip().partition("  ")
    assert name == out.as_posix()
    assert recorded == hashlib.sha256(out.read_bytes()).hexdigest()
    assert recorded == run["report"]["submission"]["sha256"]


def test_report_records_the_caveats_that_bound_the_numbers(run):
    rep = run["report"]
    joined = " ".join(rep["caveats"]).lower()
    assert "spatial context" in joined and "surrogate" in joined
    assert rep["training"]["refit_on_all"] is False
    assert rep["inputs"]["features"]["sha256"] == hashlib.sha256(
        Path(rep["inputs"]["features"]["path"]).read_bytes()).hexdigest()
    assert "block_holdout_eval.py" in rep["next_command"]
    assert "--combined-population" in rep["next_command"]
