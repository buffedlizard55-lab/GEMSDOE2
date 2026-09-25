"""scripts/read_landed_reports.py: the reading of the landed runner reports must be derivation only.

The script exists because session 18 left one instruction - "read the four landed reports" - and
because the committed paired contrast bootstrapped only ONE population: the new-fault-like proxy,
whose truth is cut from the very raster the pseudo-labels came from.  These tests pin the three
properties that make the reading trustworthy:

1. EXACTNESS - every DTI is either copied from a committed report or recomposed from that report's
   own per-block components (TP_w/FP_w/N_w are sums, so no re-scoring is involved), and the
   recomposition is cross-checked against the runner's committed paired contrast;
2. DERIVATION - the policy label comes from the decision record, the expected folds from the
   training config and the trigger params, the pseudo-label source (and therefore the circularity
   of the proxy arm) from the config YAML, and the verdict from the two measured signs.  Nothing is
   typed as a fact;
3. LOUDNESS - a missing fold, a missing arm, a pruned candidate or a report that disagrees with
   itself stops the run instead of printing a partial table that looks complete.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALPHA, BETA, EPS = 0.2, 0.8, 1e-7
POLICY_LABEL = "floor0.1_w0px"


def _mod():
    spec = importlib.util.spec_from_file_location("rlr", ROOT / "scripts" / "read_landed_reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# -------------------------------------------------------------------------------------- builders
def _row(per_block, label=POLICY_LABEL, floor=0.1, width=0, support=1000):
    """A candidate row whose global numbers are the SUM of its per-block components."""
    tp = sum(b["TP_w"] for b in per_block)
    fp = sum(b["FP_w"] for b in per_block)
    fn = sum(b["FN_w"] for b in per_block)
    ngt = sum(b["n_gt"] for b in per_block)
    return dict(label=label, floor=floor, width=width, support_px=support,
                global_dti=round(tp / (tp + ALPHA * fp + BETA * fn + EPS), 6),
                TP_w=tp, FP_w=fp, FN_w=fn, n_gt=ngt, pred_px=support,
                emission_key=label, duplicate_of=None, blocks=per_block)


def _blk(i, tp, fp, fn, ngt):
    denom = tp + ALPHA * fp + BETA * fn + EPS
    return dict(block=i, dti=round(tp / denom, 6) if ngt else None, TP_w=tp, FP_w=fp, FN_w=fn,
                n_gt=ngt, pred_px=int(tp + fp), scoreable=bool(ngt))


def _report(rows_by_pop, blocks=(3, 4), block_px=512, n_boot=200, seed=0, units=8,
            proxy_path="data/evidence/proxy/proxy_catalogue.tif", score_fold=0, complement=False,
            partition_seed=0, n_folds=4, mode="balanced"):
    return dict(
        generated_by="scripts/block_holdout_eval.py",
        metric=dict(alpha=ALPHA, beta=BETA, eps=EPS, R_pixels=3),
        blocks=dict(block_px=block_px, n_blocks=56, n_blocks_scoreable=len(blocks),
                    partition=dict(block_px=block_px, seed=partition_seed, n_folds=n_folds,
                                   mode=mode)),
        restriction=dict(mode="fold_heldout", score_fold=score_fold, complement=complement,
                         blocks=list(blocks),
                         blocks_scoreable=len(blocks), note="synthetic"),
        populations={pop: dict(n_gt=sum(r["n_gt"] for r in rows), alpha=ALPHA, beta=BETA,
                               R_pixels=3, eps=EPS, candidates=list(rows))
                     for pop, rows in rows_by_pop.items()},
        bootstrap=dict(n=n_boot, seed=seed, reliability=dict(
            resampling_units=units, min_units_for_a_readable_ci=12,
            interval_readable=bool(units >= 12), note="synthetic")),
        inputs=dict(pred="outputs/prob_raw.tif", labels="data/labels.tif", proxy=proxy_path),
    )


def _gap(fold, held, trained, blocks_held=8, blocks_trained=26, candidate=POLICY_LABEL):
    return dict(fold=fold, candidate=candidate, held_out_dti=held,
                held_out_ci95=[round(held - 0.02, 6), round(held + 0.02, 6)],
                held_out_blocks=blocks_held, trained_on_dti=trained,
                trained_on_ci95=[round(trained - 0.02, 6), round(trained + 0.02, 6)],
                trained_on_blocks=blocks_trained,
                generalisation_gap=round(trained - held, 6), reading="synthetic")


DECISION = dict(verdict=dict(best_measured_candidate=dict(
    policy="sweep_best_t0_0.1_width0px", measured_dti=0.136452)))


def _write_yaml(path: Path, pseudo_path="data/evidence/proxy/proxy_catalogue.tif", code=2,
                weight=1.0):
    path.write_text(
        "data:\n  feature_path: data/training_features.tif\n"
        f"  pseudo_label_path: \"{pseudo_path}\"\n  pseudo_code: {code}\n"
        f"training:\n  holdout: spatial_blocks\n  block_fold: 0\n  seed: 46\n"
        f"  pseudo_weight: {weight}\n")
    return path


# --------------------------------------------------------------------------- derivation, not typing
def test_the_policy_label_is_parsed_from_the_decision_record():
    m = _mod()
    pol = m.adopted_policy_label(DECISION)
    assert pol == dict(floor=0.1, width=0, policy="sweep_best_t0_0.1_width0px",
                       candidate_label="floor0.1_w0px", measured_dti=0.136452,
                       source="data/evidence/emission_decision.json")


def test_an_unparseable_decision_record_stops_the_run():
    m = _mod()
    with pytest.raises(SystemExit):
        m.adopted_policy_label(dict(verdict=dict(best_measured_candidate=dict(policy="handpicked"))))
    with pytest.raises(SystemExit):
        m.adopted_policy_label({})


def test_expected_folds_are_read_from_the_config_and_the_trigger(tmp_path):
    m = _mod()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training:\n  holdout: spatial_blocks\n  block_px: 512\n  block_folds: 4\n")
    params = tmp_path / "params"
    params.write_text("# comment\nSUBMISSION=x.tif\nTRAIN_FOLDS=1,2,3\nSEED=0\n")
    got = m.expected_folds(cfg, params)
    assert got["block_folds"] == 4 and got["all_folds"] == [0, 1, 2, 3]
    assert got["dispatched"] == [1, 2, 3]
    assert got["sources"]["config"].endswith("cfg.yaml")


def test_the_circularity_flag_is_derived_from_the_config(tmp_path):
    """The proxy arm's truth and the pseudo-labels come from the same raster - the config says so."""
    m = _mod()
    circular = m.pseudo_provenance(_write_yaml(tmp_path / "a.yaml"))
    assert circular["circular_source"] is True
    assert circular["pseudo_label_path"] == "data/evidence/proxy/proxy_catalogue.tif"
    assert circular["pseudo_code"] == 2 and circular["pseudo_weight"] == 1.0
    assert circular["holdout"] == "spatial_blocks" and circular["seed"] == 46
    assert "same catalogue" in circular["circular_source_meaning"]

    other = m.pseudo_provenance(_write_yaml(tmp_path / "b.yaml",
                                            pseudo_path="data/external/other_catalogue.tif"))
    assert other["circular_source"] is False
    assert "not the proxy truth raster" in other["circular_source_meaning"]

    wrong_code = m.pseudo_provenance(_write_yaml(tmp_path / "c.yaml", code=1))
    assert wrong_code["circular_source"] is False, "code 1 is the part the labels already contain"


# --------------------------------------------------------------------------- gap summary
def test_the_gap_summary_is_arithmetic_and_names_the_folds_that_are_missing(tmp_path):
    m = _mod()
    for fold, held, trained in ((0, 0.0806, 0.0920), (1, 0.0789, 0.0719)):
        (tmp_path / f"fold{fold}_generalisation_gap.json").write_text(
            json.dumps(_gap(fold, held, trained)))
    (tmp_path / "fold0_heldout.json").write_text(json.dumps(
        _report({"proxy_only": [_row([_blk(3, 10, 100, 20, 30)])]})))
    gaps = m.load_fold_gaps(tmp_path)
    assert [g["_fold"] for g in gaps] == [0, 1]
    summary = m.summarise_fold_gaps(gaps, dict(block_folds=4, all_folds=[0, 1, 2, 3],
                                               dispatched=[1, 2, 3]), tmp_path)
    assert summary["n_folds_committed"] == 2
    assert summary["folds_missing"] == [2, 3] and summary["complete"] is False
    assert summary["gap"]["mean"] == pytest.approx((0.0114 + (-0.0070)) / 2, abs=1e-6)
    assert summary["gap"]["min"] == pytest.approx(-0.0070, abs=1e-6)
    assert summary["gap"]["max"] == pytest.approx(0.0114, abs=1e-6)
    assert summary["gap"]["folds_with_positive_gap"] == 1
    assert summary["gap"]["folds_with_negative_gap"] == 1
    assert summary["interval_readability"]["folds_with_a_readable_ci"] == 0
    assert summary["per_fold"][0]["source"] == "data/evidence/block_holdout/fold0_generalisation_gap.json" \
        or summary["per_fold"][0]["source"].endswith("fold0_generalisation_gap.json")


def test_a_fold_that_disagrees_with_itself_is_refused(tmp_path):
    """gap must equal trained_on - held_out; a file that says otherwise is not summarised."""
    m = _mod()
    bad = _gap(0, 0.08, 0.09)
    bad["generalisation_gap"] = 0.5                       # inconsistent on purpose
    (tmp_path / "fold0_generalisation_gap.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="disagrees with itself"):
        m.summarise_fold_gaps(m.load_fold_gaps(tmp_path), dict(all_folds=[0]), tmp_path)


def test_no_gap_files_is_an_error_not_an_empty_table(tmp_path):
    m = _mod()
    with pytest.raises(SystemExit, match="nothing to summarise"):
        m.summarise_fold_gaps(m.load_fold_gaps(tmp_path), dict(all_folds=[0, 1, 2, 3]), tmp_path)


# --------------------------------------------------------------------------- paired contrast
def test_the_paired_contrast_is_hand_computable_on_one_block(tmp_path):
    m = _mod()
    base = _report({"proxy_only": [_row([_blk(3, 100.0, 1000.0, 200.0, 300)], support=900)],
                    "labels": [_row([_blk(3, 200.0, 1000.0, 100.0, 300)], support=900)]})
    pseudo = _report({"proxy_only": [_row([_blk(3, 250.0, 3000.0, 50.0, 300)], support=3000)],
                      "labels": [_row([_blk(3, 120.0, 3000.0, 180.0, 300)], support=3000)]})
    got = m.paired(base["populations"]["proxy_only"]["candidates"][0],
                   pseudo["populations"]["proxy_only"]["candidates"][0],
                   ALPHA, BETA, EPS, n_boot=500, seed=0,
                   names=("baseline_no_pseudo", "pseudo"))
    tp, fp, fn = 100.0, 1000.0, 200.0
    want_ref = tp / (tp + ALPHA * fp + BETA * fn + EPS)
    assert got["reference_dti"] == pytest.approx(round(want_ref, 6), abs=1e-6)
    assert got["contrast"] == pytest.approx(got["candidate_dti"] - got["reference_dti"], abs=1e-12)
    # one resampling unit -> every bootstrap draw is that block -> P is degenerate and exact
    assert got["bootstrap"]["prob_candidate_beats_reference"] == 1.0
    assert got["recomposition_exact"] == {"baseline_no_pseudo": True, "pseudo": True}
    assert got["candidate_support_px"] == 3000 and got["reference_support_px"] == 900


def test_the_verdict_is_derived_from_the_two_signs(tmp_path):
    m = _mod()

    def arms(proxy_b, proxy_p, labels_b, labels_p, extra_pop=None):
        pops_b = {"proxy_only": [_row([_blk(3, proxy_b[0], proxy_b[1], proxy_b[2], 300)])],
                  "labels": [_row([_blk(3, labels_b[0], labels_b[1], labels_b[2], 300)])]}
        pops_p = {"proxy_only": [_row([_blk(3, proxy_p[0], proxy_p[1], proxy_p[2], 300)])],
                  "labels": [_row([_blk(3, labels_p[0], labels_p[1], labels_p[2], 300)])]}
        if extra_pop:
            pops_b["combined"] = [_row([_blk(3, *extra_pop[0], 300)])]
            pops_p["combined"] = [_row([_blk(3, *extra_pop[1], 300)])]
        return _report(pops_b), _report(pops_p)

    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))

    def verdict_for(**kw):
        b, p = arms(**kw)
        for name, rep in (("fold0_heldout.json", b), ("fold0_complement.json", b)):
            (base_dir / name).write_text(json.dumps(rep))
        for name, rep in (("fold0_heldout.json", p), ("fold0_complement.json", p)):
            (pseudo_dir / name).write_text(json.dumps(rep))
        return m.contrast_report(0, DECISION, prov, base_dir, pseudo_dir)

    # proxy gains, catalogue loses -> the trade-off this script exists to surface
    r = verdict_for(proxy_b=(100.0, 1000.0, 200.0), proxy_p=(250.0, 1200.0, 50.0),
                    labels_b=(200.0, 1000.0, 100.0), labels_p=(120.0, 1200.0, 180.0))
    assert r["verdict"] == "TRADE_OFF_PROXY_GAINS_CATALOGUE_LOSSES"
    assert r["shippable_evidence"] is False and "FIELD_SELECTION_RULE" in r["shipping_note"]

    # the mirror image
    r = verdict_for(proxy_b=(250.0, 1000.0, 50.0), proxy_p=(100.0, 1200.0, 200.0),
                    labels_b=(120.0, 1000.0, 180.0), labels_p=(200.0, 1200.0, 100.0))
    assert r["verdict"] == "TRADE_OFF_CATALOGUE_GAINS_PROXY_LOSSES"

    # both gain / both lose
    r = verdict_for(proxy_b=(100.0, 1000.0, 200.0), proxy_p=(250.0, 1000.0, 50.0),
                    labels_b=(100.0, 1000.0, 200.0), labels_p=(250.0, 1000.0, 50.0))
    assert r["verdict"] == "GAIN_ON_BOTH_POPULATIONS"
    r = verdict_for(proxy_b=(250.0, 1000.0, 50.0), proxy_p=(100.0, 1000.0, 200.0),
                    labels_b=(250.0, 1000.0, 50.0), labels_p=(100.0, 1000.0, 200.0))
    assert r["verdict"] == "LOSS_ON_BOTH_POPULATIONS"

    # when the combined (Phase-2-like) population is measured, IT decides - not the two components
    r = verdict_for(proxy_b=(250.0, 1000.0, 50.0), proxy_p=(100.0, 1200.0, 200.0),
                    labels_b=(120.0, 1000.0, 180.0), labels_p=(200.0, 1200.0, 100.0),
                    extra_pop=((300.0, 2000.0, 200.0), (400.0, 2000.0, 100.0)))
    assert r["verdict"] == "GAIN_ON_THE_COMBINED_SURROGATE"
    assert "combined" in r["verdict_reason"]


def test_a_missing_arm_is_reported_missing_not_invented(tmp_path):
    m = _mod()
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    (base_dir / "fold0_heldout.json").write_text(json.dumps(
        _report({"proxy_only": [_row([_blk(3, 100.0, 1000.0, 200.0, 300)])],
                 "labels": [_row([_blk(3, 100.0, 1000.0, 200.0, 300)])]})))
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    r = m.contrast_report(0, DECISION, prov, base_dir, pseudo_dir)
    assert r["arms"]["heldout"]["status"] == "MISSING"
    assert r["verdict"] == "NOT_MEASURABLE"
    assert "both arms must be committed" in r["arms"]["heldout"]["note"]


def test_arms_scored_on_different_blocks_are_refused(tmp_path):
    m = _mod()
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    b = _report({"proxy_only": [_row([_blk(3, 100.0, 1000.0, 200.0, 300)])],
                 "labels": [_row([_blk(3, 100.0, 1000.0, 200.0, 300)])]})
    p = _report({"proxy_only": [_row([_blk(9, 100.0, 1000.0, 200.0, 300)])],
                 "labels": [_row([_blk(9, 100.0, 1000.0, 200.0, 300)])]}, blocks=(9,))
    (base_dir / "fold0_heldout.json").write_text(json.dumps(b))
    (pseudo_dir / "fold0_heldout.json").write_text(json.dumps(p))
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    with pytest.raises(SystemExit, match="different blocks"):
        m.contrast_report(0, DECISION, prov, base_dir, pseudo_dir)


def test_a_pruned_candidate_says_what_would_fix_it(tmp_path):
    m = _mod()
    row = _row([_blk(3, 100.0, 1000.0, 200.0, 300)])
    row["blocks"] = []                                    # pruned as a duplicate emission
    row["duplicate_of"] = "floor0.15_w0px"
    with pytest.raises(SystemExit, match="keep-duplicate-rows"):
        m.candidate_row({"populations": {"proxy_only": {"candidates": [row]}}},
                        "proxy_only", POLICY_LABEL)


def test_main_refuses_to_write_when_the_recomposition_is_inexact(tmp_path, monkeypatch):
    """A derived number that disagrees with its own committed components must stop the run."""
    m = _mod()
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    good = _row([_blk(3, 100.0, 1000.0, 200.0, 300)])
    bad = _row([_blk(3, 100.0, 1000.0, 200.0, 300)])
    bad["global_dti"] = 0.9                               # inconsistent with its own components
    for name in ("fold0_heldout.json", "fold0_complement.json"):
        (base_dir / name).write_text(json.dumps(
            _report({"proxy_only": [good], "labels": [good]})))
        (pseudo_dir / name).write_text(json.dumps(
            _report({"proxy_only": [bad], "labels": [bad]})))
    (base_dir / "fold0_generalisation_gap.json").write_text(json.dumps(_gap(0, 0.08, 0.09)))
    monkeypatch.setattr(m, "BH", base_dir)
    monkeypatch.setattr(m, "PL", pseudo_dir)
    monkeypatch.setattr(m, "PSEUDO_CONFIG", _write_yaml(tmp_path / "cfg.yaml"))
    monkeypatch.setattr(m, "DECISION", (tmp_path / "decision.json"))
    (tmp_path / "decision.json").write_text(json.dumps(DECISION))
    with pytest.raises(SystemExit, match="recomposition is NOT exact"):
        m.main(["--fold", "0", "--gaps-out", str(tmp_path / "gaps.json"),
                "--contrast-out", str(tmp_path / "contrast.json"), "--quiet"])
    assert not (tmp_path / "contrast.json").exists(), "nothing may be written on a refused reading"


# --------------------------------------------------------------------------- committed evidence
COMMITTED = (ROOT / "data/evidence/pseudo_labels/fold0_paired_vs_baseline.json").exists() and \
            (ROOT / "data/evidence/block_holdout/fold0_generalisation_gap.json").exists()


@pytest.mark.skipif(not COMMITTED, reason="the landed runner reports are not in this checkout")
def test_the_recomposition_reproduces_the_runners_own_paired_contrast():
    """Same rows, same seed, same n_boot -> the sandbox reproduces the runner digit for digit."""
    m = _mod()
    contrast = m.contrast_report(0, m.load_json(m.DECISION), m.pseudo_provenance())
    assert m.self_check(contrast, 0) == [], "the recomposed contrast disagrees with the runner's"
    proxy = contrast["arms"]["heldout"]["populations"]["proxy_only"]
    committed = json.loads((ROOT / "data/evidence/pseudo_labels/fold0_paired_vs_baseline.json")
                           .read_text())
    assert proxy["reference_dti"] == pytest.approx(committed["baseline_held_out_dti"], abs=1e-6)
    assert proxy["candidate_dti"] == pytest.approx(committed["pseudo_held_out_dti"], abs=1e-6)
    assert proxy["bootstrap"]["prob_candidate_beats_reference"] == pytest.approx(
        committed["bootstrap"]["pseudo"]["prob_beats_reference"], abs=1e-9)
    assert proxy["reference_support_px"] == committed["baseline_held_out_px"]
    assert proxy["candidate_support_px"] == committed["pseudo_held_out_px"]


@pytest.mark.skipif(not COMMITTED, reason="the landed runner reports are not in this checkout")
def test_the_committed_reading_is_not_shippable_and_the_union_population_decides():
    """The finding this script exists to make visible, on the committed re-fire (run 35477119490).

    Pinned qualitatively rather than by value on purpose: two fires of the same seed, config and
    partition on different runners differed by ~0.036 on the proxy arm (+0.1036 then +0.0677), so a
    test that hard-codes one replicate's decimals would fail on the next honest re-measurement while
    proving nothing.  What must hold for ANY fire is the structure asserted here - including the
    tripwire at the end, which fails loudly the day this contrast actually reaches the adoption bar
    (docs/FIELD_SELECTION_RULE.md R3: P >= 0.95) so the docs cannot silently stay stale.
    """
    m = _mod()
    r = m.contrast_report(0, m.load_json(m.DECISION), m.pseudo_provenance())
    held = r["arms"]["heldout"]["populations"]
    # the two COMPONENT populations disagree in sign - the reason the union had to be scored at all
    assert held["proxy_only"]["contrast"] > 0, "the source-circular arm gains"
    assert held["labels"]["contrast"] < 0, "the independent (catalogue) arm loses"
    assert held["labels"]["bootstrap"]["prob_candidate_beats_reference"] < 0.05
    # the union arm exists, and it says where its two arms came from
    assert held["combined"]["status"] == "MEASURED", "the union arm must be scored, not inferred"
    src = held["combined"]["sources"]
    assert src["baseline"].endswith("fold0_heldout_combined.json"), \
        "the baseline's union numbers come from the sha256-gated re-score of its runner artifact"
    assert src["pseudo"].endswith("pseudo_labels/fold0_heldout.json")
    # and it is the population that decides the verdict
    expected = ("GAIN_ON_THE_COMBINED_SURROGATE" if held["combined"]["contrast"] > 0
                else "LOSS_ON_THE_COMBINED_SURROGATE")
    assert r["verdict"] == expected
    assert r["shippable_evidence"] is False, "this reader makes no shipping decision, ever"
    assert "FIELD_SELECTION_RULE.md" in r["shipping_note"]
    assert r["pseudo_provenance"]["circular_source"] is True
    # the memorisation check: the same populations on the blocks that fold trained on
    comp = r["arms"]["complement"]["populations"]
    assert comp["proxy_only"]["contrast"] > 0 and comp["labels"]["contrast"] < 0
    assert comp["combined"]["status"] == "MEASURED"
    assert (comp["combined"]["contrast"] > 0) != (held["combined"]["contrast"] > 0), \
        "the committed fire's two scopes disagree in sign on the union - that instability is part " \
        "of the finding, and if a future fire stabilises it the docs must be rewritten"
    # the coarse-interval caveat travels with every fold-restricted number
    assert held["proxy_only"]["interval_readable"] is False
    assert held["combined"]["interval_readable"] is False
    assert "COARSE" in r["caveat"] or "spread indicator" in r["caveat"]
    # TRIPWIRE: the committed contrast does not reach the adoption bar.  If this fires, the union
    # gain is real and STATUS/EXECUTIVE_SUMMARY/LIMITATIONS/the site all have to say so.
    b = held["combined"]["bootstrap"]
    ci = b["contrast_ci95"]
    assert b["prob_candidate_beats_reference"] < 0.95 or (ci[0] < 0 < ci[1]), \
        f"the union contrast now clears R3's bar ({b['prob_candidate_beats_reference']}, {ci}) - " \
        "the shipping story in the docs is stale and must be rewritten"


@pytest.mark.skipif(not COMMITTED, reason="the landed runner reports are not in this checkout")
def test_the_gap_summary_reads_the_committed_folds():
    m = _mod()
    s = m.summarise_fold_gaps(m.load_fold_gaps(), m.expected_folds())
    assert s["n_folds_committed"] >= 3
    assert s["expected"]["block_folds"] == 4
    for row in s["per_fold"]:
        assert row["reliability"]["interval_readable"] is False, \
            "a fold-restricted bootstrap has 8-9 units; the scorer itself calls that coarse"
        assert abs(row["generalisation_gap"] -
                   (row["trained_on_dti"] - row["held_out_dti"])) < 1e-6


@pytest.mark.skipif(not COMMITTED, reason="the landed runner reports are not in this checkout")
def test_main_writes_both_artefacts_and_check_writes_nothing(tmp_path):
    m = _mod()
    assert m.main(["--fold", "0", "--check", "--quiet"]) == 0
    gaps_out, contrast_out = tmp_path / "g.json", tmp_path / "c.json"
    assert m.main(["--fold", "0", "--gaps-out", str(gaps_out),
                   "--contrast-out", str(contrast_out), "--quiet"]) == 0
    g, c = json.loads(gaps_out.read_text()), json.loads(contrast_out.read_text())
    assert g["generated_by"] == m.__name__.replace("__main__", "") or \
        g["generated_by"] == "scripts/read_landed_reports.py"
    held_comb = c["arms"]["heldout"]["populations"]["combined"]
    assert held_comb["status"] == "MEASURED"
    assert c["verdict"] == ("GAIN_ON_THE_COMBINED_SURROGATE" if held_comb["contrast"] > 0
                            else "LOSS_ON_THE_COMBINED_SURROGATE"), \
        "once the union is scored it decides the verdict"
    assert c["policy"]["candidate_label"] == POLICY_LABEL
    assert set(c["arms"]) == {"heldout", "complement"}


# --------------------------------------------------------------------------- sibling re-scores
def _with_sha(rep, sha):
    rep.setdefault("inputs", {})["pred_grid"] = dict(sha256=sha, bytes=1, width=1, height=1)
    return rep


def _write_pair(base_dir, pseudo_dir, name, rep_b, rep_p):
    (base_dir / name).write_text(json.dumps(rep_b))
    (pseudo_dir / name).write_text(json.dumps(rep_p))


def test_a_sibling_combined_report_supplies_the_third_population(tmp_path):
    """The baseline's raw field survives only as a runner artifact, so its combined population is a
    re-score committed NEXT TO the original report - the reader must pick it up and say where the
    number came from."""
    m = _mod()
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    row = lambda tp, fp, fn: _row([_blk(3, tp, fp, fn, 300)])            # noqa: E731
    primary_b = _with_sha(_report({"proxy_only": [row(100.0, 1000.0, 200.0)],
                                   "labels": [row(200.0, 1000.0, 100.0)]}), "aaa")
    primary_p = _with_sha(_report({"proxy_only": [row(250.0, 1200.0, 50.0)],
                                   "labels": [row(120.0, 1200.0, 180.0)]}), "bbb")
    _write_pair(base_dir, pseudo_dir, "fold0_heldout.json", primary_b, primary_p)
    _write_pair(base_dir, pseudo_dir, "fold0_complement.json", primary_b, primary_p)
    # the same two fields, re-scored with the union truth (combined loses -> the verdict must flip)
    _write_pair(base_dir, pseudo_dir, "fold0_heldout_combined.json",
                _with_sha(_report({"proxy_only": [row(100.0, 1000.0, 200.0)],
                                   "labels": [row(200.0, 1000.0, 100.0)],
                                   "combined": [row(300.0, 2000.0, 300.0)]}), "aaa"),
                _with_sha(_report({"proxy_only": [row(250.0, 1200.0, 50.0)],
                                   "labels": [row(120.0, 1200.0, 180.0)],
                                   "combined": [row(400.0, 2000.0, 200.0)]}), "bbb"))
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    r = m.contrast_report(0, DECISION, prov, base_dir, pseudo_dir)
    comb = r["arms"]["heldout"]["populations"]["combined"]
    assert comb["status"] == "MEASURED"
    assert comb["sources"]["baseline"].endswith("fold0_heldout_combined.json")
    assert comb["sources"]["pseudo"].endswith("fold0_heldout_combined.json")
    assert r["arms"]["heldout"]["populations"]["proxy_only"]["sources"]["baseline"].endswith(
        "fold0_heldout.json"), "the other populations must still come from the primary reports"
    # combined gains (400/2000/200 -> 0.4167 beats 300/2000/300 -> 0.3191) while the two
    # component populations disagree, so the combined population is the one that decides
    assert comb["contrast"] > 0
    assert r["verdict"] == "GAIN_ON_THE_COMBINED_SURROGATE"


def test_a_sibling_that_is_a_different_field_is_refused(tmp_path):
    m = _mod()
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    row = _row([_blk(3, 100.0, 1000.0, 200.0, 300)])
    primary = _with_sha(_report({"proxy_only": [row], "labels": [row]}), "field-A")
    sibling = _with_sha(_report({"proxy_only": [row], "labels": [row], "combined": [row]}),
                        "field-B")
    _write_pair(base_dir, pseudo_dir, "fold0_heldout.json", primary, primary)
    (base_dir / "fold0_heldout_combined.json").write_text(json.dumps(sibling))
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    with pytest.raises(SystemExit, match="NOT the same probability field"):
        m.contrast_report(0, DECISION, prov, base_dir, pseudo_dir)


def test_strict_fails_the_run_when_the_cross_check_disagrees(tmp_path, monkeypatch, capsys):
    """--strict is what a runner step uses: an independent recomputation of the same rows must
    reproduce the runner's own committed contrast, or nothing derived from it may be committed."""
    m = _mod()
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    row = _row([_blk(3, 100.0, 1000.0, 200.0, 300)])
    rep = _report({"proxy_only": [row], "labels": [row]})
    for name in ("fold0_heldout.json", "fold0_complement.json"):
        _write_pair(base_dir, pseudo_dir, name, rep, rep)
    (base_dir / "fold0_generalisation_gap.json").write_text(json.dumps(_gap(0, 0.08, 0.09)))
    # a committed paired contrast whose numbers are NOT what the rows give
    (pseudo_dir / "fold0_paired_vs_baseline.json").write_text(json.dumps(dict(
        baseline_held_out_dti=0.5, pseudo_held_out_dti=0.6,
        bootstrap=dict(pseudo=dict(prob_beats_reference=0.5, contrast_vs_reference_p50=0.1,
                                   dti_ci95=[0.1, 0.9])))))
    monkeypatch.setattr(m, "BH", base_dir)
    monkeypatch.setattr(m, "PL", pseudo_dir)
    monkeypatch.setattr(m, "PSEUDO_CONFIG", _write_yaml(tmp_path / "cfg.yaml"))
    monkeypatch.setattr(m, "DECISION", tmp_path / "decision.json")
    (tmp_path / "decision.json").write_text(json.dumps(DECISION))
    args = ["--fold", "0", "--gaps-out", str(tmp_path / "g.json"),
            "--contrast-out", str(tmp_path / "c.json"), "--quiet"]
    with pytest.raises(SystemExit) as exc:
        m.main(args + ["--strict"])
    assert exc.value.code == 2
    assert not (tmp_path / "c.json").exists(), "a refused cross-check must not write evidence"
    assert "cross-check" in capsys.readouterr().out
    # the same inputs without --strict only warn (the sandbox reading is still useful)
    assert m.main(args) == 0
    assert (tmp_path / "c.json").exists()


def test_gaps_only_needs_no_pseudo_arm_and_writes_no_contrast(tmp_path, monkeypatch):
    """block-holdout.yml calls --gaps-only the moment a fold lands, when no pseudo arm exists.

    It must therefore not touch the emission decision record or the pseudo-label config, and it
    must not invent a contrast file: the summary is the only thing that fold can add.
    """
    m = _mod()
    d = tmp_path / "bh"
    d.mkdir()
    (d / "fold0_generalisation_gap.json").write_text(json.dumps(_gap(0, 0.0806, 0.0920)))
    (d / "fold2_generalisation_gap.json").write_text(json.dumps(_gap(2, 0.0480, 0.0619)))
    monkeypatch.setattr(m, "BH", d)
    monkeypatch.setattr(m, "PL", tmp_path / "pl")                     # never created
    monkeypatch.setattr(m, "DECISION", tmp_path / "no_such_decision.json")
    monkeypatch.setattr(m, "PSEUDO_CONFIG", tmp_path / "no_such_config.yaml")
    out = tmp_path / "summary.json"
    assert m.main(["--gaps-only", "--gaps-out", str(out), "--quiet"]) == 0
    got = json.loads(out.read_text())
    assert got["n_folds_committed"] == 2
    assert [r["fold"] for r in got["per_fold"]] == [0, 2]
    assert got["gap"]["mean"] == pytest.approx(((0.0920 - 0.0806) + (0.0619 - 0.0480)) / 2,
                                               abs=1e-6)
    assert not (tmp_path / "pl").exists(), "--gaps-only must not create a contrast file"


def test_gaps_only_still_reports_the_folds_that_have_not_landed(tmp_path, monkeypatch, capsys):
    """The summary's job is to say what is MISSING, so a partial partition is never read as a whole."""
    m = _mod()
    d = tmp_path / "bh"
    d.mkdir()
    (d / "fold0_generalisation_gap.json").write_text(json.dumps(_gap(0, 0.0806, 0.0920)))
    cfg = tmp_path / "config_block_holdout.yaml"
    cfg.write_text("training:\n  block_folds: 4\n")
    monkeypatch.setattr(m, "BH", d)
    monkeypatch.setattr(m, "HOLDOUT_CONFIG", cfg)
    monkeypatch.setattr(m, "TRIGGER_PARAMS", tmp_path / "no_params")
    monkeypatch.setattr(m, "PL", tmp_path / "pl")
    monkeypatch.setattr(m, "DECISION", tmp_path / "no_decision.json")
    monkeypatch.setattr(m, "PSEUDO_CONFIG", tmp_path / "no_config.yaml")
    out = tmp_path / "summary.json"
    assert m.main(["--gaps-only", "--gaps-out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["folds_missing"] == [1, 2, 3]
    assert got["expected"]["block_folds"] == 4
    assert got["expected"]["sources"]["config"].endswith("config_block_holdout.yaml"), \
        "the partition size must be read from the committed config, not typed into the summary"
    printed = capsys.readouterr().out
    assert "MISSING folds" in printed and "[1, 2, 3]" in printed
    assert "pseudo-label contrast" not in printed, "no contrast section without a contrast"


# --------------------------------------------------------------------- pooled multi-fold contrast
# EXECUTIVE_SUMMARY §11 item 7: one fold has 7-9 scoreable blocks (below the scorer's 12-unit
# readability bar) and its replicate noise (~0.036) swamps the +0.031 union effect. Pooling the
# four DISJOINT block folds is the only reading that can settle it. These tests pin that the pool
# is arithmetic on the committed per-block components, that it refuses to pool non-disjoint or
# mismatched folds, and that the readability bar drives the verdict.
def _pool_dirs(tmp_path, folds_blocks, *, pseudo_shift=0.0, base_shift=0.0):
    """Write a baseline+pseudo report per fold, each fold on its OWN disjoint block ids.

    folds_blocks: {fold: [block_id, ...]}. Both arms carry proxy_only/labels/combined; the pseudo
    arm's combined TP is shifted by pseudo_shift so a gain/loss can be dialled in.
    """
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(exist_ok=True), pseudo_dir.mkdir(exist_ok=True)
    for fold, ids in folds_blocks.items():
        def mk(shift):
            pops = {}
            for pop in ("proxy_only", "labels", "combined"):
                blks = [_blk(i, 100.0 + shift, 1000.0, 200.0 - shift, 300) for i in ids]
                pops[pop] = [_row(blks)]
            return _report(pops, blocks=tuple(ids), score_fold=fold)
        # give both arms a DIFFERENT field sha so the "same field" guard is not tripped
        b = mk(base_shift); p = mk(pseudo_shift)
        b["inputs"]["pred_grid"] = dict(sha256="a" * 64)
        p["inputs"]["pred_grid"] = dict(sha256="b" * 64)
        (base_dir / f"fold{fold}_heldout.json").write_text(json.dumps(b))
        (pseudo_dir / f"fold{fold}_heldout.json").write_text(json.dumps(p))
    return base_dir, pseudo_dir


def test_pooled_contrast_concatenates_disjoint_folds(tmp_path):
    m = _mod()
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    base_dir, pseudo_dir = _pool_dirs(tmp_path, {0: [3, 4], 1: [10, 11], 2: [20], 3: [30, 31, 32]},
                                      pseudo_shift=40.0)
    r = m.pooled_contrast([0, 1, 2, 3], DECISION, prov, base_dir, pseudo_dir, n_boot=200)
    comb = r["populations"]["combined"]
    assert comb["status"] == "MEASURED"
    assert comb["folds_pooled"] == [0, 1, 2, 3]
    # 2+2+1+3 = 8 scoreable blocks pooled -> above the 12 bar? no; but they must all be counted
    assert comb["resampling_units"] == 8
    assert comb["n_folds_pooled"] == 4
    # the pooled DTI is the SUM of every block's components, hand-checkable
    tp = (100.0 + 40.0) * 8
    fp = 1000.0 * 8
    fn = (200.0 - 40.0) * 8
    assert comb["pooled_candidate_dti"] == pytest.approx(round(tp / (tp + ALPHA * fp + BETA * fn + EPS), 6),
                                                         abs=1e-6)
    # pseudo TP up, FN down -> a gain
    assert comb["pooled_contrast"] > 0


def test_pooled_contrast_refuses_non_disjoint_folds(tmp_path):
    """The four folds partition the grid; a block id in two folds means the geography is resampled
    twice, which would fake precision. That must be refused, not silently pooled."""
    m = _mod()
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    base_dir, pseudo_dir = _pool_dirs(tmp_path, {0: [3, 4], 1: [4, 5]}, pseudo_shift=10.0)
    with pytest.raises(SystemExit) as e:
        m.pooled_contrast([0, 1], DECISION, prov, base_dir, pseudo_dir, n_boot=100)
    assert "disjoint" in str(e.value) or "repeat" in str(e.value)


def test_pooled_contrast_refuses_mismatched_partitions(tmp_path):
    m = _mod()
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    base_dir, pseudo_dir = _pool_dirs(tmp_path, {0: [3, 4]}, pseudo_shift=10.0)
    # a second fold whose partition seed differs -> not comparable blocks
    def mk(shift, seed):
        pops = {pop: [_row([_blk(10, 100.0 + shift, 1000.0, 200.0, 300)])]
                for pop in ("proxy_only", "labels", "combined")}
        rep = _report(pops, blocks=(10,), score_fold=1, partition_seed=seed)
        rep["inputs"]["pred_grid"] = dict(sha256=("a" if shift == 0 else "b") * 64)
        return rep
    (base_dir / "fold1_heldout.json").write_text(json.dumps(mk(0.0, 99)))
    (pseudo_dir / "fold1_heldout.json").write_text(json.dumps(mk(10.0, 99)))
    with pytest.raises(SystemExit) as e:
        m.pooled_contrast([0, 1], DECISION, prov, base_dir, pseudo_dir, n_boot=100)
    assert "partition" in str(e.value)


def test_pooled_contrast_refuses_the_same_field_on_both_arms(tmp_path):
    """A contrast of a probability field with itself is not a measurement."""
    m = _mod()
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    base_dir, pseudo_dir = tmp_path / "bh", tmp_path / "pl"
    base_dir.mkdir(), pseudo_dir.mkdir()
    pops = {pop: [_row([_blk(3, 100.0, 1000.0, 200.0, 300)])]
            for pop in ("proxy_only", "labels", "combined")}
    rep = _report(pops, blocks=(3,), score_fold=0)
    rep["inputs"]["pred_grid"] = dict(sha256="c" * 64)
    (base_dir / "fold0_heldout.json").write_text(json.dumps(rep))
    (pseudo_dir / "fold0_heldout.json").write_text(json.dumps(rep))   # SAME sha
    with pytest.raises(SystemExit) as e:
        m.pooled_contrast([0], DECISION, prov, base_dir, pseudo_dir, n_boot=100)
    assert "same" in str(e.value).lower() or "itself" in str(e.value).lower()


def test_pooled_contrast_names_missing_folds_and_marks_under_power(tmp_path):
    """A fold whose arm has not landed is listed in `missing`, never dropped from the claim's
    denominator, and a pool below 12 units is marked not-readable / under-powered."""
    m = _mod()
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    base_dir, pseudo_dir = _pool_dirs(tmp_path, {0: [3, 4], 1: [10, 11]}, pseudo_shift=40.0)
    r = m.pooled_contrast([0, 1, 2, 3], DECISION, prov, base_dir, pseudo_dir, n_boot=200)
    comb = r["populations"]["combined"]
    assert comb["folds_pooled"] == [0, 1]
    assert [x["fold"] for x in comb["missing"]] == [2, 3]
    assert comb["interval_readable"] is False           # 4 blocks < 12
    assert comb["clears_pre_registered_bars"] is False
    assert r["shippable_evidence"] is False


def test_pooled_contrast_readable_pool_can_clear_the_bars(tmp_path):
    """With >=12 disjoint scoreable blocks and a real gain, the pool is readable and can clear the
    pre-registered bars -> the state that would let the field even be PROPOSED to the field gate."""
    m = _mod()
    prov = m.pseudo_provenance(_write_yaml(tmp_path / "cfg.yaml"))
    folds = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11, 12], 3: [13, 14, 15, 16]}
    base_dir, pseudo_dir = _pool_dirs(tmp_path, folds, pseudo_shift=60.0)
    r = m.pooled_contrast([0, 1, 2, 3], DECISION, prov, base_dir, pseudo_dir, n_boot=500)
    comb = r["populations"]["combined"]
    assert comb["resampling_units"] == 16 and comb["interval_readable"] is True
    assert comb["pooled_contrast"] > m.ADOPT_CONTRAST_BAR
    assert comb["bootstrap"]["prob_candidate_beats_reference"] >= m.ADOPT_P_BAR
    assert comb["clears_pre_registered_bars"] is True
    assert r["verdict"] == "POOL_CLEARS_THE_BARS_ON_THE_COMBINED_SURROGATE"
    assert r["shippable_evidence"] is True


def test_pool_reproduces_the_committed_single_fold_number(tmp_path):
    """Pooling fold 0 alone with block_mode='all' must reproduce the committed fold-0 union numbers.

    'all' sums every block of the partition, which is exactly the row's whole-footprint global_dti,
    so on one fold the pool is the identity. This is the guard that the pool does not silently
    change what a single-fold reading already established.
    """
    m = _mod()
    committed = ROOT / "data/evidence/pseudo_labels/fold0_heldout.json"
    base_committed = ROOT / "data/evidence/block_holdout/fold0_heldout_combined.json"
    if not committed.exists() or not base_committed.exists():
        pytest.skip("committed fold-0 arms not present in this checkout")
    r = m.pooled_contrast([0], m.load_json(m.DECISION), m.pseudo_provenance(),
                          block_mode="all", n_boot=2000, seed=0)
    comb = r["populations"]["combined"]
    assert comb["status"] == "MEASURED"
    # the committed two-population contrast bootstraps the same rows on the union
    ref = m.load_json(ROOT / "data/evidence/pseudo_labels/fold0_two_population_contrast.json")
    combined_arm = ref["arms"]["heldout"]["populations"]["combined"]
    assert comb["pooled_candidate_dti"] == pytest.approx(combined_arm["candidate_dti"], abs=1e-6)
    assert comb["pooled_reference_dti"] == pytest.approx(combined_arm["reference_dti"], abs=1e-6)
    # 'all' draws all 56 blocks but only the scoreable ones are resampling units
    assert comb["n_blocks_drawn"] == 56 and comb["resampling_units"] == 7
