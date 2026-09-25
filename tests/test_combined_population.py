"""scripts/block_holdout_eval.py --combined-population: the Phase-2-like surrogate must be exact.

WHY THIS POPULATION EXISTS.  Phase 2 re-scores the SAME submission against "the full, revised new
fault dataset" (rules 1.1/3.6, verbatim in data/evidence/rules_quotes.json): the existing mapped
faults plus the faults the expert panel adds.  Neither committed population is that set - `labels`
is only the catalogue and `proxy_only` is only the new-fault-like part - so the SGMC pseudo-label
result (proxy DTI up, catalogue DTI down) could not be read as a gain or a loss from either arm.
Their union is the closest surrogate this repository can build from bytes it already has.

WHAT IS PINNED HERE.
1. the union really is the union: n_gt == |labels| + |proxy_only| (the two sets are disjoint by
   construction, code 2 excludes pixels within R of a label) and the global DTI equals an
   INDEPENDENT computation with src.metrics on the union array;
2. the mathematical property that makes the population non-decomposable is respected: FP_w on the
   union is <= FP_w on each component (more truth can only reduce the false-positive penalty),
   while TP_w and FN_w add exactly;
3. the flag is opt-in: without it the report schema is byte-for-byte the one every other test and
   every committed evidence file uses (no `combined` population, no `combined_population` block);
4. the restriction path (--score-fold) restricts the union too, so a fold-restricted combined
   number cannot silently include blocks the fold trained on;
5. a config can enable it (block_holdout_eval.combined_population), and the report says what the
   population is a surrogate for and what it is not.
"""
from __future__ import annotations

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

from src.metrics import compute_distance_weighted_tversky  # noqa: E402

TRANSFORM = from_origin(243350.0, 4508550.0, 100.0, 100.0)
CRS = "EPSG:32611"


def _mod():
    spec = importlib.util.spec_from_file_location("bhe_comb", ROOT / "scripts" / "block_holdout_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(path: Path, arr: np.ndarray, dtype: str) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=CRS, transform=TRANSFORM) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


@pytest.fixture()
def synthetic(tmp_path):
    """A 96x128 grid: a catalogue trace, an independent proxy-only trace, and a prediction near both."""
    H, W = 96, 128
    footprint = np.ones((H, W), bool)
    footprint[:6, :] = False
    footprint[-6:, :] = False

    labels = np.zeros((H, W), np.uint8)
    labels[30:32, 15:110] = 1                 # catalogue trace
    proxy = np.zeros((H, W), np.uint8)
    proxy[60:62, 20:120] = 2                  # code 2: new-fault-like, absent from the labels
    proxy[30:32, 15:50] = 1                   # code 1: already within R of a label -> NOT truth

    body = np.zeros((H, W), np.float32)
    body[29:33, 14:111] = 0.6                 # over the catalogue trace
    body[59:63, 19:121] = 0.25                # over the proxy-only trace
    body[70:74, 5:15] = 0.4                   # a false alarm, far from both
    pred = np.full((H, W), np.nan, np.float32)
    pred[footprint] = body[footprint]

    return dict(
        pred=_write(tmp_path / "pred.tif", pred, "float32"),
        labels=_write(tmp_path / "labels.tif", np.where(footprint, labels, -1).astype(np.int8), "int8"),
        proxy=_write(tmp_path / "proxy.tif", proxy, "uint8"),
        template=_write(tmp_path / "template.tif",
                        np.where(footprint, 0.0, np.nan).astype(np.float32), "float32"),
        arrays=dict(footprint=footprint, labels=labels, proxy=proxy, pred=pred),
    )


def _run(m, files, out: Path, **kw):
    argv = ["--pred", str(files["pred"]), "--labels", str(files["labels"]),
            "--proxy", str(files["proxy"]), "--template", str(files["template"]),
            "--block-px", str(kw.pop("block_px", 32)), "--folds", str(kw.pop("folds", 4)),
            "--floors", kw.pop("floors", "0.1,0.3"), "--widths", kw.pop("widths", "0,2"),
            "--reference-floor", str(kw.pop("reference_floor", 0.1)),
            "--reference-width", str(kw.pop("reference_width", 0)),
            "--bootstraps", str(kw.pop("bootstraps", 200)),
            "--crosscheck-sweep", kw.pop("crosscheck_sweep", ""),
            "--out", str(out), "--quiet", *[str(x) for x in kw.pop("extra", [])]]
    assert m.main(argv) == 0
    return json.loads(out.read_text())


# --------------------------------------------------------------------------- 3. opt-in
def test_the_flag_is_opt_in_and_the_default_schema_is_untouched(synthetic, tmp_path):
    """Every committed report and every other test depends on the two-population schema."""
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "default.json")
    assert set(rep["populations"]) == {"labels", "proxy_only"}
    assert "combined_population" not in rep
    assert "combined" not in (rep.get("bootstrap") or {})
    assert rep["verdict"]["reference_combined_dti"] is None, \
        "the verdict must carry an explicit null, not a silently absent key"


def test_the_combined_population_appears_only_when_asked_for(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "combined.json", extra=["--combined-population"])
    assert set(rep["populations"]) == {"labels", "proxy_only", "combined"}
    assert "combined" in rep["bootstrap"]
    assert "combined_population" in rep


# --------------------------------------------------------------------------- 1. exactness
def test_the_union_is_the_union_and_its_dti_is_independently_reproducible(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "combined.json", floors="0.1,0.3", widths="0,2",
               extra=["--combined-population"])
    a = synthetic["arrays"]
    lab = (a["labels"] > 0.5) & a["footprint"]
    pro = (a["proxy"] == 2) & a["footprint"]
    assert not (lab & pro).any(), "the fixture must keep the two truth sets disjoint"

    pops = rep["populations"]
    assert pops["combined"]["n_gt"] == int(lab.sum()) + int(pro.sum())
    assert pops["combined"]["n_gt"] == pops["labels"]["n_gt"] + pops["proxy_only"]["n_gt"]
    assert rep["combined_population"]["provenance"]["disjoint"] is True

    # independent recomputation of the reference candidate with src.metrics on the union array
    emis = ((a["pred"] >= 0.1) & a["footprint"]).astype(np.float64)
    want = compute_distance_weighted_tversky(emis, (lab | pro).astype(np.float64), R_pixels=3,
                                             alpha=0.2, beta=0.8, eps=1e-7)
    row = next(c for c in pops["combined"]["candidates"] if c["label"] == "floor0.1_w0px")
    assert row["global_dti"] == pytest.approx(want, abs=1e-6)
    assert rep["combined_population"]["reference"]["dti"] == pytest.approx(want, abs=1e-6)
    assert rep["verdict"]["reference_combined_dti"] == pytest.approx(want, abs=1e-6)


def test_components_add_for_truth_and_shrink_for_predictions(synthetic, tmp_path):
    """TP_w and FN_w are sums over truth pixels, so they add across disjoint truths; FP_w is a sum
    over PREDICTION pixels against the nearest truth, so a bigger truth can only reduce it.  This is
    exactly why the combined population has to be scored and cannot be inferred from the other two."""
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "combined.json", floors="0.1,0.3", widths="0,2",
               extra=["--combined-population"])
    pops = rep["populations"]
    by = {name: {c["label"]: c for c in pops[name]["candidates"]} for name in pops}
    for label in by["combined"]:
        c, l, p = by["combined"][label], by["labels"][label], by["proxy_only"][label]
        assert c["TP_w"] == pytest.approx(l["TP_w"] + p["TP_w"], rel=1e-9, abs=1e-9)
        assert c["FN_w"] == pytest.approx(l["FN_w"] + p["FN_w"], rel=1e-9, abs=1e-9)
        assert c["FP_w"] <= min(l["FP_w"], p["FP_w"]) + 1e-9
        # and the per-block rows must sum to the global score on the combined population too
        for key, glob in (("TP_w", c["TP_w"]), ("FP_w", c["FP_w"]), ("FN_w", c["FN_w"])):
            if c["blocks"]:
                assert sum(b[key] for b in c["blocks"]) == pytest.approx(glob, rel=1e-9, abs=1e-8)
        if c["blocks"]:
            assert sum(b["n_gt"] for b in c["blocks"]) == c["n_gt"] == pops["combined"]["n_gt"]


def test_combined_truth_structural_identity_tp_plus_fn_equals_gt(synthetic, tmp_path):
    """TP_w + FN_w == |G| holds for any prediction (src/metrics.py); it must hold on the union too."""
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "combined.json", extra=["--combined-population"])
    pop = rep["populations"]["combined"]
    for c in pop["candidates"]:
        assert c["TP_w"] + c["FN_w"] == pytest.approx(pop["n_gt"], rel=1e-9, abs=1e-8)


# --------------------------------------------------------------------------- 4. restriction
def test_a_fold_restriction_restricts_the_union_too(synthetic, tmp_path):
    m = _mod()
    whole = _run(m, synthetic, tmp_path / "whole.json", block_px=32, folds=2,
                 extra=["--combined-population"])
    held = _run(m, synthetic, tmp_path / "held.json", block_px=32, folds=2,
                extra=["--combined-population", "--score-fold", "0"])
    n_whole = whole["populations"]["combined"]["n_gt"]
    n_held = held["populations"]["combined"]["n_gt"]
    assert 0 < n_held < n_whole, "the restricted union must be a strict, non-empty subset"
    assert held["restriction"]["mode"] == "fold_heldout"
    # the union is still the union of the restricted components
    assert n_held == (held["populations"]["labels"]["n_gt"]
                      + held["populations"]["proxy_only"]["n_gt"])
    assert held["combined_population"]["provenance"]["px"]["combined"] == n_held


# --------------------------------------------------------------------------- 5. config + honesty
def test_a_config_can_enable_it_and_the_cli_still_wins(synthetic, tmp_path):
    m = _mod()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "training:\n  holdout: spatial_blocks\n  block_px: 32\n  block_folds: 4\n"
        "block_holdout_eval:\n  block_px: 32\n  folds: 4\n  combined_population: true\n"
        "  floors: '0.1,0.3'\n  widths: '0,2'\n  reference_floor: 0.1\n  reference_width: 0\n"
        "  bootstraps: 200\n  crosscheck_sweep: ''\n")
    out = tmp_path / "cfg.json"
    argv = ["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
            "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
            "--config", str(cfg), "--out", str(out), "--quiet"]
    assert m.main(argv) == 0
    rep = json.loads(out.read_text())
    assert "combined" in rep["populations"], "the config key did not enable the population"


def test_the_report_says_what_the_surrogate_is_and_is_not(synthetic, tmp_path):
    """The page that quotes this number must be able to quote its limitation from the same file."""
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "combined.json", extra=["--combined-population"])
    prov = rep["combined_population"]["provenance"]
    assert prov["inputs"]["proxy_code"] == 2
    assert prov["inputs"]["labels"].endswith("labels.tif")
    assert prov["inputs"]["proxy"].endswith("proxy.tif")
    assert "UNION" in prov["definition"]
    assert "surrogate" in prov["surrogacy"].lower()
    assert "expert" in prov["surrogacy"].lower()
    assert "3.6" in prov["surrogacy"], "the rule that defines the Phase-2 set must be named"
    assert "cannot be recomposed" in prov["why_not_decomposable"]
    assert "upward biased" in rep["combined_population"]["caveat"]
    assert rep["combined_population"]["component_dti"] == dict(
        labels=rep["verdict"]["reference_labels_dti"],
        proxy_only=rep["verdict"]["reference_proxy_dti"],
        combined=rep["verdict"]["reference_combined_dti"])


def test_pruning_and_the_bootstrap_cover_the_combined_population(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "combined.json", floors="0.1,0.15,0.3", widths="0,2",
               bootstraps=200, extra=["--combined-population"])
    boot = rep["bootstrap"]["combined"]
    for c in rep["populations"]["combined"]["candidates"]:
        assert c["label"] in boot, "every combined candidate needs a bootstrap row"
        assert "dti_ci95" in boot[c["label"]]
    assert boot["floor0.1_w0px"]["prob_beats_reference"] is None, "row 0 is the reference"
    # duplicates are detected on the emission, which is population-independent
    dups = [c for c in rep["populations"]["combined"]["candidates"] if c["duplicate_of"]]
    assert dups, "a hard band makes the floor axis degenerate; the combined rows must say so too"
    assert rep["pruning"]["candidates_pruned"] > 0
