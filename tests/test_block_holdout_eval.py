"""scripts/block_holdout_eval.py: the block-stratified error bar must be exact and honest.

Three things are pinned here:

1. the vectorised per-block aggregation equals `src.metrics.score_within_mask` on EVERY block
   (not just the two blocks the script cross-checks at runtime);
2. the report is internally consistent - parts sum to the published global score, the reference
   candidate is the one every contrast is paired against, and a candidate contrasted with itself
   is zero everywhere;
3. the failure modes are loud: a grid mismatch, an empty proxy population, or a reference
   candidate outside the swept grid must stop the run instead of producing a table of numbers
   that quietly mean something else.
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


def _mod():
    spec = importlib.util.spec_from_file_location("bhe", ROOT / "scripts" / "block_holdout_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TRANSFORM = from_origin(243350.0, 4508550.0, 100.0, 100.0)
CRS = "EPSG:32611"


def _write(path: Path, arr: np.ndarray, dtype: str) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=CRS, transform=TRANSFORM) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


@pytest.fixture()
def synthetic(tmp_path):
    """A 128x160 grid with a NaN border, two fault traces and an independent 'proxy' trace."""
    rng = np.random.default_rng(11)
    H, W = 128, 160
    footprint = np.ones((H, W), bool)
    footprint[:8, :] = False
    footprint[-8:, :] = False

    labels = np.zeros((H, W), np.uint8)
    labels[40:42, 20:140] = 1
    labels[90:120, 100:102] = 1

    proxy = np.zeros((H, W), np.uint8)
    proxy[60:62, 30:150] = 2          # code 2: absent from the training labels
    proxy[40:42, 20:60] = 1           # code 1: already within R of a label
    proxy[20:22, 10:80] = 2

    pred = np.full((H, W), np.nan, np.float32)
    body = np.zeros((H, W), np.float32)
    body[38:44, 18:142] = 0.55        # near the catalogue trace
    body[58:64, 28:152] = 0.30        # near the proxy-only trace
    body[95:110, 95:105] = 0.12       # a low-confidence blob
    body += rng.normal(0, 0.02, (H, W)).astype(np.float32)
    np.clip(body, 0, 1, out=body)
    pred[footprint] = body[footprint]

    files = dict(
        pred=_write(tmp_path / "pred.tif", pred, "float32"),
        labels=_write(tmp_path / "labels.tif", np.where(footprint, labels, -1).astype(np.int8), "int8"),
        proxy=_write(tmp_path / "proxy.tif", proxy, "uint8"),
        template=_write(tmp_path / "template.tif",
                        np.where(footprint, 0.0, np.nan).astype(np.float32), "float32"),
    )
    return files


def _run(m, files, out: Path, **kw):
    argv = ["--pred", str(files["pred"]), "--labels", str(files["labels"]),
            "--proxy", str(files["proxy"]), "--template", str(files["template"]),
            "--block-px", str(kw.pop("block_px", 32)), "--folds", str(kw.pop("folds", 4)),
            "--floors", kw.pop("floors", "0.1,0.3"), "--widths", kw.pop("widths", "0,2,4"),
            "--reference-floor", str(kw.pop("reference_floor", 0.1)),
            "--reference-width", str(kw.pop("reference_width", 0)),
            "--bootstraps", str(kw.pop("bootstraps", 300)),
            "--crosscheck-sweep", kw.pop("crosscheck_sweep", ""),
            "--out", str(out), "--quiet", *[str(x) for x in kw.pop("extra", [])]]
    rc = m.main(argv)
    assert rc == 0
    return json.loads(out.read_text())


# --------------------------------------------------------------------------------------
# structure + exactness
# --------------------------------------------------------------------------------------
def test_report_structure_and_populations(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json")
    assert rep["generated_by"] == "scripts/block_holdout_eval.py"
    assert set(rep["populations"]) == {"labels", "proxy_only"}
    assert rep["populations"]["labels"]["n_gt"] > 0
    assert rep["populations"]["proxy_only"]["n_gt"] > 0
    # code 1 (already within R of a label) must NOT be counted as new-fault-like truth
    with rasterio.open(synthetic["proxy"]) as src:
        codes = src.read(1)
    assert rep["populations"]["proxy_only"]["n_gt"] == int((codes == 2).sum())
    assert rep["blocks"]["block_px"] == 32 and rep["blocks"]["n_blocks"] == 20
    assert rep["verdict"]["reference_candidate"] == "floor0.1_w0px"
    assert rep["footprint"]["px"] == int(np.count_nonzero(~np.isnan(
        rasterio.open(synthetic["template"]).read(1))))


def test_every_candidate_sums_to_its_own_global_score(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json", floors="0.1,0.2,0.3", widths="0,1,3")
    for pop in rep["populations"].values():
        for row in pop["candidates"]:
            tp = sum(b["TP_w"] for b in row["blocks"])
            fp = sum(b["FP_w"] for b in row["blocks"])
            fn = sum(b["FN_w"] for b in row["blocks"])
            ngt = sum(b["n_gt"] for b in row["blocks"])
            assert tp == pytest.approx(row["TP_w"], rel=1e-9, abs=1e-9)
            assert fp == pytest.approx(row["FP_w"], rel=1e-9, abs=1e-9)
            assert fn == pytest.approx(row["FN_w"], rel=1e-9, abs=1e-9)
            assert ngt == row["n_gt"] == pop["n_gt"]
            recomposed = tp / (tp + pop["alpha"] * fp + pop["beta"] * fn + pop["eps"])
            assert recomposed == pytest.approx(row["global_dti"], abs=1e-6)


def test_vectorised_blocks_equal_the_reference_implementation_on_every_block(synthetic, tmp_path):
    """The script cross-checks two blocks at runtime; this checks all of them."""
    from src.metrics import GtContext, score_within_mask
    m = _mod()
    pred_raw, _ = m.read_band(synthetic["pred"])
    labels_raw, _ = m.read_band(synthetic["labels"])
    template_raw, _ = m.read_band(synthetic["template"])
    footprint = m.finite_mask(pred_raw, template_raw)
    pred = np.clip(np.nan_to_num(pred_raw.astype(np.float64)), 0, 1)
    truth = (np.nan_to_num(labels_raw.astype(np.float64)) > 0.5) & footprint
    blocks = m.block_id_map(truth.shape, 32)
    ctx = GtContext(truth.astype(np.float64), R_pixels=3)
    fpw = ctx.fp_weight()
    emis = ((pred >= 0.1) & footprint).astype(np.float64)
    comp = m.block_components(ctx, emis, blocks, 20, fpw)
    for b in range(20):
        ref = score_within_mask(emis, ctx, blocks == b, credit=comp["credit"])
        assert comp["TP"][b] == pytest.approx(ref["TP_w"], rel=1e-9, abs=1e-9), f"TP block {b}"
        assert comp["FP"][b] == pytest.approx(ref["FP_w"], rel=1e-9, abs=1e-9), f"FP block {b}"
        assert comp["FN"][b] == pytest.approx(ref["FN_w"], rel=1e-9, abs=1e-9), f"FN block {b}"
        assert int(comp["n_gt"][b]) == ref["n_gt"]


def test_reference_candidate_is_first_and_contrasted_with_itself(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json")
    for pop_name, pop in rep["populations"].items():
        assert pop["candidates"][0]["label"] == "floor0.1_w0px"
        boot = rep["bootstrap"][pop_name]
        assert boot["floor0.1_w0px"]["prob_beats_reference"] is None
        assert boot["floor0.1_w0px"]["contrast_vs_reference_p50"] == 0.0
        for label, b in boot.items():
            lo, hi = b["dti_ci95"]
            point = next(c["global_dti"] for c in pop["candidates"] if c["label"] == label)
            assert lo <= hi
            # the block bootstrap must contain the point estimate unless the score is dominated
            # by a single block (then say so rather than silently widening)
            if b["n_scoreable_blocks"] > 2:
                assert lo - 1e-6 <= point <= hi + 1e-6, f"{label}: {point} outside [{lo}, {hi}]"


def test_bootstrap_is_reproducible_and_seed_sensitive(synthetic, tmp_path):
    m = _mod()
    a = _run(m, synthetic, tmp_path / "a.json", bootstraps=200)
    b = _run(m, synthetic, tmp_path / "b.json", bootstraps=200)
    assert a["bootstrap"]["proxy_only"] == b["bootstrap"]["proxy_only"]
    m.main(["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
            "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
            "--block-px", "32", "--floors", "0.1,0.3", "--widths", "0,2,4",
            "--bootstraps", "200", "--seed", "7", "--crosscheck-sweep", "",
            "--out", str(tmp_path / "d.json"), "--quiet"])
    d = json.loads((tmp_path / "d.json").read_text())
    assert d["bootstrap"]["proxy_only"] != a["bootstrap"]["proxy_only"], \
        "a different bootstrap seed must give a different resample"


def test_widening_changes_the_emission_and_the_score(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json", floors="0.1", widths="0,4,8")
    rows = {c["label"]: c for c in rep["populations"]["proxy_only"]["candidates"]}
    assert rows["floor0.1_w0px"]["support_px"] < rows["floor0.1_w4px"]["support_px"] \
        < rows["floor0.1_w8px"]["support_px"]
    assert rows["floor0.1_w0px"]["FP_w"] < rows["floor0.1_w8px"]["FP_w"]


def test_sign_agreement_table_covers_every_candidate(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json", floors="0.1,0.3", widths="0,2")
    labels = [c["label"] for c in rep["population_agreement"]["per_candidate"]]
    assert labels == [c["label"] for c in rep["populations"]["labels"]["candidates"]]
    assert rep["population_agreement"]["reference"] == "floor0.1_w0px"
    for row in rep["population_agreement"]["per_candidate"]:
        assert row["blocks_agree_in_sign"] <= row["blocks_both_populations"]
        if row["agreement_fraction"] is not None:
            assert 0.0 <= row["agreement_fraction"] <= 1.0


# --------------------------------------------------------------------------------------
# loud failures
# --------------------------------------------------------------------------------------
def test_grid_mismatch_stops_the_run(synthetic, tmp_path):
    m = _mod()
    small = _write(tmp_path / "small_labels.tif", np.zeros((64, 80), np.uint8), "uint8")
    with pytest.raises(SystemExit) as exc:
        m.main(["--pred", str(synthetic["pred"]), "--labels", str(small),
                "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
                "--block-px", "32", "--out", str(tmp_path / "never.json"), "--quiet"])
    assert "not on the prediction grid" in str(exc.value)
    assert not (tmp_path / "never.json").exists(), "a failed run must not leave a report behind"


def test_empty_proxy_population_stops_the_run(synthetic, tmp_path):
    m = _mod()
    empty = _write(tmp_path / "empty_proxy.tif", np.zeros((128, 160), np.uint8), "uint8")
    with pytest.raises(SystemExit) as exc:
        m.main(["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
                "--proxy", str(empty), "--template", str(synthetic["template"]),
                "--block-px", "32", "--out", str(tmp_path / "never.json"), "--quiet"])
    assert "new-fault-like population is empty" in str(exc.value)


def test_reference_outside_the_swept_grid_is_refused(synthetic, tmp_path):
    m = _mod()
    with pytest.raises(SystemExit):
        m.main(["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
                "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
                "--block-px", "32", "--floors", "0.1,0.2", "--widths", "0,2",
                "--reference-floor", "0.7", "--reference-width", "0",
                "--out", str(tmp_path / "never.json"), "--quiet"])


def test_footprint_can_come_from_the_prediction_itself(synthetic, tmp_path):
    """--template '' must work, because an entrant may only have their own raster."""
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json")
    argv = ["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
            "--proxy", str(synthetic["proxy"]), "--template", "", "--block-px", "32",
            "--bootstraps", "100", "--crosscheck-sweep", "",
            "--out", str(tmp_path / "nojson.json"), "--quiet"]
    assert m.main(argv) == 0
    rep2 = json.loads((tmp_path / "nojson.json").read_text())
    assert rep2["inputs"]["template"] is None
    assert rep2["footprint"]["px"] == rep["footprint"]["px"], \
        "the shipped raster is NaN exactly where the template is, so both must give one footprint"


# --------------------------------------------------------------------------------------
# the reproduction gate: sandbox bytes vs committed runner evidence
# --------------------------------------------------------------------------------------
def test_crosscheck_refuses_a_sweep_that_describes_different_bytes(synthetic, tmp_path):
    """A synthetic raster must NOT be allowed to "reproduce" the real runner sweep."""
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json")
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"results": {"shaping_sweep": [
        {"t0": 0.1, "thin": True, "dilate": 0, "soft": False, "dti": 0.099859,
         "TP_w": 8378.12, "FP_w": 164461.741, "FN_w": 53285.88, "emission_px": 172974}]}}))
    rc = m.crosscheck_against_committed_sweep(rep, sweep, dict(floor=0.1, width=0))
    assert rc["status"] == "MISMATCH", "a different raster must never be reported as a reproduction"


def test_crosscheck_reports_missing_evidence_as_unavailable(synthetic, tmp_path):
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json")
    rc = m.crosscheck_against_committed_sweep(rep, tmp_path / "does_not_exist.json",
                                              dict(floor=0.1, width=0))
    assert rc["status"] == "not_available" and "does not exist" in rc["reason"]
    empty = tmp_path / "no_row.json"
    empty.write_text(json.dumps({"results": {"shaping_sweep": [
        {"t0": 0.9, "thin": True, "dilate": 5, "soft": False, "dti": 0.1}]}}))
    rc2 = m.crosscheck_against_committed_sweep(rep, empty, dict(floor=0.1, width=0))
    assert rc2["status"] == "not_available" and "no row" in rc2["reason"]


def test_committed_sandbox_report_reproduces_the_committed_runner_sweep():
    """The claim the docs make, checked from the two committed JSONs (no rasters, no network).

    data/evidence/block_holdout/block_stratified.json is scored in the development sandbox from
    the shipped submission bytes; data/evidence/proxy/eval_sweep-mean12.json is the row the
    GitHub-hosted runner measured for the same policy on the pre-shaping mean field.  Equality on
    DTI and on all three metric components is what makes "the sandbox reproduces runner evidence"
    a fact rather than a sentence.
    """
    rep_path = ROOT / "data/evidence/block_holdout/block_stratified.json"
    sweep_path = ROOT / "data/evidence/proxy/eval_sweep-mean12.json"
    if not (rep_path.exists() and sweep_path.exists()):
        pytest.skip("committed evidence not present")
    m = _mod()
    rep = json.loads(rep_path.read_text())
    rc = m.crosscheck_against_committed_sweep(rep, sweep_path, dict(floor=0.1, width=0))
    assert rc["status"] == "reproduced", json.dumps(rc, indent=1)[:2000]
    assert rep.get("reproduction", {}).get("status") == "reproduced", \
        "the committed report must carry its own reproduction record"


def test_duplicate_emissions_are_reported_not_counted():
    """A hard-band submission makes the floor axis degenerate; the report must say so."""
    rep_path = ROOT / "data/evidence/block_holdout/block_stratified.json"
    if not rep_path.exists():
        pytest.skip("committed evidence not present")
    rep = json.loads(rep_path.read_text())
    v = rep["verdict"]
    assert v["n_candidates_swept"] >= v["n_distinct_emissions"]
    if rep["prediction"]["hard_band"]:
        assert v["floor_axis_degenerate"] is True
        assert v["n_distinct_emissions"] < v["n_candidates_swept"], \
            "a two-valued raster cannot produce more distinct emissions than swept widths"
        dupes = [c for c in rep["populations"]["proxy_only"]["candidates"] if c["duplicate_of"]]
        assert dupes, "duplicates were detected but not labelled"
        for c in dupes:
            twin = next(x for x in rep["populations"]["proxy_only"]["candidates"]
                        if x["label"] == c["duplicate_of"])
            assert twin["global_dti"] == c["global_dti"], "labelled duplicates must score identically"


def test_interpretation_warns_when_blocks_disagree_with_the_global_ranking():
    rep_path = ROOT / "data/evidence/block_holdout/block_stratified.json"
    if not rep_path.exists():
        pytest.skip("committed evidence not present")
    rep = json.loads(rep_path.read_text())
    interp = rep["interpretation"]
    assert "not decomposable" in interp["interpretation"]
    if interp["n_conflicts"]:
        for g in interp["per_block_majority_conflicts_with_global"]:
            assert g["global_dti"] < g["reference_global_dti"]
            assert g["blocks_gaining"] * 2 > g["blocks_scoreable"]


def test_conflict_counters_require_both_signs(synthetic, tmp_path):
    """A block only counts as a population conflict when BOTH populations are scoreable and the
    two contrasts have opposite signs.  (An earlier draft counted the labels side alone and so
    overstated the conflict - see the note in per_block_sign_agreement's docstring.)"""
    m = _mod()
    rep = _run(m, synthetic, tmp_path / "out.json", bootstraps=50)
    for row in rep["population_agreement"]["per_candidate"]:
        assert row["blocks_conflict_labels_lose_proxy_gains"] + \
               row["blocks_conflict_proxy_lose_labels_gain"] <= row["blocks_both_populations"]
        assert row["blocks_conflict_labels_lose_proxy_gains"] <= row["blocks_labels_lose"]
        agree = row["blocks_agree_in_sign"]
        conflicts = (row["blocks_conflict_labels_lose_proxy_gains"]
                     + row["blocks_conflict_proxy_lose_labels_gain"])
        assert agree + conflicts == row["blocks_both_populations"], \
            "sign agreement plus both conflict kinds must exhaust the blocks both populations score"
        assert row["agreement_fraction"] == pytest.approx(
            agree / row["blocks_both_populations"], rel=1e-4)


def test_committed_evidence_carries_accurate_conflict_counters():
    rep_path = ROOT / "data/evidence/block_holdout/block_stratified.json"
    if not rep_path.exists():
        pytest.skip("committed evidence not present")
    rep = json.loads(rep_path.read_text())
    rows = rep["population_agreement"]["per_candidate"]
    assert rows, "the agreement table must not be empty"
    for row in rows:
        assert "blocks_conflict_labels_lose_proxy_gains" in row
        assert row["blocks_conflict_labels_lose_proxy_gains"] <= row["blocks_labels_lose"]
    worst = max(rows, key=lambda r: r["blocks_labels_lose"])
    assert worst["blocks_labels_lose"] >= worst["blocks_conflict_labels_lose_proxy_gains"], \
        "the conflict subset cannot exceed the labels-lose count"


# --------------------------------------------------------------------------------------
# --config: the training partition and the scoring partition must be the same partition
# --------------------------------------------------------------------------------------
def _cfg(tmp_path, body, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_config_supplies_defaults_and_cli_still_wins(synthetic, tmp_path):
    m = _mod()
    cfg = _cfg(tmp_path, """
training: {holdout: spatial_blocks, block_px: 32, block_folds: 4}
block_holdout_eval:
  block_px: 32
  folds: 4
  floors: "0.1,0.4"
  widths: "0,3"
  reference_floor: 0.4
  reference_width: 3
  bootstraps: 60
  crosscheck_sweep: ""
""")
    out = tmp_path / "cfg_out.json"
    argv = ["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
            "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
            "--config", str(cfg), "--out", str(out), "--quiet"]
    assert m.main(argv) == 0
    rep = json.loads(out.read_text())
    assert rep["blocks"]["block_px"] == 32
    labels = [c["label"] for c in rep["populations"]["proxy_only"]["candidates"]]
    # row 0 is the reference (bootstrap_from_blocks pairs every contrast against row 0), then the
    # swept grid in (floor, width) order minus the reference - see the comment in build_report.
    assert labels[0] == "floor0.4_w3px", labels
    assert sorted(labels) == sorted(["floor0.1_w0px", "floor0.1_w3px", "floor0.4_w0px",
                                     "floor0.4_w3px"]), labels
    assert rep["verdict"]["reference_candidate"] == "floor0.4_w3px", "the config set the reference"
    # CLI override beats the config
    out2 = tmp_path / "cli_out.json"
    assert m.main(argv + ["--reference-floor", "0.1", "--reference-width", "0",
                          "--out", str(out2), "--quiet"]) == 0
    assert json.loads(out2.read_text())["verdict"]["reference_candidate"] == "floor0.1_w0px"


def test_config_partition_disagreement_is_refused(synthetic, tmp_path):
    """block_holdout_eval.block_px != training.block_px => not a holdout measurement."""
    m = _mod()
    cfg = _cfg(tmp_path, """
training: {holdout: spatial_blocks, block_px: 512, block_folds: 4}
block_holdout_eval: {block_px: 32, folds: 4}
""")
    with pytest.raises(SystemExit) as exc:
        m.main(["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
                "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
                "--config", str(cfg), "--out", str(tmp_path / "never.json"), "--quiet"])
    msg = str(exc.value)
    assert "block_holdout_eval.block_px=32" in msg and "training.block_px=512" in msg
    assert not (tmp_path / "never.json").exists()


def test_config_fold_disagreement_is_refused(synthetic, tmp_path):
    m = _mod()
    cfg = _cfg(tmp_path, """
training: {holdout: spatial_blocks, block_px: 32, block_folds: 6}
block_holdout_eval: {block_px: 32, folds: 4}
""")
    with pytest.raises(SystemExit) as exc:
        m.main(["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
                "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
                "--config", str(cfg), "--out", str(tmp_path / "never.json"), "--quiet"])
    assert "folds=4" in str(exc.value) and "block_folds=6" in str(exc.value)


def test_config_notes_when_the_model_was_not_spatially_held_out(synthetic, tmp_path):
    m = _mod()
    cfg = _cfg(tmp_path, """
training: {holdout: random, block_px: 32, block_folds: 4}
block_holdout_eval: {block_px: 32, folds: 4, bootstraps: 40, crosscheck_sweep: ""}
""")
    out = tmp_path / "note.json"
    assert m.main(["--pred", str(synthetic["pred"]), "--labels", str(synthetic["labels"]),
                   "--proxy", str(synthetic["proxy"]), "--template", str(synthetic["template"]),
                   "--config", str(cfg), "--out", str(out)]) == 0
    _, bh, conflict, note = m.load_config(cfg)
    assert conflict is None and note and "RESHAPING" in note
    assert bh["block_px"] == 32


def test_config_missing_file_is_refused(tmp_path):
    m = _mod()
    with pytest.raises(SystemExit) as exc:
        m.load_config(tmp_path / "absent.yaml")
    assert "does not exist" in str(exc.value)


def test_committed_block_holdout_config_is_self_consistent():
    """configs/config_block_holdout.yaml must not trip its own cross-check."""
    cfg_path = ROOT / "configs" / "config_block_holdout.yaml"
    if not cfg_path.exists():
        pytest.skip("config not present")
    m = _mod()
    cfg, bh, conflict, note = m.load_config(cfg_path)
    assert conflict is None, conflict
    assert note is None, f"the block-holdout config must actually hold out blocks: {note}"
    assert cfg["training"]["holdout"] == "spatial_blocks"
    assert bh["block_px"] == cfg["training"]["block_px"]
    assert bh["folds"] == cfg["training"]["block_folds"]
    assert float(bh["reference_floor"]) == 0.1 and int(bh["reference_width"]) == 0, \
        "the reference must stay the adopted policy (floor 0.1, width 0)"
    assert bh["alpha"] == 0.2 and bh["beta"] == 0.8 and bh["R"] == 3


# --------------------------------------------------------------------------------------
# --score-fold: restricting the measurement to one fold's blocks (the generalisation reading)
# --------------------------------------------------------------------------------------
@pytest.fixture()
def dotted(tmp_path):
    """A grid with catalogue AND proxy-only truth in every block, so any fold restriction scores.

    The `synthetic` fixture concentrates its traces in two row bands, which would make a
    single-fold restriction empty for some folds - a property of that fixture, not of the code.
    """
    H, W = 128, 160
    footprint = np.ones((H, W), bool)
    footprint[:8, :] = False
    footprint[-8:, :] = False
    labels = np.zeros((H, W), np.uint8)
    proxy = np.zeros((H, W), np.uint8)
    pred = np.full((H, W), np.nan, np.float32)
    body = np.zeros((H, W), np.float32)
    for r in range(12, H - 8, 16):                 # every 32-px block row gets traces
        for c in range(8, W - 8, 20):              # and every block column
            labels[r, c:c + 12] = 1                # catalogue fault
            proxy[r + 6, c:c + 12] = 2             # independent fault, no label within R
            proxy[r + 12, c:c + 6] = 1             # independent fault the labels already have
            body[r - 1:r + 2, c - 1:c + 13] = 0.5
            body[r + 5:r + 8, c - 1:c + 13] = 0.25
    pred[footprint] = body[footprint]
    return dict(
        pred=_write(tmp_path / "pred.tif", pred, "float32"),
        labels=_write(tmp_path / "labels.tif", np.where(footprint, labels, -1).astype(np.int8), "int8"),
        proxy=_write(tmp_path / "proxy.tif", proxy, "uint8"),
        template=_write(tmp_path / "template.tif",
                        np.where(footprint, 0.0, np.nan).astype(np.float32), "float32"),
    )


def test_score_fold_restricts_the_measurement_to_that_folds_blocks(dotted, tmp_path):
    m = _mod()
    full = _run(m, dotted, tmp_path / "full.json", bootstraps=50)
    # the per-block rows cover EVERY block of the partition; only `scoreable` ones were measured
    all_blocks = {b["block"] for b in full["populations"]["proxy_only"]["candidates"][0]["blocks"]
                  if b["scoreable"]}
    assert full["restriction"]["mode"] == "all_blocks" and full["restriction"]["score_fold"] is None
    assert len(all_blocks) == full["restriction"]["blocks_scoreable"]

    seen = set()
    for fold in range(4):
        rep = _run(m, dotted, tmp_path / f"fold{fold}.json", bootstraps=50,
                   extra=["--score-fold", str(fold)])
        r = rep["restriction"]
        assert r["mode"] == "fold_heldout" and r["score_fold"] == fold and r["complement"] is False
        ids = {b["block"] for b in rep["populations"]["proxy_only"]["candidates"][0]["blocks"]
               if b["scoreable"]}
        assert ids and ids <= set(r["blocks"]), f"fold {fold}: scored blocks {ids} not in {r['blocks']}"
        assert len(ids) == r["blocks_scoreable"]
        assert ids <= all_blocks
        seen |= ids
        assert r["px_included"] + r["px_excluded"] == full["footprint"]["px"]
        assert 0 < r["px_included"] < full["footprint"]["px"]
    assert seen == all_blocks, "the four folds' blocks must cover every scoreable block exactly"


def test_complement_and_heldout_are_disjoint_and_exhaustive(dotted, tmp_path):
    m = _mod()
    held = _run(m, dotted, tmp_path / "held.json", bootstraps=50, extra=["--score-fold", "2"])
    comp = _run(m, dotted, tmp_path / "comp.json", bootstraps=50,
                extra=["--score-fold", "2", "--complement"])
    assert held["restriction"]["mode"] == "fold_heldout"
    assert comp["restriction"]["mode"] == "fold_complement" and comp["restriction"]["complement"]
    hb, cb = set(held["restriction"]["blocks"]), set(comp["restriction"]["blocks"])
    assert hb.isdisjoint(cb), "a block cannot be both held out and trained on"
    full = _run(m, dotted, tmp_path / "full.json", bootstraps=50)
    all_ids = {b["block"] for b in full["populations"]["proxy_only"]["candidates"][0]["blocks"]
               if b["scoreable"]}
    assert hb | cb == all_ids
    assert held["restriction"]["px_included"] + comp["restriction"]["px_included"] == \
        full["footprint"]["px"]
    # the two readings are different measurements, not the same number twice
    assert held["verdict"]["reference_proxy_dti"] != comp["verdict"]["reference_proxy_dti"]


def test_score_fold_out_of_range_is_refused(dotted, tmp_path):
    m = _mod()
    with pytest.raises(SystemExit) as exc:
        _run(m, dotted, tmp_path / "never.json", extra=["--score-fold", "9"])
    assert "outside [0,4)" in str(exc.value)
    assert not (tmp_path / "never.json").exists()


def test_restriction_scope_is_always_recorded(dotted, tmp_path):
    """A block-restricted number must never be quotable without the scope it was measured on."""
    m = _mod()
    rep = _run(m, dotted, tmp_path / "r.json", bootstraps=50, extra=["--score-fold", "0"])
    note = rep["restriction"]["note"]
    assert "spatial_blocks" in note and "block_fold=0" in note
    assert "generalisation reading ONLY if" in note
    full = _run(m, dotted, tmp_path / "f.json", bootstraps=50)
    assert "RESHAPING" in full["restriction"]["note"]


# --------------------------------------------------------------------------------------
# how much a block bootstrap from THIS many blocks is worth
# --------------------------------------------------------------------------------------
def test_few_resampling_units_are_flagged_as_coarse(dotted, tmp_path):
    """Measured on the real grid: 8 blocks flipped the sign of the width decision (P=0.66 for a
    widening the 34-block measurement rejects at P=0.008).  The report must say so."""
    m = _mod()
    rep = _run(m, dotted, tmp_path / "one_fold.json", bootstraps=80, extra=["--score-fold", "0"])
    rel = rep["bootstrap"]["reliability"]
    assert rel["resampling_units"] == rep["blocks"]["n_blocks_scoreable"]
    assert rel["scope"] == "fold_heldout"
    if rel["resampling_units"] < rel["min_units_for_a_readable_ci"]:
        assert rel["interval_readable"] is False
        assert "sampling noise" in rel["note"] and "NOT evidence" in rel["note"]
        assert rel["warning"]
    else:
        assert rel["interval_readable"] is True and "warning" not in rel


def test_the_full_grid_run_reports_a_readable_interval():
    rep_path = ROOT / "data/evidence/block_holdout/block_stratified.json"
    if not rep_path.exists():
        pytest.skip("committed evidence not present")
    rep = json.loads(rep_path.read_text())
    rel = rep["bootstrap"]["reliability"]
    assert rel["resampling_units"] >= rel["min_units_for_a_readable_ci"]
    assert rel["interval_readable"] is True and rel["scope"] == "all_blocks"
    assert "not uncertainty over truth definitions" in rel["note"]


def test_reliability_threshold_is_a_property_of_the_function():
    m = _mod()
    lo = m.bootstrap_reliability(8, 56, 2000, dict(mode="fold_heldout"))
    hi = m.bootstrap_reliability(34, 56, 2000, dict(mode="all_blocks"))
    assert lo["interval_readable"] is False and hi["interval_readable"] is True
    assert lo["resampling_units"] == 8 and hi["resampling_units"] == 34
    assert "warning" in lo and "warning" not in hi
    edge = m.bootstrap_reliability(m.MIN_BLOCKS_FOR_A_READABLE_CI, 56, 2000, {})
    assert edge["interval_readable"] is True, "the threshold itself is readable (>=, not >)"
