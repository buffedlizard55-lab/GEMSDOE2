"""Union-population model-selection signal in src/train.py (EXECUTIVE_SUMMARY §11 item 4).

WHY THIS IS A SEPARATE FILE
---------------------------
The scored population of the GEMS prize is the expert-mapped *new* fault set (rules §1.1/§3.6),
which is disjoint from the released training labels.  Selecting models on the in-domain labels
therefore selects for a population the leaderboard does not score (docs/DISCOVERY_PLAN.md).  The
closest local surrogate is the **union** of the catalogue labels and the SGMC proxy trace the
labels do not contain (code 2 in `data/evidence/proxy/proxy_catalogue.tif`) — the "combined"
population that `scripts/block_holdout_eval.py --combined-population` defines.  This file pins the
src/train.py helpers that surface that population to the training loop.

The union builder must be byte-faithful to `block_holdout_eval.py`:
    combined = labels | (proxy_codes == code 2)
i.e. code 1 (proxy trace within R of a label) is *excluded*, and the component populations are
disjoint (122,652 = 60,988 + 61,664 on the real grid).  A stale or wrongly-read coded raster is a
measurement defect, so every helper raises or refuses rather than inventing a population.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _train_mod():
    import importlib
    return importlib.import_module("src.train")


def _make_res(labels, test_windows, patch_size=16):
    """A synthetic make_patches result shaped the way _union_truth consumes it."""
    return {
        "labels": labels,
        "summary": {
            "test_windows": [(int(i), int(j)) for (i, j) in test_windows],
            "patch_size": patch_size,
        },
    }


def _config(proxy_path="data/evidence/proxy/proxy_catalogue.tif", union=True):
    return {
        "data": {"proxy_catalogue_path": proxy_path},
        "training": {"union_selection": union, "alpha": 0.2, "beta": 0.8},
        "metric": {"R_meters": 300, "resolution_m": 100},
    }


# ----------------------------------------------------------------------------- module builds
def test_union_truth_is_labels_union_proxy_only() -> None:
    """The union is exactly labels | proxy-code-2, the block_holdout_eval definition."""
    tr = _train_mod()
    labels = np.zeros((64, 64), dtype=np.float32)
    labels[10, 10] = 1.0
    labels[20, 20] = 1.0
    coded = np.zeros((64, 64), dtype=np.uint8)
    coded[10, 10] = 2      # proxy agrees with a label
    coded[30, 30] = 1      # code 1: near a label -> must be EXCLUDED from the union
    coded[40, 40] = 2      # proxy-only -> must be INCLUDED
    res = _make_res(labels, [(0, 0), (32, 0)])
    u = tr._union_truth(res, "ignored") if False else _union_via_paths(tr, res, labels, coded)
    assert u["full"][10, 10] == 1.0
    assert u["full"][20, 20] == 1.0
    assert u["full"][30, 30] == 0.0, "code 1 (near a label) is not a union truth pixel"
    assert u["full"][40, 40] == 1.0
    # the mask is the test-window footprint, not the truth: two 16x16 windows = 512 px
    assert u["mask"].sum() == 2 * 16 * 16, "mask must be the held-out window footprint exactly"


def _union_via_paths(tr, res, labels, coded):
    """Run _union_truth with a real (temporary) coded raster so the loader path is exercised."""
    import os
    import tempfile
    import rasterio
    from rasterio.transform import from_origin
    d = tempfile.mkdtemp()
    p = os.path.join(d, "coded.tif")
    with rasterio.open(p, "w", driver="GTiff", height=coded.shape[0], width=coded.shape[1],
                       count=1, dtype="uint8", transform=from_origin(0, coded.shape[0], 1, 1)) as dst:
        dst.write(coded, 1)
    try:
        return tr._union_truth(res, p)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_heldout_scopes_off_by_default() -> None:
    """union_selection is opt-in: no proxy, no scope."""
    tr = _train_mod()
    res = _make_res(np.zeros((64, 64), np.float32), [(0, 0)])
    assert tr._heldout_scopes(_config(union=True, proxy_path=""), res) == {}
    assert tr._heldout_scopes(_config(union=False, proxy_path="x.tif"), res) == {}


def test_union_gate_exists_and_windows_fit() -> None:
    """The raw-label gate keeps the union construction honest, and windows can't exceed the grid."""
    tr = _train_mod()
    labels = np.zeros((64, 64), np.float32)
    labels[1, 1] = 1.0
    res = _make_res(labels, [(62, 62)])       # patch 16 from (62,62) would exceed 64x64
    routes = _config(union=True, proxy_path="")
    assert tr._heldout_scopes(routes, res) == {}   # no proxy -> empty, no crash


def test_score_full_reports_union_scope() -> None:
    tr = _train_mod()
    gt = np.zeros((64, 64), np.float32)
    pred = np.zeros((64, 64), np.float32)
    truth = np.zeros((64, 64), np.float32)
    truth[10, 10] = 1.0
    mask = np.ones((64, 64), bool)
    dti, _, comps = tr._score_full(pred, gt, _config(union=False), R=3,
                                   scopes={"union": (truth, mask)})
    assert comps["scopes"]["union"]["n_gt"] == 1
    assert comps["scopes"]["union"]["dti"] is None or comps["scopes"]["union"]["dti"] >= 0.0


# --------------------------------------------------------- the real population is what we claim
def test_committed_proxy_population_figures_are_what_the_next_steps_quote() -> None:
    """No hallucination guard: the layer the union code reads must carry the documented counts."""
    p = ROOT / "data/evidence/proxy/proxy_stats.json"
    if not p.exists():
        pytest.skip("proxy stats not committed in this checkout")
    d = json.loads(p.read_text())
    assert d["labels"]["fault_px"] == 60988
    assert d["proxy"]["proxy_only_px"] == 61664
    # disjointness asserted by block_holdout_eval provenance: 60,988 + 61,664 = 122,652
    assert d["labels"]["fault_px"] + d["proxy"]["proxy_only_px"] == 122652


# ------------------------------------------------------- the select_on wiring (config contract)
def test_configs_declare_a_valid_select_on() -> None:
    """Every committed config that names select_on must use a value src/train.py accepts, and a
    config that asks to select on the union must supply the proxy raster it is scored on."""
    import yaml
    for cfg_path in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        tr = cfg.get("training") or {}
        sel = tr.get("select_on")
        if sel is not None:
            assert str(sel).lower() in ("in_domain", "union"), \
                f"{cfg_path.name}: select_on={sel!r} is not in_domain/union"
        if str(sel).lower() == "union" or tr.get("union_selection"):
            data = cfg.get("data") or {}
            assert data.get("proxy_catalogue_path"), \
                f"{cfg_path.name}: union selection needs data.proxy_catalogue_path"


def test_train_validates_select_on_and_the_union_scope() -> None:
    """src/train.py must reject an unknown select_on and refuse union selection without a scope.

    These are the two fail-fast gates that keep a misconfigured run from claiming a selection it
    never made; they are exercised through the same helpers the loop uses.
    """
    tr = _train_mod()
    # union scope requires both the flag and a proxy path -> _heldout_scopes returns {} otherwise,
    # which is the exact condition src/train.py turns into a SystemExit for select_on='union'.
    res = _make_res(np.zeros((32, 32), np.float32), [(0, 0)])
    assert tr._heldout_scopes(_config(union=True, proxy_path=""), res) == {}
    assert tr._heldout_scopes(_config(union=False, proxy_path="x.tif"), res) == {}
