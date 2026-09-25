"""Tests for the proxy-catalogue instrument (fetch -> build -> evaluate).

Why these matter: every quality number in this repository was, until now, measured against
`labels.tif` - the faults the catalogue *already contains* - while both prize phases score faults it
does *not* contain (rules §1.1/§3.3, verified verbatim in `data/evidence/rules_quotes.json`).  The
proxy catalogue is the first local measurement on the right population, so it has to be provably a
population of *absent* faults and not a restatement of the labels.

Everything here runs offline: the GeoJSON is synthetic, built by projecting known UTM coordinates
into EPSG:4326 with rasterio (no network, no competition rasters).

Run: python -m pytest tests/test_proxy_catalogue.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import GtContext  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    """A 96x96 EPSG:32611 grid at 100 m, with a labelled fault at column 10."""
    d = tmp_path_factory.mktemp("proxy")
    size, px = 96, 100.0
    transform = from_origin(500000.0, 4100000.0, px, px)
    labels = np.zeros((size, size), dtype="float32")
    labels[8:88, 10] = 1.0                        # the "catalogue": one vertical trace
    labels[8:88, 11] = 0.4                        # a thick label pixel, below the 0.5 threshold
    lab_path = d / "labels.tif"
    with rasterio.open(lab_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:32611", transform=transform) as dst:
        dst.write(labels, 1)
    return dict(dir=d, size=size, transform=transform, labels=labels, labels_path=lab_path)


def _line_geojson(col: float, rows=(8, 88)) -> dict:
    """A vertical GeoJSON line at pixel column `col`, expressed in EPSG:4326 (valid GeoJSON)."""
    transform = from_origin(500000.0, 4100000.0, 100.0, 100.0)
    xs, ys = [], []
    for r in np.linspace(rows[0], rows[1], 9):
        x, y = transform * (col + 0.5, r)
        xs.append(x)
        ys.append(y)
    lon, lat = warp_transform("EPSG:32611", "EPSG:4326", xs, ys)
    return {"type": "Feature", "properties": {"OBJECTID": 1, "RuleID": 43, "STATE": "NV",
                                              "DESCRIPTION": "Normal fault, certain (ball on down side)"},
            "geometry": {"type": "LineString", "coordinates": [[a, b] for a, b in zip(lon, lat)]}}


def _write_geojson(path: Path, features: list[dict]) -> Path:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


# --------------------------------------------------------------------------------------------
# fetch side: the query is derived from the service's own domain, not from a hand-copied list
# --------------------------------------------------------------------------------------------

def test_fault_classes_come_from_the_service_domain():
    fetch = _load("fetch_proxy_faults")
    meta = {"fields": [
        {"name": "RuleID", "domain": {"codedValues": [
            {"code": 3, "name": "Anticline, certain"},
            {"code": 22, "name": "Fault, unknown type, certain"},
            {"code": 37, "name": "Lineament"},
            {"code": 43, "name": "Normal fault, certain (ball on down side)"},
            {"code": 8, "name": "Contact, certain"},
        ]}},
        {"name": "STATE"},                       # no domain: must be skipped, not crashed on
    ]}
    assert set(fetch.fault_rule_ids(meta)) == {"22", "43"}
    with_lines = fetch.fault_rule_ids(meta, ("Lineament",))
    assert set(with_lines) == {"22", "43", "37"} and with_lines["37"] == "Lineament"


def test_query_url_is_reproducible_and_carries_the_filter_and_envelope():
    fetch = _load("fetch_proxy_faults")
    bbox = (-120.0, 37.0, -116.0, 41.0)
    url = fetch.build_query(fetch.SERVICE, 1, bbox, {"22": "x", "43": "y"}, 0, 1000)
    assert url.startswith(fetch.SERVICE + "/1/query?")
    assert "RuleID+IN+%2822%2C43%29" in url or "RuleID%20IN%20%2822%2C43%29" in url
    assert "geometry=-120.000000%2C37.000000%2C-116.000000%2C41.000000" in url
    assert "outSR=4326" in url and "f=json" in url and "resultRecordCount=1000" in url
    assert fetch.build_query(fetch.SERVICE, 1, bbox, {"43": "y", "22": "x"}, 0, 1000) == url


def test_esri_polyline_becomes_valid_geojson_geometry():
    fetch = _load("fetch_proxy_faults")
    geom = {"paths": [[[0.0, 1.0], [1.0, 2.0]], [[1.0, 2.0], [2.0, 3.0]]]}
    out = fetch.esri_paths_to_geojson(geom)
    assert out["type"] == "MultiLineString" and len(out["coordinates"]) == 2
    assert fetch.esri_paths_to_geojson({"paths": [[[0.0, 1.0]]]}) is None       # 1 vertex: no line


def test_raster_bbox_is_the_geographic_envelope_of_the_competition_grid(grid):
    """The query envelope must be the footprint in lon/lat - checked by round-tripping its centre
    back into the raster CRS, not by eyeballing a plausible-looking number."""
    fetch = _load("fetch_proxy_faults")
    box = fetch.raster_bbox_4326(grid["labels_path"])
    assert len(box) == 4 and box[0] < box[2] and box[1] < box[3]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    back_x, back_y = warp_transform("EPSG:4326", "EPSG:32611", [cx], [cy])
    size = grid["size"]
    exp_x = 500000.0 + size * 100.0 / 2
    exp_y = 4100000.0 - size * 100.0 / 2
    assert abs(back_x[0] - exp_x) < 5.0 and abs(back_y[0] - exp_y) < 5.0


# --------------------------------------------------------------------------------------------
# build side: the split into "already in the catalogue" vs "absent" is the whole point
# --------------------------------------------------------------------------------------------

def test_proxy_split_is_the_R_neighbourhood_of_the_labels(grid, monkeypatch):
    build = _load("build_proxy_catalogue")
    feats = [_line_geojson(10), _line_geojson(12), _line_geojson(40)]   # on, near, far
    proxy_path = _write_geojson(grid["dir"] / "proxy.geojson", feats)
    out = grid["dir"] / "proxy_catalogue.tif"
    stats_path = grid["dir"] / "proxy_stats.json"
    monkeypatch.setattr(sys, "argv", [
        "build_proxy_catalogue.py", "--proxy", str(proxy_path),
        "--template", str(grid["labels_path"]), "--labels", str(grid["labels_path"]),
        "--out", str(out), "--stats", str(stats_path),
        "--fetch-meta", str(grid["dir"] / "absent.json")])
    assert build.main() == 0, "proxy with an absent-fault trace must not be refused"
    with rasterio.open(out) as src:
        coded = src.read(1)
    assert coded[40:88, 10].max() == 1            # on the label -> already in the catalogue
    assert coded[40:88, 12].max() == 1            # 2 px away, R=3 -> still in the catalogue
    assert coded[40:88, 40].max() == 2            # 30 px away -> absent from the labels
    stats = json.loads(stats_path.read_text())
    assert stats["proxy"]["proxy_only_px"] > 0 and stats["proxy"]["near_label_px"] > 0
    assert 0.0 < stats["proxy"]["catalogue_already_covers_fraction"] < 1.0
    assert stats["proxy"]["proxy_only_components"]["components"] == 1
    assert stats["inputs"]["R_pixels"] == 3


def test_build_refuses_a_grid_mismatch(grid, tmp_path, monkeypatch):
    build = _load("build_proxy_catalogue")
    other = tmp_path / "other_grid.tif"
    with rasterio.open(other, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
                       crs="EPSG:26911", transform=from_origin(0, 0, 100, 100)) as dst:
        dst.write(np.zeros((8, 8), dtype="float32"), 1)
    proxy = _write_geojson(tmp_path / "p.geojson", [_line_geojson(10)])
    monkeypatch.setattr(sys, "argv", [
        "build_proxy_catalogue.py", "--proxy", str(proxy), "--template",
        str(grid["labels_path"]), "--labels", str(other), "--out", str(tmp_path / "o.tif"),
        "--stats", str(tmp_path / "s.json")])
    with pytest.raises(SystemExit, match="not on the"):
        build.main()


def test_build_refuses_an_empty_proxy(tmp_path):
    build = _load("build_proxy_catalogue")
    empty = _write_geojson(tmp_path / "empty.geojson", [])
    with pytest.raises(SystemExit, match="no features"):
        build.load_geojson(empty)


# --------------------------------------------------------------------------------------------
# evaluate side: the acceptance property that makes the population usable
# --------------------------------------------------------------------------------------------

def test_catalogue_copy_scores_exactly_zero_on_proxy_only_truth():
    """The identity the instrument rests on: code-2 pixels are >R from every labelled pixel, so a
    perfect reproduction of the catalogue earns zero TP credit - the number cannot be gamed by
    restating the labels, which is exactly what catalogue DTI rewards."""
    size = 64
    truth = np.zeros((size, size), dtype="float32")
    truth[8:56, 40] = 1.0                                  # proxy-only fault
    catalogue = np.zeros_like(truth)
    catalogue[8:56, 10] = 1.0                              # a labelled fault 30 px away
    ctx = GtContext(truth, R_pixels=3)
    dti, (tp, fp, fn) = ctx.score(catalogue, alpha=0.2, beta=0.8, eps=1e-7, return_components=True)
    assert tp == 0.0 and dti == 0.0 and fn > 0.0


def test_eval_reports_baselines_and_the_acceptance_flag(grid, tmp_path, monkeypatch):
    """End-to-end: build a proxy with an absent fault, then score a prediction that (a) covers the
    listed fault only, (b) covers the absent fault, (c) blankets the map."""
    build = _load("build_proxy_catalogue")
    ev = _load("eval_proxy_catalogue")
    size = grid["size"]
    proxy_path = _write_geojson(tmp_path / "proxy.geojson", [_line_geojson(10), _line_geojson(60)])
    coded_path = tmp_path / "proxy.tif"
    monkeypatch.setattr(sys, "argv", [
        "build_proxy_catalogue.py", "--proxy", str(proxy_path), "--template",
        str(grid["labels_path"]), "--labels", str(grid["labels_path"]),
        "--out", str(coded_path), "--stats", str(tmp_path / "stats.json"),
        "--fetch-meta", str(tmp_path / "absent.json")])
    assert build.main() == 0

    pred = np.zeros((size, size), dtype="float32")
    pred[8:88, 10] = 1.0                                   # catalogue-shaped prediction
    pred_path = tmp_path / "pred.tif"
    with rasterio.open(pred_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(pred, 1)

    out = tmp_path / "eval.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(pred_path), "--proxy", str(coded_path),
        "--labels", str(grid["labels_path"]), "--out", str(out)])
    assert ev.main() == 0                                  # exit 0 == the acceptance property held
    rep = json.loads(out.read_text())
    assert rep["results"]["as_submitted"]["dti"] == 0.0, "a catalogue-only prediction earns nothing"
    assert rep["results"]["baselines"]["catalogue_copy"]["dti"] == 0.0
    assert rep["results"]["baselines"]["blanket_ones"]["dti"] > 0.0
    assert rep["results"]["baselines"]["zeros"]["dti"] == 0.0
    assert rep["results"]["acceptance"]["passed"] is True
    assert rep["truth"]["mode"] == "only" and rep["truth"]["px"] > 0
    assert rep["inputs"]["metric"]["R_pixels"] == 3


def test_eval_refuses_an_empty_truth_set(grid, tmp_path, monkeypatch):
    ev = _load("eval_proxy_catalogue")
    size = grid["size"]
    coded = np.zeros((size, size), dtype="uint8")          # no proxy faults at all
    coded_path = tmp_path / "empty_proxy.tif"
    with rasterio.open(coded_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="uint8", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(coded, 1)
    pred_path = tmp_path / "pred.tif"
    with rasterio.open(pred_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(np.ones((size, size), dtype="float32"), 1)
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(pred_path), "--proxy", str(coded_path),
        "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="empty truth set"):
        ev.main()


def test_sweep_ranks_emission_widths_on_the_proxy_population(grid, tmp_path, monkeypatch):
    """The width question the two existing measurements disagreed about, on the right population."""
    ev = _load("eval_proxy_catalogue")
    size = grid["size"]
    # an ensemble-like probability field: the absent fault at column 60, plus faint mass elsewhere
    prob = np.zeros((size, size), dtype="float32")
    prob[10:80, 60] = 0.9
    prob[10:80, 58:63] = 0.4
    p_path = tmp_path / "prob.tif"
    with rasterio.open(p_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(prob, 1)
    truth = np.zeros((size, size), dtype="float32")
    truth[10:80, 61] = 1.0                                 # truth offset by 1 px from the ridge
    coded = np.where(truth > 0, 2, 0).astype("uint8")
    c_path = tmp_path / "coded.tif"
    with rasterio.open(c_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="uint8", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(coded, 1)
    out = tmp_path / "sweep.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(c_path),
        "--out", str(out), "--sweep", "--shaping-grid", "3", "--dilate-grid", "0,1,2,4"])
    assert ev.main() == 0
    rep = json.loads(out.read_text())
    widths = rep["results"]["best_per_emission_width"]
    assert set(widths) == {"0", "1", "2", "4"}
    # truth is 1 px off the 0.9 ridge: the skeleton should still win by a clear margin here
    assert widths["0"] >= widths["4"] - 1e-12
    assert rep["results"]["sweep_verdict"]["skeleton_dti"] == widths["0"]
    assert all(k in rep["results"]["sweep_best"] for k in ("t0", "thin", "dilate", "dti"))


def test_sweep_scores_the_shipped_floor_and_compares_against_it(grid, tmp_path, monkeypatch):
    """MEASURED gap (session 10): the ensemble-1 proxy sweep evaluated floors [0, 1e-4, 2.08e-3,
    4.33e-2, 0.9] - log-spaced, so on that field the first four rows are the SAME mask and the
    shipped floor (0.469674) was never scored.  Its acceptance rule then compared candidates
    against the t0=0 skeleton (0.0144) instead of the shipped policy (0.0247).  `--reference-t0`
    puts the shipped floor in the grid and makes it the reference row."""
    ev = _load("eval_proxy_catalogue")
    size = grid["size"]
    prob = np.zeros((size, size), dtype="float32")
    prob[10:80, 60] = 0.9
    prob[10:80, 58:63] = 0.4
    p_path = tmp_path / "prob.tif"
    with rasterio.open(p_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(prob, 1)
    truth = np.zeros((size, size), dtype="float32")
    truth[10:80, 61] = 1.0
    coded = np.where(truth > 0, 2, 0).astype("uint8")
    c_path = tmp_path / "coded.tif"
    with rasterio.open(c_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="uint8", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(coded, 1)

    out = tmp_path / "sweep_ref.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(c_path),
        "--out", str(out), "--sweep", "--shaping-grid", "3", "--dilate-grid", "0,4",
        "--reference-t0", "0.4"])
    assert ev.main() == 0
    rep = json.loads(out.read_text())
    rows = rep["results"]["shaping_sweep"]
    at_ref = [r for r in rows if r["t0"] == 0.4]
    assert [r["dilate"] for r in at_ref] == [0, 4], "the shipped floor must be swept at every width"
    v = rep["results"]["sweep_verdict"]
    current = next(r for r in at_ref if r["dilate"] == 0)
    assert v["reference_t0"] == 0.4 and rep["inputs"]["reference_t0"] == 0.4
    assert v["current_policy_dti"] == current["dti"] != v["skeleton_dti"]
    assert v["beats_current_policy_by"] == round(rep["results"]["sweep_best"]["dti"] - current["dti"], 6)

    # without a reference floor the fields are explicit None, never a silent stand-in
    out2 = tmp_path / "sweep_noref.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(c_path),
        "--out", str(out2), "--sweep", "--shaping-grid", "3", "--dilate-grid", "0"])
    assert ev.main() == 0
    v2 = json.loads(out2.read_text())["results"]["sweep_verdict"]
    assert v2["reference_t0"] is None and v2["current_policy_dti"] is None
    assert "NOT the shipped policy" in v2["acceptance_rule"]


def test_reference_floor_can_come_from_a_blend_report(grid, tmp_path, monkeypatch):
    """The sweep job already has the blend report of the very policy it is evaluating; read the
    shipped floor from it rather than hard-coding a number that can drift."""
    ev = _load("eval_proxy_catalogue")
    size = grid["size"]
    prob = np.zeros((size, size), dtype="float32")
    prob[10:80, 60] = 0.9
    p_path = tmp_path / "prob.tif"
    with rasterio.open(p_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(prob, 1)
    coded = np.zeros((size, size), dtype="uint8")
    coded[10:80, 61] = 2
    c_path = tmp_path / "coded.tif"
    with rasterio.open(c_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="uint8", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(coded, 1)

    report = tmp_path / "blend_report.json"
    report.write_text(json.dumps({"shaping": {"t0": 0.469674, "thin": True, "dilate": 0}}))
    out = tmp_path / "sweep_report.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(c_path),
        "--out", str(out), "--sweep", "--shaping-grid", "3", "--dilate-grid", "0",
        "--reference-report", str(report)])
    assert ev.main() == 0
    rep = json.loads(out.read_text())
    assert rep["inputs"]["reference_t0"] == 0.469674
    assert rep["inputs"]["reference_source"].endswith("blend_report.json:shaping.t0")
    assert any(r["t0"] == 0.469674 for r in rep["results"]["shaping_sweep"])

    # an unshaped report is a loud error, not a silent 0
    report.write_text(json.dumps({"shaping": {"t0": None, "thin": None, "dilate": 0}}))
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(c_path),
        "--out", str(tmp_path / "x.json"), "--sweep", "--reference-report", str(report)])
    with pytest.raises(SystemExit, match="no shaping.t0"):
        ev.main()


# ---------------------------------------------------------------------------------------------
# The projection onto a hidden scored truth of unknown size (scripts/eval_proxy_catalogue.py).
#
# Why this exists: an absolute DTI is only a monitor - beta*|G| depends on a label set nobody has
# seen.  FP_w is set by the prediction, so the two error terms do not scale together, and a policy
# that wins on the proxy (6,166 km of trace) need not win on the scored set.  These tests pin the
# projection's algebra and the fact that it is capable of showing a crossover at all.
# ---------------------------------------------------------------------------------------------

def test_projection_reproduces_the_measured_dti_at_the_proxy_size():
    ev = _load("eval_proxy_catalogue")
    tp, fp, g = 5000.0, 1234.0, 100000.0
    measured = tp / (0.2 * (tp + fp) + 0.8 * g)
    assert abs(ev.project_dti(tp / g, fp, g) - measured) < 1e-12


def test_projection_finds_the_size_at_which_the_skeleton_stops_winning():
    ev = _load("eval_proxy_catalogue")
    # A skeleton that reaches 5 % of the truth with almost no wrong mass, against a 3-px band that
    # reaches three times as much but pays 30x the false positives: the classic trade of this metric.
    sweep = [{"t0": 0.5, "thin": True, "dilate": 0, "TP_w": 0.05 * 1e5, "FP_w": 1000.0},
             {"t0": 0.5, "thin": True, "dilate": 3, "TP_w": 0.15 * 1e5, "FP_w": 30000.0}]
    rows, sens = ev.sensitivity_table(sweep, n_truth=100000, sizes=[1000, 5000, 20000, 100000])
    per = {p["truth_px"]: p for p in sens["per_size"]}
    assert not per[1000]["best_is_wider_than_skeleton"], "a small truth cannot pay for a wide band"
    assert per[100000]["best_is_wider_than_skeleton"], "a large truth must reward coverage"
    assert per[100000]["passes_acceptance"] is True
    assert per[100000]["gain_over_skeleton"] > 0.01
    assert 1000 in sens["verdict"]["sizes_where_the_skeleton_wins"]
    assert 100000 in sens["verdict"]["sizes_where_a_wider_band_passes"]
    # the projection is monotone in truth size for a fixed policy (more truth, same coverage, the
    # wrong-mass is amortised further)
    d = [r["projected_dti"]["1000"] for r in rows if r["dilate"] == 0][0]
    e = [r["projected_dti"]["100000"] for r in rows if r["dilate"] == 0][0]
    assert e > d


# ---------------------------------------------------------------------------------------------
# The scoring contract of scripts/eval_proxy_catalogue.py (2026-09-16, session 7 fixes)
#
# Three defects were found by inspection and are pinned here, because each one silently changed
# what the published numbers MEAN while still producing a plausible number:
#   (a) truth length was converted with 0.01 km/px instead of 0.1 km/px, so every committed truth
#       length was reported 10x short (61,664 px printed as 616.6 km instead of 6,166 km);
#   (b) the blanket-ones baseline was taken over np.isfinite(pred), so its definition - and its
#       value - changed with whichever raster was being scored;
#   (c) the score of the raster handed to --pred was called "as_submitted", which is only true when
#       --pred IS the committed submission (it was not, in the shaping sweep, and the site printed
#       that row under "this submission, as committed").
# ---------------------------------------------------------------------------------------------

def _write(path, arr, transform, dtype):
    kw = {"nodata": np.nan} if dtype.startswith("float") else {}
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs="EPSG:32611", transform=transform, **kw) as dst:
        dst.write(arr.astype(dtype), 1)


def test_eval_units_support_and_role_are_unambiguous(tmp_path):
    ev = _load("eval_proxy_catalogue")
    size = 64
    transform = from_origin(500000.0, 4100000.0, 100.0, 100.0)
    # a fault footprint: NaN edges (outside the data footprint), one truth trace inside
    labels = np.full((size, size), np.nan, dtype="float32")
    labels[8:56, 8:56] = 0.0
    labels[20:40, 30] = 1.0                       # the catalogue lives inside the footprint
    lab_path = tmp_path / "labels.tif"
    _write(lab_path, labels, transform, "float32")
    # a proxy fault absent from the catalogue (code 2), and the labelled one (code 1)
    coded = np.zeros((size, size), dtype="uint8")
    coded[20:40, 30] = 1
    coded[10:50, 45] = 2
    coded_path = tmp_path / "coded.tif"
    _write(coded_path, coded, transform, "uint8")
    # a prediction that is finite EVERYWHERE (an ensemble map, not a legal submission) and misses
    # the absent fault: this is the case that made the old baseline definition ambiguous
    pred = np.full((size, size), 1.0, dtype="float32")
    pred[10:50, 44] = 1.0
    p_path = tmp_path / "prob.tif"
    _write(p_path, pred, transform, "float32")

    out = tmp_path / "ev.json"
    argv = ["eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(coded_path),
            "--labels", str(lab_path), "--out", str(out)]
    old = sys.argv
    try:
        sys.argv = argv
        assert ev.main() == 0
    finally:
        sys.argv = old
    rep = json.loads(out.read_text())

    # (a) 100 m pixels: km = px * 0.1, and the truth is the code-2 trace only
    assert rep["truth"]["mode"] == "only"
    assert rep["truth"]["px"] == 40
    assert abs(rep["truth"]["km"] - rep["truth"]["px"] * 0.1) < 1e-9

    # (b) the blanket baseline is anchored to the LABEL footprint, not to the prediction
    assert rep["baseline_support"]["source"].startswith("labels")
    assert rep["baseline_support"]["px"] == int(np.isfinite(labels).sum())
    assert rep["baseline_support"]["px"] < size * size          # the footprint really is smaller
    b = rep["results"]["baselines"]
    assert b["blanket_ones"]["dti"] > b["blanket_ones_whole_grid"]["dti"], \
        "covering only the legal footprint must beat covering the whole grid"

    # (c) the provided raster is named as such, keeps the old key as an alias, and the score of its
    #     legal (footprint-clipped) version is recorded because it differs
    r = rep["results"]
    assert r["as_provided"] == r["as_submitted"], "the alias must be the same measurement"
    assert "prediction_role" in r
    assert "as_provided_clipped_to_footprint" in r, "mass outside the footprint must be re-scored"
    assert r["as_provided_clipped_to_footprint"]["dti"] > r["as_provided"]["dti"], \
        "dropping illegal out-of-footprint mass cannot hurt"


def test_emission_decision_reconciles_the_measurements():
    """scripts/decide_emission_width.py: the projection and crossover algebra, on hand values."""
    dec = _load("decide_emission_width")
    # identical to eval's project_dti for alpha+beta=1, and exact at |G| = |G_proxy|
    row = {"policy": "p", "coverage_fraction": 0.05, "wrong_mass_FP_w": 1000.0}
    g = 100000
    measured = (0.05 * g) / (0.2 * (0.05 * g + 1000.0) + 0.8 * g)
    assert abs(dec.project(row, g) - measured) < 1e-12
    # the crossover: a cleaner policy wins below it, the wider one above it
    clean = {"policy": "clean", "coverage_fraction": 0.05, "wrong_mass_FP_w": 1000.0}
    wide = {"policy": "wide", "coverage_fraction": 0.15, "wrong_mass_FP_w": 30000.0}
    x = dec.crossover(clean, wide)
    assert x["crossover_px"] and x["crossover_px"] > 0
    assert dec.project(clean, x["crossover_px"]) == pytest.approx(
        dec.project(wide, x["crossover_px"]), abs=1e-9), "the curves must actually cross there"
    assert dec.project(clean, x["crossover_px"] / 4) > dec.project(wide, x["crossover_px"] / 4)
    assert dec.project(wide, x["crossover_px"] * 4) > dec.project(clean, x["crossover_px"] * 4)
    # a policy that is better on both axes has no crossing (it dominates)
    dom = {"policy": "dom", "coverage_fraction": 0.05, "wrong_mass_FP_w": 100.0}
    assert dec.crossover(dom, clean)["crossover_px"] is None


def test_committed_emission_decision_states_its_conditions():
    """The committed record must state its conditions - and its verdict must FOLLOW from them.

    This test used to require an unmet condition ("an unmet condition must be recorded, not
    hidden"), which pinned the state of the evidence rather than the rule: once the ensemble-2
    sweep satisfied condition 3 the same assertion failed *because the pipeline worked*.  What
    actually has to hold is the implication: the conclusion is one of the three derived branches
    and it cannot claim more than the conditions passed.
    """
    p = ROOT / "data/evidence/emission_decision.json"
    if not p.exists():
        pytest.skip("emission decision not computed in this checkout")
    d = json.loads(p.read_text())
    v = d["verdict"]
    conditions = v["conditions"]
    assert len(conditions) >= 3
    assert all(isinstance(c["passes"], bool) for c in conditions)
    assert any("0.01" in c["condition"] for c in conditions), \
        "the pre-registered gain condition must be present"
    assert d["crossovers"]["blanket_vs_shipped"]["crossover_px"] > 0
    cond3 = next(c for c in conditions if "second" in c["condition"])
    measured = "NOT MEASURED" not in cond3["measured"]
    if all(c["passes"] for c in conditions):
        assert v["conclusion"].startswith("SHIP the measured policy")
        assert measured, "a SHIP verdict needs a measured reproduction, not an assumed one"
    elif not measured:
        assert v["conclusion"].startswith("measured, not yet reproduced")
    else:
        assert v["conclusion"].startswith("measured, not shipped")
    # whatever the branch, a SHIP cannot be claimed while a condition is unmet
    if not all(c["passes"] for c in conditions):
        assert not v["conclusion"].startswith("SHIP")


# ---------------------------------------------------------------------------------------------
# Session 11: the two axes of the shaping search, and the measurement that decides how many floors
# are worth scoring at all.
#
# Session 10 found the ensemble-1 sweep's floor dimension was effectively binary: log-spaced floors
# [0, 1e-4, 2.08e-3, 4.33e-2, 0.9] produced the SAME emitted mask, because the field emits nothing
# between 0.043 and 0.9.  A finer grid cannot fix that, and no reader could tell from the evidence
# whether the floors were identical or the search was broken.  So the sweep now records, per floor,
# how many pixels the field has above it BEFORE thinning - and it scores a second axis (the values
# written into the band, hard vs ramp) whose candidates are provably on the same support.
# ---------------------------------------------------------------------------------------------

def _prob_and_proxy(tmp_path, grid, ridge_cols=(19, 20, 21), truth_col=61):
    """A faint-shouldered ridge 1 px off the truth, plus a coded truth raster."""
    size = grid["size"]
    prob = np.zeros((size, size), dtype="float32")
    prob[10:80, ridge_cols[0]:ridge_cols[-1] + 1] = 0.4
    prob[10:80, ridge_cols[1]] = 0.9
    p_path = tmp_path / "prob_band.tif"
    with rasterio.open(p_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(prob, 1)
    coded = np.zeros((size, size), dtype="uint8")
    coded[10:80, truth_col] = 2
    c_path = tmp_path / "coded_band.tif"
    with rasterio.open(c_path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="uint8", crs="EPSG:32611", transform=grid["transform"]) as dst:
        dst.write(coded, 1)
    return p_path, c_path


def test_field_mass_profile_makes_the_floor_grid_auditable(grid, tmp_path, monkeypatch):
    ev = _load("eval_proxy_catalogue")
    p_path, c_path = _prob_and_proxy(tmp_path, grid)
    out = tmp_path / "sweep_profile.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(c_path),
        "--out", str(out), "--sweep", "--shaping-values", "0,0.3,0.95", "--dilate-grid", "0,2"])
    assert ev.main() == 0
    rep = json.loads(out.read_text())
    prof = rep["results"]["field_mass_profile"]
    assert [r["t0"] for r in prof["rows"]] == [0.0, 0.3, 0.95]
    # floors 0.0 and 0.3 see the same mask (the shoulder is 0.4, above both); 0.95 sees only the
    # ridge - so the 0.0/0.3 pair is indistinguishable and the grid above 0.4 is what matters
    assert prof["rows"][0]["support_px_above_floor"] == prof["rows"][1]["support_px_above_floor"]
    assert prof["rows"][2]["support_px_above_floor"] < prof["rows"][0]["support_px_above_floor"]
    assert prof["distinct_supports"] == 2
    assert "finer floor grid" in prof["reading"]
    # an explicit grid is honoured, and the reference floor is still appended when given
    assert sorted({r["t0"] for r in rep["results"]["shaping_sweep"]}) == [0.0, 0.3, 0.95]


def test_ramp_candidates_are_scored_on_the_same_support_as_the_hard_band(grid, tmp_path, monkeypatch):
    ev = _load("eval_proxy_catalogue")
    p_path, c_path = _prob_and_proxy(tmp_path, grid)
    out = tmp_path / "sweep_ramp.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_proxy_catalogue.py", "--pred", str(p_path), "--proxy", str(c_path),
        "--out", str(out), "--sweep", "--shaping-values", "0.3", "--dilate-grid", "0,1,2",
        "--soft-band", "--band-gammas", "1,2", "--reference-t0", "0.3"])
    assert ev.main() == 0
    rep = json.loads(out.read_text())
    res = rep["results"]
    rows = res["shaping_sweep"]
    hard = [r for r in rows if not r.get("soft")]
    ramp = [r for r in rows if r.get("soft")]
    assert hard and ramp, "both axes must be scored"
    # every ramp candidate has a hard candidate with the SAME (floor, width) - i.e. same support
        # by construction, so the DTI difference is the value of the ramp alone
    for r in ramp:
        twin = next((h for h in hard if h["t0"] == r["t0"] and h["dilate"] == r["dilate"]), None)
        assert twin is not None, f"ramp candidate without a hard twin: {r}"
        assert r["mean_kept_px"] == twin["mean_kept_px"], "supports differ - not a values-only A/B"
    assert {r["gamma"] for r in ramp} == {1.0, 2.0}
    assert res["best_per_emission"]["ramp_n_candidates"] == len(ramp)
    assert res["best_per_emission"]["hard"] is not None
    assert res["sweep_best"]["dti"] == max(r["dti"] for r in rows)
    # the projection table describes policies by (floor, thin, band width), which does not identify a
    # ramp candidate: ramp rows must not leak into it
    assert len(res["sensitivity"]["rows"]) == len(hard)
    assert {r["dilate"] for r in res["sensitivity"]["rows"]} == {h["dilate"] for h in hard}
    # width-0 rows are not ramp candidates: a ramp needs a width
    assert all(r["dilate"] > 0 for r in ramp)
    # the shipped-policy reference is a hard, un-widened row and stays identifiable as such
    assert res["sweep_verdict"]["current_policy_dti"] == next(
        h["dti"] for h in hard if h["t0"] == 0.3 and h["dilate"] == 0)
