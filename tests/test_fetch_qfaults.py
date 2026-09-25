"""QFaults fetch (scripts/fetch_qfaults.py) - catalogue B for the cross-catalogue transfer test.

No network: `http_json` and `head_check` are replaced by a router that answers from a synthetic
service whose metadata mimics the real layer 21 response verified live on 2026-09-18
(esriGeometryPolyline, maxRecordCount 2000, symbology-coded renderer vocabulary).

What is pinned here:
* the INTEGRITY GATE - fetched count must equal the count the service itself reported, otherwise
  the run fails instead of writing a silently truncated catalogue (a partial catalogue would
  understate transfer and look like a negative scientific result);
* paging with resultOffset/resultRecordCount until the count is reached, with no duplicates;
* RuleIDs are derived from the service's OWN renderer vocabulary, and a class the renderer did not
  list is appended with a stable id and reported - never dropped;
* the output is consumable by scripts/build_proxy_catalogue.py unchanged (same RuleID contract),
  which is what makes catalogue B a drop-in second population;
* the failure modes that would otherwise produce a plausible-looking but wrong catalogue:
  a non-polyline layer, a zero/negative count, and runaway paging.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import urllib.parse
from pathlib import Path

import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LAYER_META = {
    "name": "National Database",
    "geometryType": "esriGeometryPolyline",
    "maxRecordCount": 2000,
    "fields": [{"name": "OBJECTID"}, {"name": "fault_name"}, {"name": "symbology"},
               {"name": "age"}, {"name": "linetype"}, {"name": "slip_rate"}],
    "drawingInfo": {"renderer": {"uniqueValueInfos": [
        {"value": "latest Quaternary Moderately Constrained"},
        {"value": "latest Quaternary Well Constrained"},
        {"value": "Quaternary age uncertain"},
        {"value": "latest Quaternary Moderately Constrained"},      # duplicate -> deduped
    ]}},
}


def _features(n: int, start: int = 1) -> list[dict]:
    """n synthetic polyline features cycling the three renderer classes (plus one unlisted)."""
    classes = ["latest Quaternary Moderately Constrained",
               "latest Quaternary Well Constrained",
               "Quaternary age uncertain",
               "Holocene age, undocumented class"]                  # not in the vocabulary
    return [{"type": "Feature",
             "properties": {"OBJECTID": start + i,
                            "fault_name": f"Synthetic fault {start + i}",
                            "symbology": classes[i % len(classes)],
                            "age": "Qa", "linetype": "certain", "slip_rate": "1"},
             "geometry": {"type": "LineString",
                          "coordinates": [[-118.0 + 0.001 * i, 38.0],
                                          [-118.0 + 0.001 * i, 38.02]]}}
            for i in range(n)]


class FakeService:
    """Answers exactly the URLs fetch_qfaults.py builds, from an in-memory feature list."""

    def __init__(self, features: list[dict], count: int | None = None,
                 layer_meta: dict | None = None, page_cap: int | None = None,
                 runaway: bool = False):
        self.features = features
        self.count = len(features) if count is None else count
        self.layer_meta = LAYER_META if layer_meta is None else layer_meta
        self.page_cap = page_cap                 # simulate a service that refuses to page
        self.runaway = runaway                   # simulate a query that never converges
        self.urls: list[str] = []
        self.pages: list[int] = []

    def __call__(self, url: str, timeout: int = 300, attempts: int = 4) -> dict:
        self.urls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if url.endswith("?f=json") or "/query" not in url:
            return self.layer_meta
        if q.get("returnCountOnly", ["false"])[0] == "true":
            return {"count": self.count}
        offset = int(q.get("resultOffset", ["0"])[0])
        want = int(q.get("resultRecordCount", ["1000"])[0])
        self.pages.append((offset, want))
        batch = self.features[offset:offset + want]
        if self.page_cap is not None and offset >= self.page_cap:
            batch = []                            # service silently stops paging
        if self.runaway and offset < self.count:
            # a full page every time, forever: the offset never reaches the reported count
            batch = _features(want, start=offset + 1)
        served = offset + len(batch)
        return {"type": "FeatureCollection", "features": batch,
                "exceededTransferLimit": served < min(self.count, len(self.features))}


def _write_tiny(tmp_path, name: str = "labels.tif") -> Path:
    """A 4x4 EPSG:32611 raster: enough for bbox_from_raster to compute a real envelope."""
    import numpy as _np
    p = tmp_path / name
    with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8",
                       crs="EPSG:32611",
                       transform=from_origin(243350.0, 4508550.0, 100.0, 100.0)) as dst:
        dst.write(_np.zeros((4, 4), _np.uint8), 1)
    return p


@pytest.fixture()
def bbox_source(tmp_path) -> Path:
    return _write_tiny(tmp_path)


def _run(m, service, tmp_path, bbox_source, extra=()):
    m.http_json = service
    m.head_check = lambda urls: {u: f"OK_200 ({k})" for k, u in urls.items()}
    out, meta = tmp_path / "qfaults.geojson", tmp_path / "fetch_meta.json"
    argv = ["--bbox-source", str(bbox_source), "--out", str(out), "--meta", str(meta),
            "--page-size", "1000", *extra]
    rc = m.main(argv)
    return rc, out, meta


# --------------------------------------------------------------------------------------
# the class vocabulary comes from the service, not from this repository
# --------------------------------------------------------------------------------------
def test_class_vocabulary_is_read_from_the_renderer_and_sorted():
    m = _load("fq", "fetch_qfaults.py")
    vocab = m.class_vocabulary(LAYER_META)
    assert vocab == sorted(set(v["value"] for v in
                               LAYER_META["drawingInfo"]["renderer"]["uniqueValueInfos"]))
    assert len(vocab) == 3, "the duplicated renderer value must not create a fourth class"
    assert m.class_vocabulary({}) == []
    grouped = {"drawingInfo": {"renderer": {"uniqueValueGroups": [
        {"classes": [{"values": [["b"], ["a"]]}]}]}}}
    assert m.class_vocabulary(grouped) == ["a", "b"]


def test_unlisted_classes_are_appended_and_recorded_not_dropped():
    m = _load("fq", "fetch_qfaults.py")
    vocab = m.class_vocabulary(LAYER_META)
    extra: dict = {}
    known = m.feature_class({"symbology": vocab[0]}, vocab, extra)
    assert known in (1, 2, 3) and extra == {}
    unknown = m.feature_class({"symbology": "Holocene age, undocumented class"}, vocab, extra)
    assert unknown == 4 and extra == {"Holocene age, undocumented class": 4}
    # the fallback fields are used when symbology is missing, and a placeholder when all are
    missing = m.feature_class({"age": "Qa", "linetype": "certain"}, vocab, extra)
    assert missing == 5 and "Qa certain" in extra
    blank = m.feature_class({}, vocab, extra)
    assert blank == 6 and "<missing class>" in extra
    # ids stay stable for a value already seen
    assert m.feature_class({"symbology": "Holocene age, undocumented class"}, vocab, extra) == 4


# --------------------------------------------------------------------------------------
# query construction
# --------------------------------------------------------------------------------------
def test_build_query_carries_the_footprint_bbox_and_paging():
    m = _load("fq", "fetch_qfaults.py")
    url = m.build_query("https://example.test/MapServer", 21,
                        (-120.0372, 37.3312, -116.1409, 40.7279), 2000, 1000)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["geometry"] == ["-120.0372,37.3312,-116.1409,40.7279"]
    assert q["geometryType"] == ["esriGeometryEnvelope"] and q["inSR"] == ["4326"]
    assert q["spatialRel"] == ["esriSpatialRelIntersects"]
    assert q["outSR"] == ["4326"] and q["f"] == ["geojson"]
    assert q["resultOffset"] == ["2000"] and q["resultRecordCount"] == ["1000"]
    count_url = m.build_query("https://example.test/MapServer", 21,
                              (-120.0, 37.0, -116.0, 40.0), 0, 0, count_only=True)
    cq = urllib.parse.parse_qs(urllib.parse.urlparse(count_url).query)
    assert cq["returnCountOnly"] == ["true"] and cq["f"] == ["json"]
    assert "resultOffset" not in cq


def test_bbox_is_computed_from_the_raster_and_can_be_padded(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    bbox, grid = m.bbox_from_raster(bbox_source)
    assert grid["crs"] == "EPSG:32611" and (grid["width"], grid["height"]) == (4, 4)
    assert -121 < bbox[0] < bbox[2] < -115 and 37 < bbox[1] < bbox[3] < 42
    padded, _ = m.bbox_from_raster(bbox_source, pad_deg=0.25)
    assert padded[0] == pytest.approx(bbox[0] - 0.25) and padded[3] == pytest.approx(bbox[3] + 0.25)


# --------------------------------------------------------------------------------------
# the fetch itself
# --------------------------------------------------------------------------------------
def test_fetch_writes_every_feature_with_a_ruleid_and_full_provenance(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    feats = _features(2500)                       # forces 3 pages at page size 1000
    svc = FakeService(feats)
    rc, out, meta_path = _run(m, svc, tmp_path, bbox_source)
    assert rc == 0 and out.exists() and meta_path.exists()

    doc = json.loads(out.read_text())
    assert doc["type"] == "FeatureCollection" and len(doc["features"]) == 2500
    ids = [f["properties"]["OBJECTID"] for f in doc["features"]]
    assert len(set(ids)) == 2500, "paging must not duplicate or skip features"
    assert all(isinstance(f["properties"]["RuleID"], int) for f in doc["features"])
    assert {f["properties"]["RuleID"] for f in doc["features"]} == {1, 2, 3, 4}
    props = doc["properties"]
    assert props["doi"] == m.QFAULTS_DOI and props["service"].endswith("MapServer")
    assert props["layer"] == 21 and "Public Domain" in props["license"]

    meta = json.loads(meta_path.read_text())
    assert meta["counts"]["service_reported"] == 2500 == meta["counts"]["fetched"]
    assert meta["counts"]["line_geometries"] == 2500
    assert meta["classes"]["vocabulary_field"] == "symbology"
    assert meta["classes"]["n_classes_appended_at_runtime"] == 1
    assert set(meta["classes"]["rule_id_to_class"]) == {"1", "2", "3", "4"}
    assert meta["requests"]["pages"] == 3, "2500 features at page size 1000 is three pages"
    assert meta["requests"]["total_requests"] == len(svc.urls) == 5   # meta + count + 3 pages
    assert meta["requests"]["layer_metadata"].endswith("/21?f=json")
    assert "returnCountOnly=true" in meta["requests"]["count_only"]
    assert len(meta["requests"]["all_page_urls"]) == 3
    assert meta["source"]["geometry_type"] == "esriGeometryPolyline"
    assert meta["source"]["max_record_count"] == 2000
    assert meta["output"]["sha256"] == __import__("hashlib").sha256(
        out.read_bytes()).hexdigest()
    assert meta["bbox"]["pad_deg"] == 0.0
    assert any("build_proxy_catalogue.py" in s for s in meta["next_steps"]), \
        "the sidecar must tell the reader how to rasterise what was just fetched"


def test_paging_stops_on_a_short_page_and_on_the_reported_count(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    svc = FakeService(_features(1500))
    rc, out, _ = _run(m, svc, tmp_path, bbox_source)
    assert rc == 0
    assert json.loads(out.read_text())["features"].__len__() == 1500
    assert svc.pages == [(0, 1000), (1000, 1000)], f"unexpected paging: {svc.pages}"


def test_page_size_is_capped_by_the_service_maxrecordcount(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    svc = FakeService(_features(50))
    rc, _, meta_path = _run(m, svc, tmp_path, bbox_source, extra=("--page-size", "5000"))
    assert rc == 0
    meta = json.loads(meta_path.read_text())
    assert meta["source"]["page_size_used"] == 2000, \
        "asking for more than maxRecordCount would silently truncate every page"


# --------------------------------------------------------------------------------------
# integrity gates: a wrong catalogue must fail loudly, not look plausible
# --------------------------------------------------------------------------------------
def test_partial_fetch_is_refused(tmp_path, bbox_source):
    """A service that stops paging early must fail the run rather than ship 1000 of 3000."""
    m = _load("fq", "fetch_qfaults.py")
    feats = _features(3000)
    svc = FakeService(feats, count=3000, page_cap=1000)      # only the first page is served
    with pytest.raises(SystemExit) as exc:
        _run(m, svc, tmp_path, bbox_source)
    msg = str(exc.value)
    assert "refusing to write a partial catalogue" in msg and "3,000" in msg
    assert not (tmp_path / "qfaults.geojson").exists(), "no partial catalogue may be written"


def test_reported_count_mismatch_is_refused(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    svc = FakeService(_features(10), count=12)               # service claims 12, serves 10
    with pytest.raises(SystemExit) as exc:
        _run(m, svc, tmp_path, bbox_source)
    assert "fetched 10" in str(exc.value) and "reports 12" in str(exc.value)


def test_zero_count_is_refused(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    svc = FakeService([], count=0)
    with pytest.raises(SystemExit) as exc:
        _run(m, svc, tmp_path, bbox_source)
    assert "empty catalogue" in str(exc.value)


def test_a_non_polyline_layer_is_refused(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    meta = dict(LAYER_META, geometryType="esriGeometryPolygon", name="Fault Areas")
    svc = FakeService(_features(5), layer_meta=meta)
    with pytest.raises(SystemExit) as exc:
        _run(m, svc, tmp_path, bbox_source, extra=("--layer", "22"))
    assert "esriGeometryPolyline" in str(exc.value)
    assert not (tmp_path / "qfaults.geojson").exists()


def test_no_line_geometries_is_refused(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    feats = _features(5)
    for f in feats:
        f["geometry"] = {"type": "Point", "coordinates": [-118.0, 38.0]}
    svc = FakeService(feats)
    with pytest.raises(SystemExit) as exc:
        _run(m, svc, tmp_path, bbox_source)
    assert "no line geometries" in str(exc.value)


def test_runaway_paging_hits_the_hard_stop(tmp_path, bbox_source):
    m = _load("fq", "fetch_qfaults.py")
    svc = FakeService(_features(30), count=10_000_000,       # a count no bbox could produce
                      runaway=True)                          # and a query that never converges
    with pytest.raises(SystemExit) as exc:
        _run(m, svc, tmp_path, bbox_source, extra=("--max-features", "20"))
    assert "--max-features" in str(exc.value) or "not converging" in str(exc.value)


# --------------------------------------------------------------------------------------
# interop: build_proxy_catalogue.py consumes this output unchanged
# --------------------------------------------------------------------------------------
def test_the_fetched_geojson_is_a_drop_in_for_build_proxy_catalogue(tmp_path, bbox_source):
    """Same RuleID contract as the SGMC fetch, so catalogue B needs no second rasteriser."""
    fq = _load("fq", "fetch_qfaults.py")
    bp = _load("bp", "build_proxy_catalogue.py")
    svc = FakeService(_features(40))
    rc, out, _ = _run(fq, svc, tmp_path, bbox_source)
    assert rc == 0

    feats = bp.load_geojson(out)
    assert len(feats) == 40
    with rasterio.open(bbox_source) as src:
        dst_crs = src.crs
    # the synthetic features are drawn near -118/38, which is inside this grid's envelope
    projected, stats = bp.project_features(feats, dst_crs)
    assert stats["projected"] == 40, stats
    assert stats["dropped"].get("missing_rule_id", 0) == 0, \
        "every feature must carry an integer RuleID for the rasteriser"
    assert all(isinstance(rule, int) and rule >= 1 for _, rule in projected)


def test_class_vocabulary_key_is_found_in_both_provenance_layouts(tmp_path):
    """SGMC nests it under query.*, QFaults under classes.*; the rasteriser reads both."""
    bp_src = (ROOT / "scripts" / "build_proxy_catalogue.py").read_text()
    assert "query.rule_id_to_class" in bp_src and "classes.rule_id_to_class" in bp_src
    fq = _load("fq", "fetch_qfaults.py")
    svc = FakeService(_features(4))
    _, _, meta_path = _run(fq, svc, tmp_path, _write_tiny(tmp_path, "src.tif"))
    meta = json.loads(meta_path.read_text())
    assert "rule_id_to_class" in meta["classes"], "QFaults meta must expose the class map"
