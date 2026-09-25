"""The site's own ability to generate the submission .tif must be measured, not described.

docs/how_to_submit.html now carries a button that writes a GeoTIFF in the reader's browser. That is
an unusual claim for this repository: normally a claim is backed by an artefact produced here, but a
browser cannot run in CI. So this file pins three things at once:

1. the payload builder: what it ships is the artifact, and it REFUSES rather than quantising a field
   its encoding cannot represent losslessly;
2. the writer: the real JavaScript, run under node exactly as the page loads it, produces a file that
   rasterio accepts, that the repository's own format validator passes, and whose float32 buffer is
   bit-identical to the artifact's — in every container variant (deflate/uncompressed, 1-row strips,
   one strip for the whole grid), because a file that only works in the configuration it was authored
   in is a file that will fail on somebody's browser;
3. the glue: a DOM shim runs docs/generate_submission.js end to end (tests/support/
   generator_ui_harness.js) and checks that a tampered manifest is refused instead of downloaded.

Skips are honest and loud: no node means the browser route is unverifiable here, which is recorded as
a skip, never as a pass. The artifact route (curl) stays tested either way.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

META = ROOT / "docs" / "submission_meta.json"
BLOB = ROOT / "docs" / "submission_field.bin"
WRITER = ROOT / "docs" / "geotiff_writer.js"
GLUE = ROOT / "docs" / "generate_submission.js"
HARNESS = ROOT / "tests" / "support" / "generator_ui_harness.js"
def _shipped_artifact():
    """The artifact the site and the payload pin, derived - never a hardcoded run directory.

    Session 27 moved the shipped artifact from an 11-fold run to the measured union of the two
    detector families; the tests that hardcoded the old path kept passing against bytes the site no
    longer offered, which is exactly the drift they exist to catch.  The payload manifest
    (docs/submission_meta.json, written by scripts/build_submission_payload.py from its ARTIFACT
    constant) is the single source of truth, which is itself the constant build_site.py uses.
    """
    meta = ROOT / "docs/submission_meta.json"
    if meta.exists():
        try:
            rel = json.loads(meta.read_text()).get("artifact", {}).get("path")
            if rel and (ROOT / rel).exists():
                return ROOT / rel
        except Exception:
            pass
    return ROOT / "data/evidence/union/submission.tif"


ARTIFACT = _shipped_artifact()
SAMPLE = ROOT / "data/sample_submission.tif"


def _mod():
    spec = importlib.util.spec_from_file_location("build_submission_payload",
                                                 ROOT / "scripts" / "build_submission_payload.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _node():
    exe = shutil.which("node")
    if not exe:
        pytest.skip("node is not on PATH: the browser code path cannot be executed here")
    return exe


def _py():
    return os.environ.get("GEMS_PYTHON", sys.executable)


# --------------------------------------------------------------------- encoding round trips
def test_encode_decode_round_trip_is_bit_exact():
    m = _mod()
    rng = np.random.default_rng(7)
    a = np.zeros((7, 13), dtype="float32")
    a[rng.random(a.shape) < 0.2] = 1.0
    a[0, :4] = np.nan                       # a NaN band on the first row
    blob, runs = m.encode_runs(a)
    back = m.decode_runs(blob, a.shape[1], a.shape[0])
    assert back.dtype == np.float32
    assert np.array_equal(np.isnan(back), np.isnan(a)), "NaN positions must survive"
    assert back.astype("<f4").tobytes() == a.astype("<f4").tobytes(), "float32 bits must be identical"
    assert len(runs) > 1


def test_a_continuous_field_is_refused_not_quantised():
    """The honesty constraint: a probability field cannot be represented by gems-rle-v1, and
    silently snapping it to {0,1} would ship a different prediction than the pinned artifact."""
    m = _mod()
    a = np.zeros((4, 4), dtype="float32")
    a[1, 1] = 0.73
    with pytest.raises(ValueError) as ex:
        m.encode_runs(a)
    assert "not binary" in str(ex.value)
    assert "Refusing" in str(ex.value)


def test_truncated_and_overlong_run_streams_are_rejected():
    m = _mod()
    a = np.zeros((3, 3), dtype="float32")
    blob, _ = m.encode_runs(a)
    with pytest.raises(ValueError):
        m.decode_runs(blob[:-1], 3, 3)                    # cut mid-run
    with pytest.raises(ValueError):
        m.decode_runs(blob + blob, 3, 3)                   # covers the grid twice
    with pytest.raises(ValueError):
        m.decode_runs(bytes([0x05, 0x03]), 3, 3)          # five px of a nine px grid


def test_unknown_value_code_is_rejected():
    m = _mod()
    with pytest.raises(ValueError, match="unknown value code"):
        m.decode_runs(bytes([0x09, 0x07]), 3, 3)          # 9 px of code 7


# --------------------------------------------------------------------- what is shipped on the site
def test_the_committed_payload_still_reproduces_the_artifact():
    """The guard against a live generator describing a dead artifact: --check must be clean."""
    assert ARTIFACT.exists(), "the adopted artifact is committed; a fresh clone needs `git pull`"
    p = subprocess.run([_py(), "scripts/build_submission_payload.py", "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1200:]
    # The check's own words: it claims the payload reproduces the artifact and that the float32 bits
    # survive the decode round trip — the two properties the page's generator depends on.
    assert "reproduce" in p.stdout and "float32 bits identical" in p.stdout


def test_payload_verify_mode_decodes_back_to_the_artifact():
    p = subprocess.run([_py(), "scripts/build_submission_payload.py", "--verify"], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-800:]
    assert "float32 bits identical: True" in p.stdout


@pytest.mark.skipif(not META.exists(), reason="the site payload has not been built here")
def test_manifest_pins_agree_with_the_bytes_on_disk():
    meta = json.loads(META.read_text())
    raw = ARTIFACT.read_bytes()
    assert meta["artifact"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert meta["artifact"]["bytes"] == len(raw)
    assert meta["artifact"]["hash_matches"] is True, "the sidecar and the artifact disagree"
    blob = BLOB.read_bytes()
    assert meta["blob"]["sha256"] == hashlib.sha256(blob).hexdigest()
    assert meta["blob"]["bytes"] == len(blob)
    assert meta["encoding"]["format"] == "gems-rle-v1"
    assert meta["round_trip"]["float32_bytes_identical"] is True
    assert meta["encoding"]["expected_pixels"] == meta["grid"]["width"] * meta["grid"]["height"]
    # the grid in the manifest is the grid GDAL sees in the artifact - re-read, not copied
    with rasterio.open(ARTIFACT) as src:
        assert meta["grid"]["width"] == src.width and meta["grid"]["height"] == src.height
        assert meta["grid"]["epsg"] == src.crs.to_epsg()
        assert [float(x) for x in meta["grid"]["transform"]] == [float(x) for x in src.transform]
    # The sample-derived row is a measurement of the build machine's data/, which is NOT committed
    # (418 MB placed by scripts/assemble_data_bridge.py). Assert the recorded pins always; open the
    # file and re-derive them only where the data actually exists, so a CI runner without a
    # placement exercises the claim instead of erroring on a missing file.
    assert meta["checks"]["sample_submission"]["shape_matches"] is True
    assert meta["checks"]["sample_submission"]["transform_matches"] is True
    assert meta["checks"]["sample_submission"]["epsg"] == 32611
    if SAMPLE.exists():
        with rasterio.open(SAMPLE) as s:
            assert (s.width, s.height) == tuple(meta["checks"]["sample_submission"]["grid"])
            assert [float(x) for x in s.transform] == [float(x) for x in
                                                        meta["checks"]["sample_submission"]["transform"]]


# --------------------------------------------------------------------- the writer, run by node
@pytest.fixture(scope="module")
def gen_dir(tmp_path_factory):
    node = _node()
    out = tmp_path_factory.mktemp("gems-gen")
    tif, zip_ = out / "submission.tif", out / "submission.zip"
    p = subprocess.run([node, str(WRITER), str(META), str(BLOB), str(tif), str(zip_)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    report = json.loads(p.stdout)
    assert report["ok"] and report["failed"] == [], report["failed"]
    yield out, tif, zip_, report
    if os.environ.get("GEMS_KEEP_TMP"):
        print("kept:", out)


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not in this checkout")
def test_node_generated_file_is_pixel_identical_to_the_artifact(gen_dir):
    _, tif, _, report = gen_dir
    with rasterio.open(tif) as g, rasterio.open(ARTIFACT) as a:
        gb = np.ascontiguousarray(g.read(1), dtype="<f4").tobytes()
        ab = np.ascontiguousarray(a.read(1), dtype="<f4").tobytes()
        assert gb == ab, "the generated file is not the artifact's field"
        assert g.count == a.count == 1
        assert g.width == a.width and g.height == a.height
        assert g.dtypes == a.dtypes == ("float32",)
        assert g.crs == a.crs and g.crs.to_epsg() == 32611
        assert [float(x) for x in g.transform] == [float(x) for x in a.transform]
        assert g.res == (100.0, 100.0)
        assert str(g.nodata) == "nan"
    assert report["bytes"] == tif.stat().st_size
    assert report["field"]["one_px"] == json.loads(META.read_text())["field"]["one_px"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not in this checkout")
def test_repository_validator_passes_on_the_generated_bytes(gen_dir):
    """Not a proxy: this is the same gate the workflows gate on, run on the browser route's output."""
    _, tif, _, _ = gen_dir
    if not (ROOT / "data/training_features.tif").exists():
        pytest.skip("data/ is not placed in this checkout; the validator needs the training grid")
    p = subprocess.run([_py(), "scripts/validate_submission.py", "--pred", str(tif),
                        "--sample", str(SAMPLE), "--train", "data/training_features.tif"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    assert "Validation PASSED - Ready for submission!" in p.stdout
    assert "✗" not in p.stdout


def test_generated_zip_holds_exactly_that_tif(gen_dir):
    """The platform's sentence is "a .zip file containing a single GeoTIFF" - measured against it."""
    _, tif, zip_, _ = gen_dir
    with zipfile.ZipFile(zip_) as z:
        assert z.testzip() is None, "a corrupt member in the archive"
        assert z.namelist() == ["submission.tif"], z.namelist()
        info = z.getinfo("submission.tif")
        assert info.compress_type == zipfile.ZIP_STORED, "the page writes a stored zip, and says so"
        assert z.read("submission.tif") == tif.read_bytes()


@pytest.mark.parametrize("extra,label", [
    (["--no-deflate"], "uncompressed strips"),
    (["--rows-per-strip", "1"], "one row per strip"),
    (["--rows-per-strip", "3730"], "one strip for the whole grid"),
    (["--rows-per-strip", "7"], "a strip height that is not a divisor"),
], ids=str)
@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not in this checkout")
def test_every_container_variant_is_readable_and_identical(extra, label, tmp_path):
    node = _node()
    out = tmp_path / f"{label.replace(' ', '_')}.tif"
    p = subprocess.run([node, str(WRITER), str(META), str(BLOB), str(out)] + extra,
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1200:]
    rep = json.loads(p.stdout)
    assert rep["ok"], rep["failed"]
    with rasterio.open(out) as g, rasterio.open(ARTIFACT) as a:
        assert (np.ascontiguousarray(g.read(1), dtype="<f4").tobytes()
                == np.ascontiguousarray(a.read(1), dtype="<f4").tobytes())
        assert g.crs.to_epsg() == 32611
        # the file must be readable with the strip geometry it declares, not by luck
        expect_strips = -(-g.height // g.profile["blockysize"])
        assert g.profile["tiled"] is False
        assert expect_strips == rep["layout"]["nStrips"], "strip count disagrees with RowsPerStrip"


def test_writer_refuses_a_payload_it_cannot_honour(tmp_path):
    """A stale/foreign manifest must stop the build, not produce a plausible-looking file."""
    node = _node()
    meta = json.loads(META.read_text())
    meta["encoding"]["format"] = "gems-rle-v9"
    bad = tmp_path / "meta.json"
    bad.write_text(json.dumps(meta))
    out = tmp_path / "x.tif"
    p = subprocess.run([node, str(WRITER), str(bad), str(BLOB), str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0
    assert "gems-rle-v1" in (p.stderr + p.stdout)
    assert not out.exists()


def test_writer_refuses_a_field_blob_of_the_wrong_length(tmp_path):
    node = _node()
    out = tmp_path / "x.tif"
    short = tmp_path / "field.bin"
    short.write_bytes(BLOB.read_bytes()[:-9])
    p = subprocess.run([node, str(WRITER), str(META), str(short), str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0
    assert "manifest says" in (p.stderr + p.stdout), "the size guard is what refuses, and it must say so"
    assert not out.exists(), "a failed build must not leave a plausible-looking .tif on disk"


def _flip_a_value_code(blob: bytes) -> bytes:
    """Return the blob with one value code changed 1 -> 0, keeping the run structure valid.

    A blind flip in the middle of the stream may land on a *run length*, which makes the decoder
    refuse before the manifest's pins are ever consulted - a refusal, but not the property this
    test exists for.  Parsing the run structure first (uleb128 length, then a code byte) makes the
    tamper land on a value, so the stream still decodes - to a field with a different number of
    pixels at 1.0 - and only the pinned hash / pixel count can catch it.
    """
    out = bytearray(blob)
    i = 0
    while i < len(out):
        while i < len(out) and out[i] & 0x80:
            i += 1
        i += 1                                  # the terminating byte of the run length
        if i >= len(out):
            break
        if out[i] == 1:                         # a value code: 1 = float32 1.0
            out[i] = 0                          # the same run now carries zeros
            return bytes(out)
        i += 1
    raise AssertionError("no value code found in the blob")


def test_writer_refuses_a_same_length_tamper_via_the_pinned_hash(tmp_path):
    """Changing one value code keeps the size and the run structure, so only the sha256 pin and the
    pixel-count check can catch it - which is why the manifest carries both."""
    node = _node()
    blob = tmp_path / "field.bin"
    blob.write_bytes(_flip_a_value_code(BLOB.read_bytes()))
    out = tmp_path / "x.tif"
    p = subprocess.run([node, str(WRITER), str(META), str(blob), str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0
    combined = p.stderr + p.stdout
    assert "REFUSED" in combined or "self-check failed" in combined
    assert not out.exists()


# --------------------------------------------------------------------- the glue (DOM shim)
@pytest.mark.parametrize("mode", ["tif", "zip"])
def test_page_glue_builds_the_file_it_claims(mode):
    node = _node()
    p = subprocess.run([node, str(HARNESS), str(ROOT / "docs"), mode], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.stdout[-1500:] or "") + (p.stderr[-1500:] or "")
    out = json.loads(p.stdout)
    assert out["ok"], [a for a in out["assertions"] if not a["ok"]]
    assert out["generated"]["bytes"] > 100_000, "a stub file would pass a 'download exists' test"


def test_page_glue_refuses_a_tampered_manifest(tmp_path):
    """The security property of the panel: a payload whose pins do not match is refused loudly and
    never handed to the reader as a submission."""
    node = _node()
    for f in (WRITER, GLUE, BLOB):
        shutil.copy(f, tmp_path / f.name)
    meta = json.loads(META.read_text())
    meta["field"]["one_px"] += 1
    meta["blob"]["sha256"] = "0" * 64
    (tmp_path / "submission_meta.json").write_text(json.dumps(meta))
    p = subprocess.run([node, str(HARNESS), str(tmp_path), "broken"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.stdout[-1200:] or "") + (p.stderr[-1200:] or "")
    out = json.loads(p.stdout)
    assert out["ok"], out["assertions"]


def test_without_placed_data_the_judge_names_the_gap_not_a_keyerror(tmp_path, monkeypatch):
    """A fresh clone has no data/: the sample template is absent, so the container-vs-template
    comparison CANNOT be made.  The judge must report that loudly (FAIL + the placement command),
    not crash into a KeyError — a traceback that hides the real gap is exactly the failure mode
    the judge exists to prevent (session 24: run on a checkout without data/)."""
    _node()  # skip honestly if node is not on PATH (the judge cannot run without it)
    spec = importlib.util.spec_from_file_location("check_site_generator",
                                                 ROOT / "scripts" / "check_site_generator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SAMPLE", "data/definitely_not_placed.tif")
    out = tmp_path / "report.json"
    rc = mod.main(["--out", str(out)])
    assert rc == 1, "a FAIL verdict must exit 1 (2 = missing prerequisites, 0 = pass)"
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "FAIL"
    by_name = {s["name"]: s for s in rep["steps"]}
    for step_name in ("rasterio reads it; pixels are the artifact's",
                      "variant: uncompressed strips"):
        s = by_name[step_name]
        assert s["status"] == "FAIL"
        err = str(s.get("measured", {}).get("error", ""))
        assert "not placed in this checkout" in err and "assemble_data_bridge" in err, \
            f"{step_name}: the gap must be named, got {err!r}"
        assert "KeyError" not in err, f"{step_name}: a KeyError hides the real gap"
    # the pixel-level judgment is unaffected by the missing template and must still have run
    s3 = by_name["rasterio reads it; pixels are the artifact's"]
    assert s3["measured"].get("float32_bits_identical") is True, \
        "pixels are the artifact's even when the template is absent; that check must still report"


def test_generator_evidence_verdict_is_pass_when_it_exists():
    """data/evidence/site_generator.json is what the page renders; a FAIL verdict there must break CI."""
    ev = ROOT / "data/evidence/site_generator.json"
    if not ev.exists():
        pytest.skip("the generator evidence has not been produced in this checkout "
                    "(python scripts/check_site_generator.py)")
    d = json.loads(ev.read_text())
    assert d["verdict"] == "PASS", [s for s in d["steps"] if s["status"] == "FAIL"]
    assert any(s["name"].startswith("rasterio reads") for s in d["steps"])
    assert not any(s["name"].startswith("variant") and s["status"] == "FAIL" for s in d["steps"])


def test_the_files_the_page_loads_are_the_files_the_tests_run():
    """One code path, not two: the page's <script> src list must resolve to files that exist here."""
    assert WRITER.exists() and GLUE.exists() and HARNESS.exists()
    page = ROOT / "docs" / "how_to_submit.html"
    if not page.exists():
        pytest.skip("the site has not been generated in this checkout")
    html = page.read_text()
    assert 'src="geotiff_writer.js"' in html, "the page no longer loads the writer it advertises"
    assert 'src="generate_submission.js"' in html
    assert 'id="tif-generator"' in html
    meta_html = ROOT / "docs" / "submission_meta.json"
    assert meta_html.exists() and BLOB.exists()


def test_every_page_that_ships_the_panel_loads_one_mount_and_the_writer():
    """Session 24: the builder is no longer only on the subpage — the landing page (index) and the
    executive summary carry the same panel, so a visitor can download the file to submit from the
    first page they open.  Each of those pages must therefore load the SAME two scripts the tests
    execute, with exactly ONE mount (the glue binds a single #tif-generator) and the writer before
    the deferred glue."""
    if not (ROOT / "docs" / "index.html").exists():
        pytest.skip("the site has not been generated in this checkout")
    for name in ("index.html", "executive_summary.html", "how_to_submit.html"):
        html = (ROOT / "docs" / name).read_text()
        assert html.count('id="tif-generator"') == 1, f"{name}: the glue binds one mount"
        assert 'src="geotiff_writer.js"' in html, f"{name}: missing the writer it advertises"
        assert 'src="generate_submission.js"' in html, f"{name}: missing the glue"
        assert html.index('<script src="geotiff_writer.js">') \
            < html.index('<script src="generate_submission.js" defer>'), \
            f"{name}: writer must load before the deferred glue"


def test_every_repository_path_the_manifest_names_actually_exists():
    """The manifest is the page's only source of facts, and it names the scripts that made it. When
    one of those files was renamed, the pointer kept pointing at the old name in the shipped payload
    and on the live page (`tests/test_submission_generator.py`, which never existed) — a citation the
    reader cannot follow is worse than no citation. So: walk the manifest, and check the paths."""
    meta = json.loads(META.read_text())
    text = json.dumps(meta)
    cited = set(re.findall(r"[\w./-]+\.(?:json|yaml|yml|html|py|js|md|bin|csv|tif|txt|pdf)", text))
    assert cited, "the manifest cites no files at all? that is also a bug"
    missing = sorted(rel for rel in cited
                     if not rel.startswith(("http", "folds", "model_")) and not (ROOT / rel).exists())
    assert not missing, f"the manifest points at files that are not in the repo: {missing}"


def test_the_writer_js_cites_tests_that_exist():
    """Same rule for the comments in the shipped JS: a reader follows them."""
    js = (ROOT / "docs" / "geotiff_writer.js").read_text() + (ROOT / "docs" / "generate_submission.js").read_text()
    cited = set(re.findall(r"tests/[\w./-]+\.py", js))
    missing = sorted(rel for rel in cited if not (ROOT / rel).exists())
    assert not missing, f"the shipped JS cites non-existent tests: {missing}"
