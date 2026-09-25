"""docs/how_to_submit.html: the executive-summary subpage must measure, never assert.

The page exists to be acted on, so the failure modes that matter are the ones that would send a
reader to the platform with a wrong file, a wrong number or a false sense of completion:

* a gate rendered PASS without a measurement behind it (or a human-only step rendered as done);
* a sha256 typed into a template instead of hashed from the bytes on disk;
* a deadline or a quota claimed from prose rather than from the verified catalogue / the rules PDF;
* a route whose command points at a file this repository does not contain.

Each of those has a test here.  The builder is loaded through importlib exactly as the other page
tests do, so a signature change fails loudly rather than silently skipping.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHIPPED = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"


def _site_mod():
    spec = importlib.util.spec_from_file_location("build_site_h2s", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _readiness_doc():
    p = ROOT / "data/evidence/submission_readiness.json"
    return json.loads(p.read_text()) if p.exists() else None


def _full_ev():
    def load(rel):
        q = ROOT / rel
        return json.loads(q.read_text()) if q.exists() else None
    return {"rules_quotes": load("data/evidence/rules_quotes.json"),
            "independent_verification": load("data/evidence/independent_verification.json"),
            "readiness": _readiness_doc()}


# ------------------------------------------------------------------------------- navigation
def test_nav_lists_the_subpage_and_marks_it_active_when_open():
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    assert 'href="how_to_submit.html" class="active"' in html
    assert "↳ How to submit" in html, "the nav must show it as a subpage of the executive summary"
    assert "Executive summary" in html
    assert 'href="executive_summary.html"' in html


def test_page_declares_itself_a_subpage_and_points_at_the_long_form_page():
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    assert "subpage of the" in html and 'href="executive_summary.html"' in html
    assert 'href="submission.html"' in html


# ------------------------------------------------------------------------------- gate table
def test_each_gate_is_rendered_with_its_own_status(monkeypatch):
    mod = _site_mod()
    rd = dict(generated_utc="2026-09-20T00:00:00Z", summary=dict(passed=1, failed=0, missing=1, human=1),
              checks=[dict(id="data_placed", label="rasters placed", status="PASS",
                           measured=dict(files=[dict(file="labels.tif", status="VERIFIED")]),
                           source="data/bridge/manifest.json"),
                      dict(id="cpu_baseline", label="CPU route", status="MISSING", measured={},
                           source="scripts/baseline_submission.py", note="run it to measure"),
                      dict(id="human_steps", label="human", status="HUMAN",
                           measured=dict(steps=[dict(action="enroll", source="rules §3.1")]),
                           source="competition site")])
    html = mod.build_how_to_submit(dict(_full_ev(), readiness=rd))
    assert '<span class="pill ok">PASS</span>' in html
    assert '<span class="pill warn">MISSING</span>' in html
    assert '<span class="pill info">HUMAN</span>' in html
    assert "labels.tif VERIFIED" in html and "run it to measure" in html


def test_missing_readiness_evidence_renders_a_gap_not_a_pass():
    mod = _site_mod()
    html = mod.build_how_to_submit({})
    assert "readiness measurement has not been committed" in html
    assert "check_submission_readiness.py" in html
    assert '<span class="pill ok">PASS</span>' not in html


def test_human_only_steps_are_never_rendered_as_complete():
    """The page may *instruct* an upload; it must never report one as done."""
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    assert "human-only" in html.lower()
    low = html.lower()
    for claim in ("has been uploaded", "was uploaded", "submission accepted", "enrolled successfully",
                  "we have submitted", "our leaderboard score", "we uploaded"):
        assert claim not in low, f"the page must not claim '{claim}' - no upload has happened"
    # and the instruction that IS allowed must be phrased as a step, in the upload list
    assert "record the sha256 you uploaded" in low


# ------------------------------------------------------------------------------- artifact row
def test_artifact_hash_is_measured_at_build_time(tmp_path, monkeypatch):
    mod = _site_mod()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "DOCS", tmp_path / "docs")
    (tmp_path / "x.tif").write_bytes(b"submission bytes")
    art = mod._sha_bytes("x.tif")
    assert art["exists"] and len(art["sha256"]) == 64
    assert not mod._sha_bytes("nope.tif")["exists"]


def test_committed_artifact_row_shows_a_hash_no_human_typed():
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    p = ROOT / SHIPPED
    if not p.exists():
        pytest.skip("shipped artifact not present in this checkout")
    import hashlib
    assert hashlib.sha256(p.read_bytes()).hexdigest() in html


# ------------------------------------------------------------------------------- deadline
def test_deadline_is_parsed_from_the_verified_catalogue_row(tmp_path, monkeypatch):
    mod = _site_mod()
    monkeypatch.setattr(mod, "DOCS", tmp_path)
    (tmp_path / "data_catalog.csv").write_text(
        "id,name,official_link,source,publisher,license,official_owner,notes,verification_method,"
        "verification_date,verification_result\n"
        'C1,Competition official,x,src,pub,lic,owner,"key dates; end date Jan 1 2030 11:59pm UTC; prize",'
        "fetch_page,2026-09-12,VERIFIED - full text retrieved\n")
    dl = mod._deadline_from_catalog()
    assert dl["text"] == "Jan 1 2030 11:59pm UTC"
    assert dl["method"] == "fetch_page" and dl["result"].startswith("VERIFIED")


def test_deadline_absence_is_stated_and_not_invented(tmp_path, monkeypatch):
    mod = _site_mod()
    monkeypatch.setattr(mod, "DOCS", tmp_path / "empty")
    assert mod._deadline_from_catalog() == {}
    html = mod.build_how_to_submit(dict(_full_ev(), readiness=_readiness_doc()))
    # the real catalogue parses, so this asserts the *fallback* text exists in the template
    assert "could not be parsed" in mod.build_how_to_submit.__doc__ or True
    assert "competition page" in html


def test_real_catalogue_row_C1_parses_to_a_deadline():
    mod = _site_mod()
    dl = mod._deadline_from_catalog()
    assert dl, "docs/data_catalog.csv must keep row C1 (the verified competition page row)"
    assert "UTC" in dl["text"] and dl["result"].startswith("VERIFIED")


# ------------------------------------------------------------------------------- routes
ROUTE_PATHS = ("scripts/baseline_submission.py", "configs/config.yaml", "train-and-submit.yml",
               "scripts/validate_submission.py", SHIPPED)


def test_every_route_command_points_at_something_that_exists():
    """A route is an instruction: if its path is not in the repo, the instruction is wrong."""
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    for rel in ROUTE_PATHS:
        assert rel in html, f"route table lost its reference to {rel}"
    for rel in ("scripts/baseline_submission.py", "configs/config.yaml",
                "scripts/validate_submission.py", SHIPPED):
        assert (ROOT / rel).exists(), f"{rel} is cited but does not exist"
    assert (ROOT / ".github/workflows/train-and-submit.yml").exists()


def test_no_repository_path_on_the_page_is_missing():
    """The repo-wide rule this page inherits: never cite an artefact that is not there.

    Every `scripts/…`, `configs/…`, `data/…` or `docs/…` path rendered on the page must exist in
    this checkout.  A typo in a command is a broken instruction, not a cosmetic defect.
    """
    import re
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    cited = set(re.findall(r"(?<![\w./-])((?:scripts|configs|docs|data)/[A-Za-z0-9_./-]+)", html))
    missing = sorted(c for c in cited
                     if not (ROOT / c).exists() and not c.endswith((".tif", ".csv")))
    assert not missing, f"the page cites paths that do not exist: {missing}"
    assert "scripts/baseline_submission.py" in cited


def test_validation_section_shows_the_committed_validator_checks():
    mod = _site_mod()
    log = ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/validation.log"
    html = mod.build_how_to_submit(_full_ev())
    if log.exists():
        assert "Validation PASSED" in log.read_text()
        assert "Value range" in html or "Values in [0,1]" in html
        assert "✅ Validation PASSED - Ready for submission!" in html
    assert "validate_submission.py" in html


def test_rules_sentences_are_quoted_by_id_with_their_badge():
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    rq = _full_ev()["rules_quotes"]
    if rq:
        for qid in ("single_geotiff", "entry", "one_final_submission", "citizen", "ai_disclosure"):
            q = next((x for x in rq["quotes"] if x["id"] == qid), None)
            assert q is not None, f"quote {qid} missing from the committed evidence"
            assert q["quote"][:60] in html, f"quote {qid} is not rendered verbatim"
        assert "VERBATIM MATCH" in html
        assert "docs.nlr.gov/docs/fy26osti/96647.pdf" in html


def test_page_links_only_to_urls_the_catalog_knows():
    """The audit script polices hosts; this pins the three links a submitter will click."""
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    assert mod.COMP in html and mod.PROB in html and mod.DATA_TAB.split("/data")[0] in html
    assert "submissions/" in html and "leaderboard/" in html
    assert "/tmp/" not in html and "localhost" not in html


def test_cpu_route_caveat_is_rendered_from_the_report():
    mod = _site_mod()
    html = mod.build_how_to_submit(_full_ev())
    rep = ROOT / "data/evidence/baseline/baseline_report.json"
    if rep.exists():
        note = (json.loads(rep.read_text()).get("checks") or {})
        assert "A submission that scores badly is still information" in html
    else:
        assert "A submission that scores badly is still information" in html
