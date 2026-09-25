"""docs/submission.html: the operational "how do I submit" page must quote, never assert.

The page is the one place a reader goes to act - enroll, place the data, pick a raster, validate,
upload, select a final submission.  A wrong number there is worse than a missing page, so every
fact on it has to come from somewhere checkable:

* the artifact's sha256 and byte size are RE-HASHED from the committed bytes at build time;
* the data-placement table compares the pins in data/bridge/manifest.json with whatever data/
  holds right now, so the page cannot claim a placement this checkout does not have;
* the validator table is the committed validator log, parsed;
* the rules sentences are quoted by id from data/evidence/rules_quotes.json with their own
  verification badge, and a missing quote renders as "not verified" instead of a paraphrase;
* the measurement rows come from the committed evidence JSON, with the surrogate caveat attached;
* the catalogue row count is read from docs/data_catalog.csv (a typed count drifts - it already
  had: the prose said 89 rows while the audit counted 92).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


SHIPPED = str(_shipped_artifact().relative_to(ROOT))
QUOTE_IDS = ("entry", "citizen", "weekly_limit", "one_final_submission", "no_private_knowledge",
             "phase2_target", "ai_disclosure", "code_assets")


def _site_mod():
    spec = importlib.util.spec_from_file_location("build_site_sub", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def _full_ev():
    return {
        "rules_quotes": _load("data/evidence/rules_quotes.json"),
        "emission_decision": _load("data/evidence/emission_decision.json"),
        "independent_verification": _load("data/evidence/independent_verification.json"),
        "placement": _load("data/evidence/data_placement.json"),
        "rasters": _load("data/evidence/rasters.json"),
        "block_stratified": _load("data/evidence/block_holdout/block_stratified.json"),
        "fold_gaps": _load("data/evidence/block_holdout/fold_gap_summary.json"),
        "pseudo_contrast": _load("data/evidence/pseudo_labels/fold0_two_population_contrast.json"),
        "combined_truth": _load("data/evidence/proxy/combined_truth_shipped.json"),
        "field_selection": _load("data/evidence/field_selection.json"),
    }


# -------------------------------------------------------------------------------------- helpers
def test_sha_bytes_hashes_the_committed_artifact(tmp_path, monkeypatch):
    """The checksum on the page is measured, not typed: prove the helper really hashes."""
    mod = _site_mod()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    f = tmp_path / "x.tif"
    f.write_bytes(b"the same bytes every time")
    got = mod._sha_bytes("x.tif")
    assert got["exists"] is True
    assert got["sha256"] == hashlib.sha256(b"the same bytes every time").hexdigest()
    assert got["bytes"] == len(b"the same bytes every time")
    assert mod._sha_bytes("absent.tif") == dict(exists=False, path="absent.tif")


def test_validation_log_parsing_reads_the_pass_line_and_every_check():
    mod = _site_mod()
    log = ("Validating submission.tif\n  Width: 3292\n  ✓ CRS EPSG:32611\n"
           "  ✓ Values in [0,1] (min 0.0000 max 1.0000)\n"
           "  i NaN fraction 57.93% (expected)\n\n✅ Validation PASSED - Ready for submission!\n")
    rows, passed = mod._validation_rows(log)
    assert passed is True
    assert rows == [("✓", "CRS EPSG:32611"), ("✓", "Values in [0,1] (min 0.0000 max 1.0000)"),
                    ("i", "NaN fraction 57.93% (expected)")]
    assert mod._validation_rows("  x Dtype float64 != float32\nFAILED")[1] is False
    assert mod._validation_rows("") == ([], False)
    assert mod._validation_rows(None) == ([], False)


def test_quote_block_prefers_an_explicit_gap_to_a_paraphrase():
    mod = _site_mod()
    rq = dict(generated_utc="2026-09-19T00:00:00Z",
              source=dict(sha256="50d854b1e0239fe6b9648d9fa5c7537bc7b6e5bc10cf6b37a2ff9aa401c36938"),
              quotes=[dict(id="weekly_limit", section="§3.2", quote="three submissions per week",
                           why="quota", exact_match=True),
                      dict(id="broken", section="§9", quote="something", why="x",
                           exact_match=False)])
    ok = mod.quote_block({"rules_quotes": rq}, "weekly_limit", "how many")
    assert "three submissions per week" in ok and "VERBATIM MATCH" in ok and "§3.2" in ok
    assert "docs.nlr.gov/docs/fy26osti/96647.pdf" in ok
    bad = mod.quote_block({"rules_quotes": rq}, "broken", "how many")
    assert "NOT FOUND IN THE PDF" in bad
    gone = mod.quote_block({}, "weekly_limit", "how many")
    assert "not in the committed rules evidence" in gone
    assert "three submissions per week" not in gone, "a missing quote must not be reconstructed"


def test_catalog_row_count_is_read_not_typed(tmp_path, monkeypatch):
    mod = _site_mod()
    monkeypatch.setattr(mod, "DOCS", tmp_path)
    assert mod._catalog_rows() is None and mod._catalog_rows_text() == ""
    (tmp_path / "data_catalog.csv").write_text(
        'id,name,official_link\nE1,x,"https://example.org/a"\nE2,y,"https://example.org/b"\n')
    assert mod._catalog_rows() == 2
    assert mod._catalog_rows_text() == "2 rows, "


def test_lead_upper_does_not_lowercase_the_rest_of_the_sentence():
    mod = _site_mod()
    assert mod._lead_upper("no decision. A pseudo-labelled FIELD ships only via R3.") == \
        "No decision. A pseudo-labelled FIELD ships only via R3."
    assert mod._lead_upper("") == ""


# -------------------------------------------------------------------------------------- the page
def test_the_page_is_in_the_nav_of_every_page():
    """`submission.html` was retitled "Submission details" when the subpage was added; the nav
    entry must exist under its new label, and the subpage must be reachable from every page."""
    mod = _site_mod()
    html = mod.page("Anywhere", "index.html", "<p>x</p>")
    assert 'href="submission.html"' in html and "Submission details" in html
    assert 'href="how_to_submit.html"' in html and "How to submit" in html
    active = mod.page("Submission details", "submission.html", "<p>x</p>")
    assert '<a href="submission.html" class="active">Submission details</a>' in active
    sub = mod.page("How to submit, exactly", "how_to_submit.html", "<p>x</p>")
    assert '<a href="how_to_submit.html" class="active">' in sub


@pytest.mark.skipif(not (ROOT / SHIPPED).exists(), reason="the shipped artifact is not committed")
def test_the_page_hashes_the_artifact_it_tells_you_to_upload():
    mod = _site_mod()
    ev = _full_ev()
    html = mod.build_submission(ev)
    digest = hashlib.sha256((ROOT / SHIPPED).read_bytes()).hexdigest()
    size = (ROOT / SHIPPED).stat().st_size
    assert digest in html, "the full sha256 of the artifact must be on the page"
    assert f"{size:,} bytes" in html, "the exact byte size must be on the page"
    assert "556.2 KB" not in html, "a binary-KB rounding that contradicts the docs must not appear"
    assert SHIPPED in html


def test_the_page_renders_the_committed_validator_output_and_the_rules_quotes():
    mod = _site_mod()
    ev = _full_ev()
    if not ev["rules_quotes"]:
        pytest.skip("rules evidence not in this checkout")
    html = mod.build_submission(ev)
    log = (ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/validation.log")
    if log.exists():
        rows, passed = mod._validation_rows(log.read_text())
        assert passed is True, "the committed log must be a PASS log for this artifact"
        for state, text in rows:
            # the page renders rows through e() (html.escape), so the raw log text with an
            # apostrophe or quote only appears in its escaped form
            assert mod.e(text) in html, f"validator check {text!r} is not rendered"
        assert "Validation PASSED" in html
    by_id = {q["id"]: q for q in ev["rules_quotes"]["quotes"]}
    for qid in QUOTE_IDS:
        assert qid in by_id, f"the page needs quote id {qid} to exist in the evidence"
        assert by_id[qid]["quote"] in html, f"the verbatim {qid} sentence is not on the page"
    assert html.count("VERBATIM MATCH") >= len(QUOTE_IDS)


def test_the_page_states_the_format_requirements_and_the_official_sources():
    mod = _site_mod()
    html = mod.build_submission(_full_ev())
    for token in ("EPSG:32611", "100 m", "float32", "3292", "3730", "single-band",
                  "page/967/#submission-format", "Make new submission",
                  "https://docs.nlr.gov/docs/fy26osti/96647.pdf",
                  "https://www.herox.com/GEMSPrize/resource/2274",
                  "scripts/validate_submission.py", "scripts/assemble_data_bridge.py",
                  "scripts/prepare_data.py", "Dec 3, 2026"):
        assert token in html, f"missing from the submission page: {token}"


def test_the_placement_table_reports_this_checkout_not_a_claim(tmp_path, monkeypatch):
    """If data/ is empty the page must say ABSENT; if the bytes match the pins it must say so."""
    mod = _site_mod()
    ev = _full_ev()
    html = mod.build_submission(ev)
    manifest = _load("data/bridge/manifest.json")
    pinned = {f["canonical"]: f["sha256"] for f in (manifest or {}).get("files", []) if f.get("canonical")}
    for canonical, sha in pinned.items():
        present = (ROOT / f"data/{canonical}").exists()
        if present and hashlib.sha256((ROOT / f"data/{canonical}").read_bytes()).hexdigest() == sha:
            assert sha[:24] in html
        elif not present:
            assert "ABSENT" in html, "an unpopulated data/ must be visible on the page"


def test_the_measurement_rows_come_from_the_committed_evidence():
    mod = _site_mod()
    ev = _full_ev()
    html = mod.build_submission(ev)
    bs = ev.get("block_stratified") or {}
    ref = (bs.get("verdict") or {})
    if ref.get("reference_proxy_dti") is not None:
        assert f"{ref['reference_proxy_dti']:.4f}" in html
    fg = ev.get("fold_gaps") or {}
    if (fg.get("gap") or {}).get("mean") is not None:
        assert f"{fg['gap']['mean']:+.4f}" in html
        assert f"{fg['n_folds_committed']} fold" in html
    pc = ev.get("pseudo_contrast") or {}
    held = ((pc.get("arms") or {}).get("heldout") or {}).get("populations") or {}
    for pop in ("proxy_only", "labels"):
        arm = held.get(pop) or {}
        if arm.get("status") == "MEASURED":
            assert f"{arm['candidate_dti']:.4f}" in html, f"the {pop} arm is not on the page"
    if pc.get("verdict"):
        assert pc["verdict"] in html
        assert mod._lead_upper(pc["shipping_note"]).split(".")[0] in html
    ct = ((ev.get("combined_truth") or {}).get("combined_population") or {})
    if (ct.get("reference") or {}).get("dti") is not None:
        assert f"{ct['reference']['dti']:.4f}" in html
        assert "surrogate" in html.lower()
    iv = ev.get("independent_verification") or {}
    if (iv.get("competition_standing") or {}).get("top_dti") is not None:
        assert str(iv["competition_standing"]["top_dti"]) in html
        assert "NOT on that leaderboard" in html
    assert "None of them is an official score" in html


def test_without_evidence_the_page_invents_nothing():
    mod = _site_mod()
    html = mod.build_submission({})
    assert "Make a submission" in html
    # the artifact hash is MEASURED from the committed bytes, so it is rendered even with no
    # evidence dict at all - but it must equal a fresh hash, never a typed constant
    if (ROOT / SHIPPED).exists():
        assert hashlib.sha256((ROOT / SHIPPED).read_bytes()).hexdigest() in html
    assert "0.1972" not in html and "0.2074" not in html and "0.0999" not in html
    assert "rank 1 of ?" not in html, "an unknown rank must not render as a placeholder"
    assert "NOT COMMITTED" in html or "not measured" in html
    assert "not in the committed rules evidence" in html, "missing quotes must be declared"
    assert "Not available" in html or "NOT PRESENT" in html
    # the procedure itself does not depend on evidence: the commands must still be there
    for token in ("scripts/assemble_data_bridge.py", "scripts/prepare_data.py",
                  "scripts/validate_submission.py", "Make new submission"):
        assert token in html


def test_the_page_links_only_to_pages_that_exist_and_never_to_raw_markdown():
    """scripts/audit_docs.py::check_published_links enforces this for the deployed site; pin it here
    too so the failure happens at build time rather than in the Pages gate."""
    mod = _site_mod()
    import re
    html = mod.page("Make a submission", "submission.html", mod.build_submission(_full_ev()))
    for href in set(re.findall(r'href="([^"#]+?)(?:#[^"]*)?"', html)):
        if href.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        assert not href.startswith(("../", "/")), f"{href} leaves the published root"
        assert not href.endswith(".md"), f"{href} points at raw markdown inside the site"
        assert (mod.DOCS / href).exists(), f"{href} would 404 on the deployed site"
