"""Regression tests for the generated site (2026-09-16, session 6).

The site is generated FROM the evidence JSON; these tests pin the rendering contract so a
schema drift fails loudly instead of shipping a false statement.  The case that motivated
them: scripts/verify_rules_quotes.py reshaped its report from `verification`/`document` to
`summary`/`source`, and build_site._scoring_universe kept reading the old keys — so the
live metric page printed "Quotation verification is INCOMPLETE (0/0)" next to a table of 19
verified quotes, contradicting the overview page that reads the same file correctly.

Stdlib + numpy-free: build_site.py imports only datetime/html/json/pathlib.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _site_mod():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evidence_quotes():
    return json.loads((ROOT / "data/evidence/rules_quotes.json").read_text())


def test_scoring_universe_badge_matches_committed_evidence():
    """The badge must quote the committed report's own counts — never a false 0/0 — and must
    link to the publisher, never to a local extraction path."""
    mod = _site_mod()
    rq = _evidence_quotes()
    html = mod._scoring_universe({"rules_quotes": rq})
    s = rq["summary"]
    assert f"{s['n_found']}/{s['n_quotes']}" in html, "badge does not quote the evidence counts"
    # The renderer's contract, stated state-independently: WARN exactly when the evidence says the
    # check is incomplete.  (Asserting "never warn" would make this test fail whenever a quote is
    # genuinely wrong - it would police the evidence, not the renderer, and the evidence is policed
    # by the Verify-workflow job that produces it.)
    if s["all_found"]:
        assert "INCOMPLETE" not in html, "false INCOMPLETE warning on a fully verified report"
    else:
        assert f"INCOMPLETE ({s['n_found']}/{s['n_quotes']})" in html, \
            "missing quotes must be visible on the page"
    assert "/tmp/" not in html, "a local extraction path leaked into a published link"
    assert ("docs.nlr.gov" in html) or ("www.nlr.gov" in html), "no rules-PDF link rendered"
    for qid in ("phase1_target", "phase2_target", "labels_source", "ranking_basis"):
        assert qid in html, f"key quote {qid} missing from the rendered table"


def test_scoring_universe_warns_only_when_quotes_are_actually_missing():
    """The INCOMPLETE warning must survive for genuinely incomplete reports — the session-6
    fix must not have been to delete the warning, but to read the right keys."""
    mod = _site_mod()
    rq = _evidence_quotes()
    rq["quotes"][0]["exact_match"] = False
    rq["summary"] = dict(rq["summary"], n_found=rq["summary"]["n_found"] - 1, all_found=False)
    html = mod._scoring_universe({"rules_quotes": rq})
    n = rq["summary"]
    assert f"INCOMPLETE ({n['n_found']}/{n['n_quotes']})" in html


def test_scoring_universe_without_evidence_says_so():
    """No evidence file -> an explicit 'not verified' note, not a wolf-crying 0/0."""
    mod = _site_mod()
    html = mod._scoring_universe({})
    assert "not been machine-verified" in html
    assert "INCOMPLETE" not in html


def _proxy_ev():
    import json as _json
    root = ROOT / "data/evidence/proxy"
    def _l(name):
        p = root / name
        return _json.loads(p.read_text()) if p.exists() else None
    return {"proxy_stats": _l("proxy_stats.json"),
            "proxy_fetch": _l("fetch_meta.json"),
            "proxy_eval": _l("eval_reblend_submission.json") or _l("eval_submission.json"),
            "proxy_eval_combined": _l("eval_submission_combined.json"),
            "proxy_sweep": _l("eval_sweep.json")}


def test_proxy_section_renders_the_evidence_not_a_claim():
    """The proxy section is the site's only number taken on a population like the scored one.

    It must quote the committed evidence (feature count, split of the rasterised catalogue, the
    submission's DTI and the catalogue-copy baseline), link the publisher, and state the caveat -
    a page that showed the number without the caveat would be exactly the overclaiming this
    repository is written against.
    """
    ev = _proxy_ev()
    if not ev["proxy_stats"]:
        import pytest
        pytest.skip("proxy evidence not built in this checkout")
    mod = _site_mod()
    html = mod._proxy_catalogue(ev)
    ps = ev["proxy_stats"]
    assert f"{ps['proxy']['proxy_only_px']:,}" in html, "proxy-only pixel count missing"
    assert f"{ps['proxy']['mask_px']:,}" in html, "rasterised count missing"
    assert f"{(ev['proxy_fetch'] or {}).get('result', {}).get('features', 0):,}" in html, \
        "fetched-feature count missing"
    assert "doi.org/10.3133/ds1052" in html, "the SGMC publication link is not rendered"
    assert "catalogue" in html and "absent from the training labels" in html
    if ev["proxy_eval"]:
        d = ev["proxy_eval"]["results"]["as_submitted"]["dti"]
        assert f"{d:.4f}" in html, "the submission's proxy DTI is not rendered"
        cc = ev["proxy_eval"]["results"]["baselines"]["catalogue_copy"]["dti"]
        assert f"{cc:.4f}" in html, "the catalogue-copy baseline is not rendered"
    assert "not</em> the competition metric" in html, "the interpretation caveat is missing"
    assert len(html) > 2000


def test_proxy_section_without_evidence_says_so():
    mod = _site_mod()
    html = mod._proxy_catalogue({})
    assert "not available" in html.lower()
    assert "0.0247" not in html and "0.0" not in html.split("</h2>")[0], \
        "a missing proxy must not render numbers"


# --------------------------------------------------------------------------- verification page
# The page added on 2026-09-16 (session 7) carries the only externally-sourced number on the site.
# Its failure mode is worse than a missing page: a leaderboard rendered from a half-transcribed
# snapshot, or a leaderboard shown WITHOUT the sentence saying this repository is not on it, would
# read as a result.  These two tests pin both halves.
def _verification_ev():
    p = ROOT / "data/evidence/independent_verification.json"
    if not p.exists():
        return {}
    return {"independent_verification": json.loads(p.read_text())}


def test_verification_page_renders_the_leaderboard_and_the_disclaimer():
    ev = _verification_ev()
    if not ev:
        import pytest
        pytest.skip("independent verification record not present in this checkout")
    mod = _site_mod()
    html = mod.build_verification(ev)
    lb = ev["independent_verification"]["competition_standing"]
    assert f"{lb['top_dti']:.4f}" in html, "the top score is not rendered"
    for row in lb["rows"]:
        assert f"{row['best_public_dti']:.4f}" in html, f"rank {row['rank']} missing"
        assert row["participant"] in html, f"rank {row['rank']} entrant missing"
    assert f"{lb['n_ranked']} ranked entrants" in html, "the ranked-entrant count is not rendered"
    assert "This repository is not on it." in html, "the disclaimer is missing"
    assert lb["url"] in html, "the leaderboard link is missing"
    # every check row must appear with its URL: an empty row is a finding, not a silent omission
    for c in ev["independent_verification"]["checks"]:
        assert c["url"] in html, f"check {c['id']} lost its source"
    assert len(html) > 4000


def test_verification_page_without_evidence_says_so_and_invents_nothing():
    mod = _site_mod()
    html = mod.build_verification({})
    assert "not available" in html.lower()
    assert "0.1972" not in html and "mzoorob" not in html, \
        "a missing verification record must not render a leaderboard"


# --------------------------------------------------------------------------- executive summary page
def test_executive_summary_page_renders_all_required_sections():
    mod = _site_mod()
    ev = {
        "rules_quotes": _evidence_quotes(),
        "emission_decision": json.loads((ROOT / "data/evidence/emission_decision.json").read_text()),
        "independent_verification": json.loads((ROOT / "data/evidence/independent_verification.json").read_text()),
        "inventory": json.loads((ROOT / "data/evidence/inventory.json").read_text()),
    }
    html = mod.build_executive_summary(ev)
    assert "Executive Summary" in html
    assert "Dual-Phase Prize Structure" in html
    assert "$300,000" in html
    assert "$50,000" in html and "$250,000" in html
    assert "Generative AI" in html
    assert "EPSG:32611" in html
    assert "100.0 m" in html or "100m" in html or "100 m" in html
    assert "Distance-Weighted Tversky" in html
    assert "Floor 0.1" in html
    assert "validate_submission.py" in html
    assert "DrivenData" in html
    assert ("docs.nlr.gov" in html) or ("www.nlr.gov" in html)
    assert "https://www.herox.com/GEMSPrize/resource/2274" in html
    assert len(html) > 10000


def test_executive_summary_without_evidence_handles_gracefully():
    mod = _site_mod()
    html = mod.build_executive_summary({})
    assert "Executive Summary" in html
    assert "EPSG:32611" in html
    assert "$300,000" in html
    assert "validate_submission.py" in html


# --------------------------------------------------------------------------- submission builder
# The 2026-09-24 (session 24) ask: the site must be able to generate the TIF required for
# submission, as easy as clicking a file to download, and it must be obvious when you visit the
# site.  That is the #tif-generator panel (docs/generate_submission.js) lifted out of
# how_to_submit.html onto the two pages a reader lands on: the overview (docs/index.html) and
# the executive summary.  These tests pin the PLACEMENT (it leads the page, it is the only mount,
# the scripts it needs are loaded) and the HONESTY (no payload -> no live-looking panel).
def _submission_ev():
    def load(rel):
        p = ROOT / rel
        return json.loads(p.read_text()) if p.exists() else None
    return {"site_payload": load("docs/submission_meta.json"),
            "site_generator": load("data/evidence/site_generator.json"),
            "rules_quotes": load("data/evidence/rules_quotes.json"),
            "independent_verification": load("data/evidence/independent_verification.json"),
            "inventory": load("data/evidence/inventory.json")}


def test_index_opens_with_the_submission_builder():
    mod = _site_mod()
    html = mod.build_index(_submission_ev())
    assert html.count('id="tif-generator"') == 1, "the glue binds ONE mount; a second would print nothing"
    assert html.index('<script src="geotiff_writer.js">') < html.index('<script src="generate_submission.js" defer>'), \
        "writer must load before the deferred glue"
    # it is the first h2 of the page: a visitor sees the builder before the analysis
    body = html.split("</header>", 1)[1]
    first_h2 = body.index("<h2")
    assert "generate-here" in body[first_h2:first_h2 + 80], "the builder no longer leads the overview"
    assert body.index("build-submission") < body.index("<h2>The task</h2>")
    # the grid card points at the hero instead of describing it
    assert 'href="#build-submission"' in html


def test_executive_summary_carries_the_builder_before_the_sections():
    mod = _site_mod()
    html = mod.build_executive_summary(_submission_ev())
    assert html.count('id="tif-generator"') == 1
    assert 'src="geotiff_writer.js"' in html and 'src="generate_submission.js" defer' in html
    assert html.index("build-submission") < html.index("<h2>1. Executive Overview")
    # the TL;DR now points at the in-page generator, not only at the subpage
    assert 'href="#generate-here"' in html


def test_builder_without_payload_shows_the_gap_not_a_panel():
    """No payload -> no mount and no script: a dead panel that prints nothing on click would be
    worse than a visible gap with the command that produces the payload."""
    mod = _site_mod()
    for fn in (mod.build_index, mod.build_executive_summary, mod.build_how_to_submit):
        html = fn({})
        assert 'id="tif-generator"' not in html, f"{fn.__name__}: a mount without a payload"
        assert 'src="generate_submission.js"' not in html, f"{fn.__name__}: glue with no panel"
        assert "Not available" in html, f"{fn.__name__}: the gap must be visible"


def test_builder_numbers_come_from_the_payload_not_the_template():
    """Every figure the hero prints must be derivable from docs/submission_meta.json: tamper the
    payload and the rendered page must change with it (grid, run count, blob size, pins)."""
    mod = _site_mod()
    ev = _submission_ev()
    html = mod.build_index(ev)
    meta = json.loads((ROOT / "docs" / "submission_meta.json").read_text())
    for token in (f'{int(meta["grid"]["width"]):,} &times; {int(meta["grid"]["height"]):,} px',
                  f'{int(meta["encoding"]["n_runs"]):,} runs',
                  f'{int(meta["blob"]["bytes"]):,} B',
                  meta["blob"]["sha256"][:16],
                  meta["artifact"]["sha256"][:16]):
        assert token in html, f"hero does not render the measured value {token!r}"


def test_nav_bar_links_to_executive_summary():
    mod = _site_mod()
    html = mod.page("Test", "test.html", "<p>content</p>")
    assert 'href="executive_summary.html"' in html
    assert "Executive summary" in html

