"""Tests for the rules-quote verifier (scripts/verify_rules_quotes.py).

Why: this script is the machine check behind the single most consequential claim in the repository -
that both prize phases score the expert-mapped NEW fault dataset while the training labels are the
*existing* catalogue (§1.1, §3.3).  Two things must hold for that claim to be trustworthy:

  1. the normalisation must not be so loose that a paraphrase would pass (no fuzzy matching), and
     must not be so strict that the same sentence fails merely because a PDF line break moved;
  2. a MISS must arrive with the document's own words next to the quoted ones, because the job log
     is not readable from the development sandbox (the Actions log host is blocked) - a silent "not
     found" is exactly the kind of unmeasurable claim this repository exists to avoid.

Offline: normalisation and diagnosis are pure string functions; the committed report is read from
disk rather than regenerated (a PDF fetch needs a runner - docs.nlr.gov is not on the sandbox
allowlist).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("vrq", ROOT / "scripts" / "verify_rules_quotes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalisation_folds_typography_but_not_language():
    m = _mod()
    doc = m.norm("The labels\u2019 owner said \u201cyes\u201d \u2014 then left\u00a0the  room.")
    assert doc == "The labels' owner said \"yes\" - then left the room."
    # a hyphen at a line break is a wrapped word, not a hyphen
    assert m.norm("over-\nlap here") == "overlap here"
    # ...but a real hyphen survives
    assert m.norm("100-m resolution") == "100-m resolution"
    # a changed word must NOT match after normalisation (no fuzzy matching anywhere)
    assert "the labels own" not in doc


def test_divergence_points_at_the_exact_disagreement():
    m = _mod()
    text = m.norm("Intro. The set of faults included in the public test dataset and the rest.")
    quote = m.norm("The set of faults included in the public test datasets and the rest.")
    d = m.divergence(quote, text)
    assert d["prefix_len"] == len("The set of faults included in the public test dataset")
    assert d["quote_continues"].startswith("s and the rest")
    assert d["doc_continues"].startswith(" and the rest")
    assert d["doc_offset"] > 0


def test_divergence_on_a_total_miss_is_empty_not_a_crash():
    m = _mod()
    d = m.divergence(m.norm("nothing like this is in the document"), m.norm("other text"))
    assert d["prefix_len"] == 0 and d["doc_continues"] is None


def test_committed_report_matches_the_quote_list():
    """The published evidence must be about the same sentences the script checks.

    If a quote is edited or added without re-running the workflow, the site would render a count
    that no longer corresponds to the list - so the schema agreement is pinned here.  (Whether each
    quote was FOUND is the runner's business; it needs the PDF.)
    """
    m = _mod()
    report = json.loads((ROOT / "data/evidence/rules_quotes.json").read_text())
    ids = [q[0] for q in m.QUOTES]
    assert len(ids) == len(set(ids)), "duplicate quote ids"
    published = {q["id"]: q["quote"] for q in report["quotes"]}
    missing = [i for i in ids if i not in published]
    assert not missing, f"quotes not present in the committed evidence: {missing}"
    mismatched = [i for i, _, q, _ in m.QUOTES if published[i] != q]
    assert not mismatched, f"quote text changed since the evidence was produced: {mismatched}"
    assert report["summary"]["n_quotes"] == len(ids)
    # A miss must carry its diagnosis - but only reports written by a diagnostic-capable run can,
    # and the committed report may predate that feature (the workflow refreshes it).  The marker is
    # `extracted_text`, which the script only writes when it also writes diagnostics.
    diagnostic_capable = report.get("extracted_text") is not None
    for q in report["quotes"]:
        if not q["exact_match"] and diagnostic_capable:
            assert q.get("diagnostic", {}).get("prefix_len") is not None, \
                f"{q['id']} failed without a diagnostic"
    assert report["source"]["canonical_url"] == m.RULES_URL, "evidence must link the publisher"


def test_page_furniture_is_removed_from_page_margins_only():
    """A bare page number at a page boundary is furniture, not prose.

    MEASURED 2026-09-16: the official rules' §3.6.2 sentence continues across a page break, and the
    extractor emitted the next page's number inside it ("... and the 12 relative weight of faults
    ..."), so a genuinely verbatim quotation could not match.  The strip must be exactly as narrow as
    that: bare numbers at the very start or end of a page, nothing inside the text.
    """
    m = _mod()
    assert m.strip_page_furniture("12\nrelative weight of faults\n13", 12) == \
        "relative weight of faults"
    assert m.strip_page_furniture("text continues here\n12", 12) == "text continues here"
    # numbers inside prose are untouched
    inside = "15 U.S.C. 1001 applies to a real sentence spanning 2026 words"
    assert m.strip_page_furniture(inside, 7) == inside
    # a rule/statute number at the start of a page is NOT a page number: the previous page ended a
    # sentence, so the number cannot be a marker glued to a wrapped sentence
    assert m.strip_page_furniture("15 U.S.C. 1001 applies", 15, None,
                                  prev_tail="shall be fined.") == "15 U.S.C. 1001 applies"
    # ...and when the previous page DOES end mid-sentence, a leading number EQUAL TO THE PAGE NUMBER
    # is the page marker pypdf glued to the wrapped sentence (this is the real §3.6.2 break:
    # page 11 ends "... and the", page 12's text begins "12 relative weight of faults ...")
    assert m.strip_page_furniture("12 relative weight of faults", 12, None,
                                  prev_tail="and the") == "relative weight of faults"
    # a leading number that is not this page's number is never touched
    assert m.strip_page_furniture("12 relative weight", 7, None,
                                  prev_tail="and the") == "12 relative weight"


def test_every_quoted_sentence_matches_the_committed_extraction():
    """Offline check that the committed report and the committed text tell the same story.

    The PDF is only reachable from a runner, so this reads the text the workflow committed.  Two
    directions matter, and both are asserted:

      * every quote the report calls a MATCH really is present in the committed text (a report that
        claims a verbatim match must be checkable by a reader with the evidence in hand);
      * every quote it calls a MISS really is absent (so a silent "not found" cannot hide a quote
        that the text does contain - which is exactly what the §3.6.2 page-number defect was).

    A report written before the page-furniture strip cannot be read this way, so it is skipped; the
    workflow refreshes it.
    """
    m = _mod()
    report = json.loads((ROOT / "data/evidence/rules_quotes.json").read_text())
    if not report.get("page_furniture_stripped"):
        import pytest
        pytest.skip("committed extraction predates the page-furniture strip")
    extracted = report.get("extracted_text")
    # verify-sources.yml intentionally records the quote verdict without committing the
    # normalised PDF text; only verify-rules.yml, which receives --dump-text, has it.
    # Treat that as an unavailable optional artifact, not ROOT/"" (which is a directory
    # and used to raise IsADirectoryError in a clean checkout).
    if not extracted or not extracted.get("path"):
        import pytest
        pytest.skip("this evidence report has no committed extracted text")
    text_path = ROOT / extracted["path"]
    if not text_path.is_file():
        import pytest
        pytest.skip("extracted text not committed in this checkout")
    text = text_path.read_text(encoding="utf-8")

    claimed = {q["id"]: bool(q["exact_match"]) for q in report["quotes"]}
    present = {q[0]: (m.norm(q[2]) in text) for q in m.QUOTES}
    wrong = {k: ("claimed match, absent from text" if claimed[k] else
                 "claimed miss, present in text") for k in claimed if claimed[k] != present[k]}
    assert not wrong, f"report and committed text disagree: {wrong}"
    assert report["summary"]["n_found"] == sum(claimed.values()), \
        "the report's own summary contradicts its rows"
    if report["summary"]["all_found"]:
        assert all(present.values()), "all_found is claimed but a quote is absent"
        # ...and if the strip was used, say so: the removals are listed in the report for review
        for r in report.get("page_furniture_removed") or []:
            assert set(r) == {"page", "removed", "why"}, "a removal must be explainable"


# The one quotation the verifier could not match (§3.6.2), replayed in every shape the extractor was
# observed to produce.  Recorded from the committed evidence of 2026-09-16:
#   page 11 raw text ends  "... and the"            (the sentence wraps onto the next page)
#   page 12 raw text begins "12 relative weight ..." or "12\nrelative weight ...", sometimes after a
#   blank first line.  The document is verbatim; only the page number interleaves.
_WRAPPED_QUOTE = ("The set of faults included in the public test dataset and the relative weight of "
                  "faults in both test datasets will be determined by the competition organizers "
                  "before the start of the competition.")
_PAGE11_HEAD = "3.6.2 Interviews DOE, at its sole discretion, may decide to hold a short interview. " \
               "This restriction is in place to encourage models that generalize well to unseen data " \
               "and to discourage overfitting to the public test set. The set of faults included in " \
               "the public test dataset and the"
_PAGE12_BODY = ("relative weight of faults in both test datasets will be determined by the "
                "competition organizers before the start of the competition.")


def _replay(m, page11: str, page12: str) -> tuple[str, str]:
    """Run the real strip over the two pages, returning (with strip, without strip)."""
    rec, trace = [], []
    stripped = m.norm("\n".join([m.strip_page_furniture(page11, 11, rec, trace=trace),
                                 m.strip_page_furniture(page12, 12, rec, trace=trace)]))
    raw = m.norm("\n".join([m.strip_page_furniture(page11, 11, None, trace=None),
                            m.strip_page_furniture(page12, 12, None, trace=None)]) if False else
                 m.norm("\n".join([page11, page12])))
    return stripped, raw


def test_the_page_number_may_be_emitted_as_its_own_line_glued_or_after_a_blank_line():
    m = _mod()
    for label, page12 in (
            ("own line", "12\n" + _PAGE12_BODY),
            ("glued to the wrapped sentence", "12 " + _PAGE12_BODY),
            ("after a blank first line", "\n12\n" + _PAGE12_BODY),
            ("numbered page, number at its foot", _PAGE12_BODY + "\n12"),
    ):
        stripped, raw = _replay(m, _PAGE11_HEAD, page12)
        assert m.norm(_WRAPPED_QUOTE) in stripped, f"the strip did not repair the {label} shape"
        if label != "numbered page, number at its foot":
            assert m.norm(_WRAPPED_QUOTE) not in raw, \
                f"the {label} shape was expected to break matching without the strip"
    # and the page number must never be silently swallowed from a quotation that needs it: the
    # removals are recorded, so a reviewer can put it back
    rec = []
    m.strip_page_furniture("12\n" + _PAGE12_BODY, 12, rec)
    assert rec == [{"page": 12, "removed": "12", "why": "bare number at the top of the page"}]


def test_a_real_pdf_with_page_margin_numbers_extracts_to_a_matchable_sentence(tmp_path):
    """End-to-end through pypdf, on a PDF built to have the same defect (skipped if not installed).

    The rules PDF itself cannot be fetched from the development sandbox (nlr.gov is not on the
    allowlist), so the mechanism is proven on a synthetic document with the same structure and the
    real document is checked by the workflow.
    """
    import pytest
    reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas").Canvas
    pypdf = pytest.importorskip("pypdf")
    pdf = tmp_path / "furniture.pdf"
    c = reportlab_canvas(str(pdf), pagesize=(612, 792))
    for number, body in (("11", _PAGE11_HEAD), ("12", _PAGE12_BODY)):
        c.drawString(72, 760, number)
        y, line = 700, ""
        for word in body.split():
            if len(line) + len(word) + 1 > 88:            # wrap on word boundaries: drawing a chunk
                c.drawString(72, y, line)                  # that splits a word makes pypdf emit a
                y, line = y - 16, word                     # spurious space inside it
            else:
                line = (line + " " + word).strip()
        c.drawString(72, y, line)
        c.showPage()
    c.save()

    m = _mod()
    notes, trace = [], []
    text = m.norm(m.extract(pdf, notes=notes, trace=trace))
    assert len(trace) == 2 and trace[0]["page"] == 1
    assert m.norm(_WRAPPED_QUOTE) in text, "the wrapped sentence is not matchable after extraction"
    assert notes, "the page numbers were not recorded as removed"
    for r in notes:
        assert set(r) == {"page", "removed", "why"} and r["removed"].isdigit()
    # ...and the escape hatch still reproduces the old behaviour, defects and all
    kept = m.norm(m.extract(pdf, keep_page_furniture=True))
    assert m.norm(_WRAPPED_QUOTE) not in kept


def test_the_strip_records_every_removal_for_audit():
    """Nothing may be removed silently.

    WHY THIS TEST EXISTS (and why it is back): the strip can only be trusted if a reviewer can see
    exactly what it took out and why - the whole point of `--keep-page-furniture` is that the same
    extraction is reproducible without it.  MEASURED 2026-09-16: this test was lost when a later edit
    sliced the file to EOF from the previous test's definition; the committed report then carried
    `page_furniture_removed: []` for a whole run without anything failing.  It is re-added here, and
    the committed report is checked against the same contract as well, so a report that stops
    recording removals fails the suite rather than passing quietly.
    """
    m = _mod()
    record: list = []
    trace: list = []
    # page 12 as pypdf really emits it: its number glued to the head of the page's first line, and
    # the next page's number at the foot.  Both are furniture; the prose between them is not.
    text = "12 relative weight of faults\n13\n"
    kept = m.strip_page_furniture(text, 12, record, prev_tail="and the", trace=trace)
    assert kept == "relative weight of faults"
    # every removal is recorded with the page it came from, the exact string removed, and the reason
    assert record, "the strip removed text without recording it"
    assert all(set(r) == {"page", "removed", "why"} for r in record)
    assert sorted(r["removed"] for r in record) == ["12", "13"]
    # a bare number at the head or foot of a page is recorded too, not just the glued case
    record2: list = []
    assert m.strip_page_furniture("11\nprose line\n12\n", 12, record2) == "prose line"
    assert sorted(r["removed"] for r in record2) == ["11", "12"]
    assert {r["page"] for r in record} == {12}
    assert all(r["why"] for r in record)
    # the trace keeps the raw page head so a reviewer can see what the extractor actually produced
    assert trace and "raw_head_repr" in trace[0] and "blank_lines_before_first_text" in trace[0]
    assert "12" in trace[0]["raw_head_repr"]

    # ...and the committed report must obey the same contract when it claims a strip happened
    rep = json.loads((ROOT / "data/evidence/rules_quotes.json").read_text())
    if rep.get("page_furniture_stripped"):
        assert rep.get("page_furniture_removed"), \
            "the report says page furniture was stripped but records no removal"
