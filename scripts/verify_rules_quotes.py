#!/usr/bin/env python
"""Verify the official rules line by line, and fail loudly when a sentence moves.

WHY THIS EXISTS.  The single most consequential fact in this challenge is which population is
scored, and it is stated in prose: Phase 1 scores "a privately withheld subset of the original new
fault dataset", Phase 2 scores "the full, revised new fault dataset", while the training labels are
the *existing* USGS Quaternary compilation.  Every strategy decision in this repository follows from
those sentences, so they are not paraphrased anywhere - they are quoted verbatim and checked
mechanically against the official document, with the document's sha256 recorded.

It also settles the opposite reading, which is worth stating because it is the intuitive one: the
pre-existing catalogue is NOT what either phase scores ("all fault labels in this updated label
set", "the complete updated test set created by expert review").  A submission that reproduces the
catalogue therefore earns credit only where the experts' new labels happen to coincide with it.

USAGE
    python scripts/verify_rules_quotes.py --url https://docs.nlr.gov/docs/fy26osti/96647.pdf \
        --out data/evidence/rules_quotes.json [--expected-sha256 <sha recorded for the mirror>]
    python scripts/verify_rules_quotes.py --pdf local.pdf --out ...
    python scripts/verify_rules_quotes.py --text extracted.txt      (offline; no extraction step)

Exit code is 0 only when every quote is found verbatim (after whitespace/quote normalisation).
The runner does the fetch: this sandbox's egress allowlist does not include docs.nlr.gov, which is
why the check is a workflow step rather than a local one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The official location of the rules, recorded in every report so that a local extraction path
# (--pdf /tmp/rules_canonical.pdf, as the workflow uses) can never become the published link.
RULES_URL = "https://docs.nlr.gov/docs/fy26osti/96647.pdf"

# (id, section hint, exact sentence as printed, why it matters here)
QUOTES = [
    ("two_phases", "§1.1",
     "Participants will submit a single entry, which will be evaluated in two prize phases using a "
     "distance-weighted Tversky index.",
     "One submission is scored twice; there is no separate Phase-2 upload to optimise for."),
    ("phase1_target", "§1.1",
     "In Phase 1, submissions will be evaluated against a privately withheld subset of the original "
     "new fault dataset compiled by expert reviewers.",
     "The scored population is NEW faults, not the catalogue we can download."),
    ("experts_revise", "§1.1",
     "After Phase 1, expert reviewers will use submitted predictions to revise the new fault dataset.",
     "Discoveries can be confirmed by experts - predictions that look like real new faults matter."),
    ("phase2_eligibility", "§1.1",
     "All Phase 1 competitors will be eligible to compete in Phase 2 and will be automatically "
     "submitted for consideration.",
     "No top-5 cutoff gates the larger prize pool: Phase-1 risk-taking buys nothing."),
    ("phase2_target", "§1.1",
     "Submissions will be reevaluated against the full, revised new fault dataset using the same "
     "distance-weighted Tversky index.",
     "Phase 2 is the same metric on the same population - one objective, two label revisions."),
    ("prize_pools", "§1.1",
     "The Phase 2 prize pool of $250,000 will be distributed among the top five competitors, as "
     "judged by their performance on all fault labels in this updated label set.",
     "$250k of the $300k is decided on the revised new-fault labels; $50k in Phase 1."),
    ("labels_source", "§2",
     "The labels for this prize come from the USGS Quaternary Fault and Fold Database and from a set "
     "of newly identified faults labeled by geology experts at the National Laboratory of the Rockies "
     "(NLR) and USGS.",
     "Two populations in one sentence: the public catalogue (training) and the expert-mapped new "
     "faults (scoring)."),
    ("ranking_basis", "§3.2",
     "Second-round prize rankings will be determined by running the selected final submissions "
     "against the complete updated test set created by expert review.",
     "Phase 2 ground truth is the *complete updated* set - expert new faults, not the old database."),
    ("features_source", "§2",
     "The feature data for this prize come from the recently released Geoscience Data Acquisition "
     "for Western Nevada (GeoDAWN)",
     "The features are the GeoDAWN geophysics (lidar/magnetic/radiometric), i.e. what the experts "
     "themselves mapped the new faults from."),
    ("dem_is_a_feature", "§2",
     "In addition, the feature data also contain U.S. Geological Survey (USGS) Digital Elevation "
     "Model (DEM) elevation data at 1-m resolution.",
     "The 1 m DEM is intended input data, not an external extra - it is admissible."),
    ("single_geotiff", "§3.2",
     "You must submit a single GeoTIFF with a single raster layer at 100-meter resolution containing "
     "your model's predictions of fault locations for the entirety of the GeoDAWN study area.",
     "Submission format, exactly."),
    ("weekly_limit", "§3.2",
     "You can make multiple submissions, subject to the limits specified on the competition website "
     "(three submissions per week).",
     "Three measured shots per week at the real metric - the only honest feedback available."),
    ("ai_disclosure", "§3.2",
     "you must indicate in the narrative (not included in the word count) the extent to which, if "
     "any, you used generative AI technology",
     "Required in the final narrative; this repository documents its own AI use."),
    ("code_assets", "§3.2",
     "The solution assets must contain a description of the resources required to build and run the "
     "solution, and they should be able to sufficiently reproduce the winning results and generate "
     "predictions on new data samples.",
     "Why every run here commits its logs, hashes, environment pins and reproduce steps."),
    # ---- added 2026-09-16 (session 7) from an INDEPENDENT network path --------------------
    # The sentences below were read from https://docs.nlr.gov/docs/fy26osti/96647.pdf through a
    # different fetcher from the one this workflow uses (the agent tool's PDF extractor, whose
    # egress is not subject to the development sandbox's allowlist).  They are added here so the
    # reading is machine-checked on every run instead of resting on a hand-read: two independent
    # extractions agreeing verbatim is stronger evidence than either alone.  A mismatch fails the
    # step, which is the point - `entry`/`citizen`/... below behave the same way.
    ("prize_split", "§1.1",
     "There will be two phases of prize awards.",
     "Two prize rounds over one submission; the split matters for risk allocation."),
    ("phase1_pool_amount", "§1.1",
     "The Phase 1 pool of $50,000 will be distributed equally among the top five competitors, as "
     "judged by their performance on the private test set of fault labels.",
     "Phase 1 is an equal split among the top five, i.e. rank inside the top five is not paid "
     "differently - the $250k Phase 2 ranking is where rank itself pays."),
    ("experts_update_labels", "§1.1",
     "A panel of experts will then use the submitted predictions to update fault labels in the region.",
     "Submissions are read by the experts before Phase 2: a false positive that is geologically "
     "plausible can become a label, which the metric alone does not reward."),
    ("features_single_geotiff", "§3.3",
     "This dataset will be provided as a single multiband GeoTIFF, with one feature per band.",
     "19 bands in one file on one grid - band order is the only documentation of semantics."),
    ("dem_download_instructions", "§3.3",
     "In addition, instructions will be provided for downloading USGS DEM elevation data at 1-m "
     "resolution for the GeoDAWN region.",
     "The 1 m DEM is delivered as a link list (1m_DEM_links.csv / the data tab's PDF), not as "
     "raster tiles - which is why this repo stores a verified tile inventory in data/dem_links.json."),
    ("all_faults", "§3.3",
     "Competitors will submit their predictions for all faults in the GeoDAWN study area as a "
     "GeoTIFF raster at 100-m resolution.",
     "'All faults' - not only the new ones and not only the catalogued ones."),
    ("one_final_submission", "§3.5",
     "Before the end of the competition, you must choose only one submission for evaluation across "
     "both prize rounds.",
     "One artefact must serve both rounds; nothing can be tuned for one and swapped for the other."),
    ("one_final_per_entity", "§3.4",
     "Each participating entity (team, organization, or individual prize competitor not on a team) "
     "is allowed to have one final submission",
     "Per-entity, not per-account: duplicate submissions from one entity are not a strategy."),
    ("no_private_knowledge", "§3.6.2",
     "You must choose only one submission to use for scoring across both prize rounds, and you must "
     "make your decision without knowledge of your scores on the private test set.",
     "Explicit anti-overfitting rule: the public leaderboard is the only feedback there is."),
    ("test_set_composition", "§3.6.2",
     "The set of faults included in the public test dataset and the relative weight of faults in "
     "both test datasets will be determined by the competition organizers before the start of the "
     "competition.",
     "Public/private composition and fault weighting are the organizers' choice and are fixed "
     "before the start - so public-LB optimising cannot reweight them."),
    ("entry", "§3.1",
     "To enter the competition, you must create a profile on the DrivenData platform and agree to "
     "abide by the competition rules and restrictions.",
     "The one genuinely human step: registration, then the account-gated data tab and upload."),
    ("citizen", "§1.3",
     "An individual prize competitor (who is not competing as a member of a group) must be a U.S. "
     "citizen or permanent resident.",
     "Eligibility - checked before any prize is contemplated."),
    ("payment", "§A.2",
     "Each competitor must sign and return to the prize administrator within 30 days of the date on "
     "the notice a completed NLR Request for ACH Banking Information form and a completed IRS W-9 "
     "form.",
     "What winning actually requires administratively."),
    ("public_elements", "§A.4",
     "The elements of the submission that are designated as public will become publicly available as "
     "part of this prize.",
     "Public-designated elements must contain no trade secrets; everything here is public already."),
    ("single_award", "§A.3",
     "The prize administrator will award a single dollar amount to the designated primary submitter, "
     "whether consisting of a single entity or multiple entities.",
     "Team allocation is the team's problem, not the sponsor's."),
]


def divergence(quote: str, text: str, window: int = 120) -> dict:
    """Where does the document stop agreeing with a quote?  (bisect the longest present prefix)

    Used only for quotes that failed, so a failure ships with the document's own words next to the
    quoted ones - which is how a typo in a quote gets distinguished from a moving document.
    """
    lo, hi = 0, len(quote)
    while lo < hi:                                    # longest prefix present as a substring
        mid = (lo + hi + 1) // 2
        if quote[:mid] in text:
            lo = mid
        else:
            hi = mid - 1
    at = text.find(quote[:lo]) if lo else -1
    start = at + lo if at >= 0 else -1
    return {"prefix_len": lo,
            "prefix_tail": quote[max(0, lo - 60):lo],
            "quote_continues": quote[lo:lo + window],
            "doc_continues": text[start:start + window] if start >= 0 else None,
            "doc_offset": start}


def norm(text: str) -> str:
    """Normalise a document and a quote to the same form.

    - NFKC folds the ligatures and full-width characters PDF extraction produces
    - curly quotes/apostrophes and en/em dashes are folded to ASCII
    - a hyphen followed by a newline is a line break inside a word, not a hyphen
    - all whitespace collapses to a single space
    """
    t = unicodedata.normalize("NFKC", text)
    t = (t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
          .replace("\u201d", '"').replace("\u2014", "-").replace("\u2013", "-")
          .replace("\u00a0", " "))
    t = re.sub(r"-\s*\n\s*", "", t)          # de-hyphenate across line breaks
    t = re.sub(r"\s+", " ", t)
    return t


def _trace_page(trace: list, page_no: int, raw: str, kept: str) -> None:
    """Record how one page came out of the extractor, so a miss can be diagnosed from evidence.

    MEASURED 2026-09-16: the first attempt at removing page furniture removed nothing at all -
    the page number was not where the diagnostic implied it would be (top or foot of the page),
    and no job log is readable from the development sandbox.  The trace answers the question the
    PDF cannot be re-fetched to ask: what does each page's own text look like at its head and tail?
    """
    def _lines(t: str) -> list:
        return [ln.strip() for ln in t.splitlines() if ln.strip()]
    rk, rl = _lines(raw), _lines(kept)
    blanks = len([ln for ln in raw.splitlines()[:3] if not ln.strip()])
    trace.append({"page": page_no, "chars": len(raw),
                  "head": (rl[0][:80] if rl else ""), "tail": (rl[-1][:80] if rl else ""),
                  "raw_head": (rk[0][:40] if rk else ""), "raw_tail": (rk[-1][:40] if rk else ""),
                  # escaped: an invisible character would otherwise survive into the report looking
                  # exactly like a plain digit
                  "raw_head_repr": (repr(rk[0])[:60] if rk else ""),
                  "blank_lines_before_first_text": blanks})


def strip_page_furniture(page_text: str, page_no: int | None = None,
                         record: list | None = None, prev_tail: str | None = None,
                         trace: list | None = None) -> str:
    """Remove page-margin furniture (a bare page number) from ONE page's extracted text.

    MEASURED 2026-09-16 (run 35153102372).  pypdf emits a page's number as the first line of that
    page, so a sentence that continues across a page break comes back with a bare number inside it:

        "... and the 12 relative weight of faults in both test datasets will be determined ..."

    That is furniture, not prose: it is what a reader sees in the margin, and leaving it in makes a
    genuinely verbatim quotation unmatchable.  Only lines that are *nothing but* a number, at the
    very start or the very end of a page, are removed; nothing inside the page's text is touched,
    and every other character is compared exactly as before.  The old behaviour is still available
    with --keep-page-furniture, so this cannot quietly become a fuzzy match.
    """
    # Blank lines are dropped rather than kept as positions: the extractor emits a leading newline
    # on some pages (MEASURED 2026-09-16 - run 35153553275 removed nothing although every page's
    # first *visible* line was its number), and the only consumer of this text normalises whitespace
    # away anyway.  The rules below must see the page's first and last REAL lines.
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]

    def _drop(idx: int, why: str) -> None:
        if record is not None:
            record.append({"page": page_no, "removed": lines[idx], "why": why})
        lines.pop(idx)

    while lines and re.fullmatch(r"\d{1,4}", lines[0]):
        _drop(0, "bare number at the top of the page")
    while lines and re.fullmatch(r"\d{1,4}", lines[-1]):
        _drop(len(lines) - 1, "bare number at the foot of the page")
    # pypdf sometimes emits the page number glued to the first line of that page
    # ("12 relative weight of faults ..."), so a sentence continuing across the break reads as
    # "... and the 12 relative weight ...".  Only a leading run of digits EQUAL TO THE PAGE NUMBER
    # is removed - a number in prose that merely starts a page is left alone.
    continues = prev_tail is None or not re.search(r"[.!?:;)\]]\s*$", prev_tail.strip())
    if page_no is not None and lines and continues:
        m = re.match(r"^(\d{1,4})\s+(\S.*)$", lines[0], flags=re.S)
        if m and int(m.group(1)) == page_no:
            if record is not None:
                record.append({"page": page_no, "removed": m.group(1),
                               "why": "page number glued to the first line of the page "
                                      "(the previous page ends mid-sentence)"})
            lines[0] = m.group(2)
    kept = "\n".join(lines)
    if trace is not None:
        _trace_page(trace, page_no if page_no is not None else -1, page_text, kept)
    return kept


def extract(path: Path, keep_page_furniture: bool = False, notes: list | None = None,
            trace: list | None = None) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        from pypdf import PdfReader
    except ImportError:                                             # pragma: no cover
        raise SystemExit("pypdf is required to read a PDF: pip install pypdf")
    reader = PdfReader(str(path))
    pages, prev_tail = [], None
    for i, pg in enumerate(reader.pages, start=1):
        raw = pg.extract_text() or ""
        t = raw
        if not keep_page_furniture:
            t = strip_page_furniture(t, i, notes, prev_tail=prev_tail, trace=trace)
            tail_lines = [ln for ln in t.splitlines() if ln.strip()]
            prev_tail = " ".join(tail_lines[-2:]) if tail_lines else None
        elif trace is not None:
            _trace_page(trace, i, raw, t)
        pages.append(t)
    return "\n".join(pages)


def fetch(url: str, dest: Path) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "gemsdoe-rules-verify/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:              # noqa: S310 - fixed https URL
        data = r.read()
    dest.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf")
    src.add_argument("--text")
    src.add_argument("--url")
    ap.add_argument("--out", default=str(ROOT / "data/evidence/rules_quotes.json"))
    ap.add_argument("--expected-sha256", default=None,
                    help="sha256 recorded for the data-tab mirror; proves the two copies are identical")
    ap.add_argument("--keep-page-furniture", action="store_true",
                    help="keep bare page numbers at page starts/ends (pre-2026-09-16 behaviour). "
                         "Kept as a flag so the removal is visible and reversible, not silent.")
    ap.add_argument("--dump-text", default=None,
                    help="also write the NORMALISED extracted text here (public-domain US "
                         "government document).  Without it, every check of a quote costs a CI "
                         "round-trip: the development sandbox cannot reach docs.nlr.gov, so the "
                         "only way to see what the extractor actually produced is to have it "
                         "committed.")
    ap.add_argument("--source-url", default=RULES_URL,
                    help="Official URL of the document.  Recorded separately from the extraction "
                         "path so the site links to the publisher, never to /tmp.")
    a = ap.parse_args()

    sha = None
    nbytes = None
    if a.url:
        tmp = Path("/tmp/rules_canonical.pdf")
        nbytes, sha = fetch(a.url, tmp)
        doc_path = tmp
    else:
        doc_path = Path(a.pdf or a.text)
        b = doc_path.read_bytes()
        nbytes, sha = len(b), hashlib.sha256(b).hexdigest()

    furniture: list = []
    page_trace: list = []
    text = norm(extract(doc_path, keep_page_furniture=a.keep_page_furniture,
                        notes=furniture, trace=page_trace))
    if a.dump_text:
        dp = Path(a.dump_text)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(text, encoding="utf-8")
        print(f"wrote the normalised document text to {dp} ({len(text)} chars)")
    print(f"page trace: {len(page_trace)} pages; first page head={page_trace[0]['raw_head']!r} "
          f"tail={page_trace[0]['raw_tail']!r}") if page_trace else None
    if furniture:
        print(f"page furniture removed before matching: {len(furniture)} line(s) - "
              f"{[f['removed'] for f in furniture][:10]}"
              f"{'...' if len(furniture) > 10 else ''} (recorded in the report)")
    results = []
    for qid, section, quote, why in QUOTES:
        found = norm(quote) in text
        row = dict(id=qid, section=section, quote=quote, why=why, exact_match=bool(found))
        if not found:
            # A miss must be diagnosable from the committed evidence, not only from a job log that
            # the development sandbox cannot even download (the Actions log host is blocked).  So
            # report where the document diverges from the quote: the longest prefix that IS present,
            # and the document's own text continuing from there.
            row["diagnostic"] = divergence(norm(quote), text)
        results.append(row)
        print(f"{'OK  ' if found else 'MISS'} {qid:20s} {section:6s} {quote[:72]}...")
        if not found:
            d = row["diagnostic"]
            print(f"      longest matching prefix: {d['prefix_len']} chars: ...{d['prefix_tail']!r}")
            print(f"      document continues with : {d['doc_continues']!r}")

    n_found = sum(1 for r in results if r["exact_match"])
    payload = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/verify_rules_quotes.py",
        source=dict(url=a.url or str(doc_path), canonical_url=a.source_url, bytes=nbytes,
                    sha256=sha),
        method=("verbatim substring match after NFKC, quote/dash folding, de-hyphenation across line "
                "breaks, whitespace collapse and removal of page numbers at page margins - no "
                "paraphrasing, no fuzzy matching"),
        page_furniture_stripped=(not a.keep_page_furniture),
        page_furniture_removed=furniture,
        pages=page_trace,
        extracted_text=(dict(path=str(a.dump_text),
                             chars=len(text)) if a.dump_text else None),
        match_against_mirror=dict(expected_sha256=a.expected_sha256,
                                  identical=bool(a.expected_sha256 and sha == a.expected_sha256)),
        quotes=results,
        summary=dict(n_quotes=len(results), n_found=n_found, all_found=n_found == len(results)),
        conclusion=("Both prize phases are scored against the expert-mapped NEW fault dataset (Phase 1 "
                    "a private subset, Phase 2 the full revised set). The catalogue in the data tab is "
                    "the training labels and is not the scored population."),
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}: {n_found}/{len(results)} quotes found; sha256 {sha}")
    if a.expected_sha256 and sha != a.expected_sha256:
        print(f"WARNING: sha256 differs from the recorded mirror ({a.expected_sha256})", file=sys.stderr)
    return 0 if n_found == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
