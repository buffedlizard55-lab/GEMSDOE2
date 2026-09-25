#!/usr/bin/env python3
"""Measure, in this checkout, whether a submission can actually be produced and uploaded.

WHY A SCRIPT AND NOT A PARAGRAPH
--------------------------------
"How to submit" is the one page a reader acts on, so every gate on it has to be a measurement
from *this* checkout rather than a sentence someone typed once.  This script produces
`data/evidence/submission_readiness.json`; `scripts/build_site.py` renders it on
`docs/how_to_submit.html`.  Each check carries:

    status   PASS | FAIL | MISSING | HUMAN
    measured the quantity it was derived from (bytes, sha256, counts, exit codes, log lines)
    source   the file or URL a reviewer can re-check by hand

The `HUMAN` status is deliberate and is not a bug: enrolling on DrivenData, uploading the file
and reading a private leaderboard cannot be done by any program in this repository.  Rendering
them as "done" would be the single most damaging lie this project could tell, so they are
rendered as what they are.

CHECKS
  1. competition data placed      - the three canonical rasters, sha256 vs data/bridge/manifest.json
  2. pre-flight passes            - scripts/prepare_data.py re-executed (its exit code, not a claim)
  3. submission artifact present  - sha256 re-hashed and compared with the recorded .sha256
  4. artifact format validated    - the committed validator log parsed for its PASSED line
  5. rules quotes verified        - data/evidence/rules_quotes.json counts + source sha256
  6. CPU baseline route           - data/evidence/baseline/baseline_report.json (the no-GPU path)
  7. in-browser generator         - runs docs/geotiff_writer.js under node and judges its output,
                                    via scripts/check_site_generator.py (the site can hand over a file)
  8. human steps remaining        - enroll / upload / read the leaderboard / pick the final entry
  9. primary Recall-Union candidate - public/evidence bytes match its report and identity
 10. artifact candidates      - every committed raster that could be uploaded, with its hash

USAGE
    python scripts/check_submission_readiness.py                    # writes the evidence JSON
    python scripts/check_submission_readiness.py --data-dir data --skip-preflight
    python scripts/check_submission_readiness.py --print             # also print a human summary
Exit code is 0 when every machine check that can pass does; 2 when a *mismatch* is found
(a wrong hash is a defect, a missing file in a fresh checkout is just a fresh checkout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANONICAL = {
    "training_features.tif": "gems-geodawn-numerical-features.tif",
    "labels.tif": "existing_faults.tif",
    "sample_submission.tif": "example_submission.tif",
}
SHIPPED_DIR = "data/evidence/runs/ens12-adopted-floor0.1-w0"
PRIMARY_REPORT = "data/evidence/runs/recall-union-v1/recall_union_report.json"
PRIMARY_PUBLIC = "docs/gemsdoe2_recall_union_submission.tif"
PRIMARY_EVIDENCE = "data/evidence/runs/recall-union-v1/submission.tif"
HUMAN_STEPS = [
    ("Create the DrivenData profile and accept the competition rules", "rules §3.1 verbatim quote"),
    ("Confirm prize eligibility (citizenship / residence; rules §1.3, App. A)", "rules §1.3 verbatim quote"),
    ("Upload the chosen .tif on the competition's Submit tab", "competition data tab (login required)"),
    ("Read the public leaderboard score the platform returns", "competition leaderboard"),
    ("Choose the single final submission before the deadline (rules §3.5, §3.6.2)", "rules quotes"),
    ("Paste the generative-AI disclosure into the submission narrative (rules §3.2)", "rules §3.2 verbatim quote"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(check_id: str, label: str, status: str, detail: dict, source: str,
           note: str = "") -> dict:
    return dict(id=check_id, label=label, status=status, measured=detail, source=source,
                note=note)


# ------------------------------------------------------------------ 1. data placement
def check_data_placed(data_dir: Path) -> dict:
    man_path = ROOT / "data/bridge/manifest.json"
    if not man_path.exists():
        return _check("data_placed", "Official competition rasters placed in data/", "MISSING",
                      dict(manifest=None), "data/bridge/manifest.json",
                      "no bridge manifest - run `python scripts/assemble_data_bridge.py`")
    man = json.loads(man_path.read_text())
    pins = {f["name"]: f for f in man["files"]}
    rows, ok, mismatch = [], True, False
    for canonical, mirror in CANONICAL.items():
        pin = pins.get(mirror)
        if pin is None:
            rows.append(dict(file=canonical, mirror=mirror, status="NOT_IN_MANIFEST"))
            ok = False
            continue
        p = data_dir / canonical
        if not p.exists():
            rows.append(dict(file=canonical, mirror=mirror, status="MISSING",
                             pinned_sha256=pin["sha256"], pinned_bytes=pin["bytes"]))
            ok = False
            continue
        got_sha, got_bytes = sha256_file(p), p.stat().st_size
        match = (got_sha == pin["sha256"] and got_bytes == pin["bytes"])
        mismatch |= (not match)
        ok &= match
        rows.append(dict(file=canonical, mirror=mirror,
                         status="VERIFIED" if match else "MISMATCH",
                         sha256=got_sha, bytes=got_bytes, pinned_sha256=pin["sha256"],
                         pinned_bytes=pin["bytes"]))
    return _check("data_placed", "Official competition rasters placed in data/",
                  "PASS" if ok else ("FAIL" if mismatch else "MISSING"), dict(files=rows),
                  "data/bridge/manifest.json (pins from data/evidence/inventory.json)")


# ------------------------------------------------------------------ 2. pre-flight
def check_preflight(data_dir: Path, skip: bool) -> dict:
    src = dict(script="scripts/prepare_data.py")
    if skip:
        return _check("preflight", "scripts/prepare_data.py passes on the placed bytes", "SKIPPED",
                      src, "scripts/prepare_data.py", "skipped by flag")
    if not (data_dir / "training_features.tif").exists():
        return _check("preflight", "scripts/prepare_data.py passes on the placed bytes", "MISSING",
                      src, "scripts/prepare_data.py", "data/ is not placed in this checkout")
    proc = subprocess.run([sys.executable, "scripts/prepare_data.py"], cwd=str(ROOT),
                          capture_output=True, text=True)
    tail = [ln for ln in (proc.stdout or "").strip().splitlines() if ln.strip()][-1:]
    src.update(exit_code=proc.returncode, last_line=tail[0] if tail else "",
               output_sha256=hashlib.sha256((proc.stdout or "").encode()).hexdigest())
    return _check("preflight", "scripts/prepare_data.py passes on the placed bytes",
                  "PASS" if proc.returncode == 0 else "FAIL", src, "scripts/prepare_data.py",
                  "the exit code of the real pre-flight, re-executed now")


# ------------------------------------------------------------------ 3/4. artifact
def check_artifact() -> dict:
    d = ROOT / SHIPPED_DIR
    art, rec = d / "submission.tif", d / "submission.sha256"
    if not art.exists():
        return _check("artifact", "Shippable artifact present and hash-consistent", "MISSING",
                      dict(path=f"{SHIPPED_DIR}/submission.tif"), f"{SHIPPED_DIR}/submission.sha256")
    got = sha256_file(art)
    recorded = None
    if rec.exists():
        recorded = rec.read_text().strip().split()[0]
    match = bool(recorded) and got == recorded
    return _check("artifact", "Shippable artifact present and hash-consistent",
                  "PASS" if match else "FAIL",
                  dict(path=f"{SHIPPED_DIR}/submission.tif", sha256=got, bytes=art.stat().st_size,
                       recorded_sha256=recorded, recorded_in=f"{SHIPPED_DIR}/submission.sha256"),
                  f"{SHIPPED_DIR}/submission.sha256")


def check_validation() -> dict:
    log = ROOT / SHIPPED_DIR / "validation.log"
    if not log.exists():
        return _check("artifact_validated", "Committed validator log reports PASSED", "MISSING",
                      dict(path=f"{SHIPPED_DIR}/validation.log"), f"{SHIPPED_DIR}/validation.log")
    text = log.read_text()
    checks = [ln.strip() for ln in text.splitlines()
              if ln.strip().startswith(("✓", "✗", "x ")) or "] NaN fraction" in ln]
    return _check("artifact_validated", "Committed validator log reports PASSED",
                  "PASS" if "Validation PASSED" in text else "FAIL",
                  dict(path=f"{SHIPPED_DIR}/validation.log", log_sha256=sha256_file(log),
                       checks=checks),
                  f"{SHIPPED_DIR}/validation.log")


# ------------------------------------------------------------------ 5. rules
def check_rules() -> dict:
    p = ROOT / "data/evidence/rules_quotes.json"
    if not p.exists():
        return _check("rules_quotes", "Official rules sentences verified verbatim", "MISSING",
                      {}, "data/evidence/rules_quotes.json")
    d = json.loads(p.read_text())
    s = d.get("summary", {})
    missing = [q["id"] for q in d.get("quotes", []) if not q.get("exact_match")]
    return _check("rules_quotes", "Official rules sentences verified verbatim",
                  "PASS" if s.get("all_found") and not missing else "FAIL",
                  dict(n_quotes=s.get("n_quotes"), n_found=s.get("n_found"),
                       all_found=bool(s.get("all_found")), not_exact=missing,
                       source_pdf_sha256=(d.get("source") or {}).get("sha256"),
                       generated_utc=d.get("generated_utc")),
                  "data/evidence/rules_quotes.json")


# ------------------------------------------------------------------ 6. CPU baseline route
def check_baseline() -> dict:
    p = ROOT / "data/evidence/baseline/baseline_report.json"
    if not p.exists():
        return _check("cpu_baseline", "CPU-only route (no GPU, no torch) can produce a submission",
                      "MISSING", dict(report=None),
                      "scripts/baseline_submission.py",
                      "run `python scripts/baseline_submission.py` to measure this route")
    d = json.loads(p.read_text())
    sub = d.get("submission", {})
    ok = bool(sub.get("sha256")) and int(sub.get("nonzero_px", 0)) > 0
    return _check("cpu_baseline", "CPU-only route (no GPU, no torch) can produce a submission",
                  "PASS" if ok else "FAIL",
                  dict(report="data/evidence/baseline/baseline_report.json",
                       generated_utc=d.get("generated_utc"), policy=sub.get("policy"),
                       submission_sha256=sub.get("sha256"), bytes=sub.get("bytes"),
                       emitted_px=sub.get("nonzero_px"),
                       global_scores=sub.get("global_scores"),
                       held_out_fold=d.get("held_out_fold"),
                       measurement_fold=d.get("measurement_fold"),
                       generalisation={k: v.get("dti") for k, v in
                                       (d.get("generalisation", {}).get("by_population") or {}).items()},
                       audit_reproduced=(d.get("generalisation", {}).get("audit") or {}).get("passed"),
                       audit_max_delta=(d.get("generalisation", {}).get("audit") or {}).get("max_abs_delta"),
                       minutes=round(float(d.get("timings_s", {}).get("total", 0)) / 60.0, 1)),
                  "data/evidence/baseline/baseline_report.json")


# ------------------------------------------------------------------ 7. human steps
def check_generator(skip: bool) -> dict:
    """Re-run scripts/check_site_generator.py rather than trusting its last committed report.

    Why this is a gate at all: docs/how_to_submit.html offers a button that writes the GeoTIFF in
    the reader's browser, so "the site can generate the file" is a claim the repository has to
    measure. The generator check executes that exact JavaScript under node and judges the output
    with rasterio plus the repository's own validator; ~10 s, and it fails if the shipped payload
    drifts from the artifact.
    """
    src = "scripts/check_site_generator.py -> data/evidence/site_generator.json"
    if skip:
        ev = ROOT / "data/evidence/site_generator.json"
        det = json.loads(ev.read_text()) if ev.exists() else {}
        return _check("site_generator", "The site can generate the .tif itself (in-browser writer)",
                      str(det.get("verdict", "SKIPPED")).upper() if det else "MISSING",
                      dict(skipped=True, committed_verdict=det.get("verdict"),
                           steps=len(det.get("steps") or [])), src,
                      "re-check omitted by --skip-generator; the verdict shown is the last committed one")
    proc = subprocess.run([sys.executable, "scripts/check_site_generator.py"], cwd=str(ROOT),
                          capture_output=True, text=True)
    ev = ROOT / "data/evidence/site_generator.json"
    det = json.loads(ev.read_text()) if ev.exists() else {}
    steps = det.get("steps") or []
    passed = [s["name"] for s in steps if s["status"] == "PASS"]
    failed = [s["name"] for s in steps if s["status"] == "FAIL"]
    status = ("PASS" if proc.returncode == 0 and det.get("verdict") == "PASS" else
              "MISSING" if proc.returncode == 2 or det.get("verdict") == "MISSING" else "FAIL")
    return _check("site_generator", "The site can generate the .tif itself (in-browser writer)", status,
                  dict(exit_code=proc.returncode, verdict=det.get("verdict"),
                       node=(det.get("summary") or {}).get("node"),
                       steps=len(steps), passed=passed, failed=failed,
                       generated_utc=det.get("generated_utc"), seconds=det.get("seconds"),
                       container_note=next((s["note"] for s in steps
                                            if "NODATA" in s["name"]), None)),
                  src,
                  "" if status == "PASS" else
                  ("no node in this environment - the browser route cannot be verified headlessly; "
                   "the artifact route (curl) still works" if status == "MISSING"
                   else "the shipped payload or the writer does not reproduce the artifact: "
                        + "; ".join(failed[:3])))


def check_human() -> dict:
    return _check("human_steps", "Steps only a person can complete", "HUMAN",
                  dict(steps=[dict(action=a, source=s) for a, s in HUMAN_STEPS]),
                  "competition site + rules PDF",
                  "no program in this repository can create an account, upload a file or read a "
                  "private leaderboard; these stay open until a human does them")


# ------------------------------------------------------------------ 8. primary candidate
def check_primary_candidate() -> dict:
    """Verify the selected public candidate's report, public bytes, and audit copy agree."""
    report_path = ROOT / PRIMARY_REPORT
    public = ROOT / PRIMARY_PUBLIC
    evidence = ROOT / PRIMARY_EVIDENCE
    if not report_path.exists() or not public.exists() or not evidence.exists():
        return _check(
            "primary_candidate",
            "Recall-Union v1 public and evidence artifacts match their report",
            "MISSING",
            dict(report=PRIMARY_REPORT, public=PRIMARY_PUBLIC, evidence=PRIMARY_EVIDENCE),
            PRIMARY_REPORT,
            "run scripts/generate_recall_union_submission.py",
        )
    report = json.loads(report_path.read_text())
    expected = str((report.get("artifact") or {}).get("sha256") or "")
    public_sha = sha256_file(public)
    evidence_sha = sha256_file(evidence)
    provenance = report.get("provenance") or {}
    no_label_backbone = provenance.get("known_labels_used_to_generate_field") is False
    ok = bool(expected) and public_sha == expected and evidence_sha == expected and no_label_backbone
    return _check(
        "primary_candidate",
        "Recall-Union v1 public and evidence artifacts match their report",
        "PASS" if ok else "FAIL",
        dict(
            report=PRIMARY_REPORT,
            public=PRIMARY_PUBLIC,
            evidence=PRIMARY_EVIDENCE,
            report_sha256=expected,
            public_sha256=public_sha,
            evidence_sha256=evidence_sha,
            bytes=public.stat().st_size,
            known_labels_used_to_generate_field=provenance.get("known_labels_used_to_generate_field"),
            component_filter=provenance.get("component_filter"),
        ),
        PRIMARY_REPORT,
        "the selected candidate is a local, format-validated artifact; official scoring still requires a human upload",
    )


# ------------------------------------------------------------------ 9. candidates
def check_candidates() -> dict:
    """Every committed raster that could be uploaded, with the hash of the bytes on disk."""
    cands = []
    for rel, kind in ((PRIMARY_PUBLIC, "Recall-Union v1 (primary candidate)"),
                      (PRIMARY_EVIDENCE, "Recall-Union v1 (audit copy)"),
                      (f"{SHIPPED_DIR}/submission.tif", "deep ensemble (11-fold blend, adopted policy)"),
                      ("data/evidence/runs/local-sandbox-smoke/submission.tif", "CPU smoke run (pipeline proof)"),
                      ("data/evidence/baseline/submission.tif", "CPU-only classical baseline")):
        p = ROOT / rel
        if p.exists():
            cands.append(dict(path=rel, kind=kind, sha256=sha256_file(p), bytes=p.stat().st_size))
    return _check("candidates", "Artifacts that exist in this checkout", "PASS" if cands else "MISSING",
                  dict(candidates=cands), "repository",
                  "existence is not a quality claim - the measured quality of each field is on the "
                  "results page")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/evidence/submission_readiness.json")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--skip-generator", action="store_true",
                    help="do not re-run the in-browser generator check; quote the last committed verdict")
    ap.add_argument("--print", dest="do_print", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    checks = [check_data_placed(ROOT / a.data_dir),
              check_preflight(ROOT / a.data_dir, a.skip_preflight),
              check_artifact(), check_validation(), check_rules(), check_baseline(),
              check_generator(a.skip_generator),
              check_human(), check_primary_candidate(), check_candidates()]
    failing = [c["id"] for c in checks if c["status"] in ("FAIL",)]
    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/check_submission_readiness.py",
        purpose="every gate on docs/how_to_submit.html is a measurement from this checkout",
        root=str(ROOT), data_dir=a.data_dir, seconds=round(time.time() - t0, 2),
        checks=checks,
        summary=dict(total=len(checks),
                     passed=sum(c["status"] == "PASS" for c in checks),
                     failed=len(failing), human=sum(c["status"] == "HUMAN" for c in checks),
                     missing=sum(c["status"] in ("MISSING", "SKIPPED") for c in checks),
                     failing_ids=failing),
    )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    if a.do_print:
        for c in checks:
            print(f"{c['status']:<8} {c['id']:<18} {c['label']}")
            for k, v in list(c["measured"].items())[:4]:
                print(f"           {k}: {str(v)[:100]}")
        print("summary:", json.dumps(report["summary"]))
    print(f"wrote {out} ({report['summary']})")
    return 2 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
