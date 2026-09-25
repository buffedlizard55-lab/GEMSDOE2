"""Structure of .github/workflows/make-submission.yml — the "no install, no GPU" route's contract.

These assertions exist because a workflow file is documentation that executes. The failure this
repository already paid for (run 35042805806) was a job that went green while its submission writer
had crashed: `| tee` without `pipefail`, and a 110-byte stub committed as the artifact. So the tests
here are not "does the YAML parse" (tests/test_workflow_yaml.py owns that) — they are "would this
job be able to report success while producing nothing submittable".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wf_helpers import executable_lines, pipes   # noqa: E402

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/make-submission.yml"


@pytest.fixture(scope="module")
def doc():
    assert WF.exists(), "the workflow the how-to-submit page tells readers to fire is missing"
    return yaml.safe_load(WF.read_text())


def _steps(doc):
    return doc["jobs"]["build"]["steps"]


def _run(step):
    return step.get("run", "")


def test_it_can_be_fired_by_hand_and_by_a_trigger_file(doc):
    on = doc.get(True) or doc.get("on")     # YAML 1.1 parses bare `on` as True
    assert "workflow_dispatch" in on, "the button on Actions → Run workflow is the point of this file"
    inputs = on["workflow_dispatch"]["inputs"]
    assert set(inputs) >= {"route", "package", "rebuild_payload"}
    assert inputs["route"]["default"] == "adopted", "the default must be the fast, honest route"
    assert "adopted" in inputs["route"]["options"] and "baseline" in inputs["route"]["options"]
    push = on.get("push")
    assert push and ".github/triggers/make-submission" in str(push), \
        "the branch-trigger convention this repository uses for bot-fired runs must apply here too"
    trig = ROOT / ".github/triggers/make-submission"
    assert trig.exists(), \
        ("the workflow's own push trigger path does not exist, so the route can never fire: the "
         "automation token cannot dispatch (`actions: write` is absent), which is exactly what that "
         "file is for")
    body = trig.read_text()
    assert "does not upload anything to DrivenData" in body, "the trigger file must state the limit"
    assert "route=adopted" in body, "a push-run gets the dispatch defaults; the file must say which"


def test_every_piping_step_uses_pipefail(doc):
    """No masked failures. `| tee` alone turned a crashed writer into a green check once already.

    Checked on executable lines only: markdown table bars inside an `echo` and a comment mentioning
    a pipe are not pipes.
    """
    bad = []
    for s in _steps(doc):
        body = _run(s)
        if not any(pipes(l) for l in executable_lines(body)):
            continue
        if not re.search(r"^\s*set\s+-[a-zA-Z]*o\s+pipefail", body, re.M):
            bad.append(s.get("name"))
    assert not bad, f"steps that pipe a real command without pipefail: {bad}"


def test_no_gate_swallows_a_failure(doc):
    """`|| true` may cover cosmetics (disk cleanup, `git add` on a possibly-empty index), never a step
    that decides whether a file is publishable."""
    COSMETIC = ("free disk space", "checkout", "commit the measurement")
    offenders = []
    for s in _steps(doc):
        if (s.get("name") or "").lower().startswith(COSMETIC):
            continue
        for line in executable_lines(_run(s)):
            if "|| true" in line:
                offenders.append((s.get("name"), line[:100]))
    assert not offenders, f"an `|| true` is hiding a gate: {offenders}"


def test_the_pipeline_places_data_before_using_it(doc):
    names = [s.get("name") or "" for s in _steps(doc)]
    def idx(frag):
        return next((i for i, n in enumerate(names) if frag in n), None)
    place, validate, artefact = idx("Place the official rasters"), idx("Refuse to publish"), idx("Upload the artefacts")
    assert place is not None and validate is not None and artefact is not None
    assert place < idx("Route A"), "the bridge has to be assembled before the artifact is validated"
    assert validate < artefact, "the size gate must run before anything is published"


def test_it_needs_no_credential_and_fetches_nothing(doc):
    """The data comes from the sha256-pinned git bridge, which is what makes this route auditable:
    no token, no Dropbox or DrivenData fetch, so a run is reproducible from the repository alone."""
    body = "\n".join(_run(s) for s in _steps(doc))
    assert "no DrivenData login" in WF.read_text(), "the header must say where the data comes from"
    for forbidden in ("secrets.", "GITHUB_TOKEN", "curl ", "wget ", "dropbox.com"):
        assert forbidden not in body, f"{forbidden}: this job must not depend on a fetch or a credential"
    assert (doc.get("permissions") or {}).get("contents") == "write", \
        "committing the evidence JSON is the only thing that needs write"
    assert "assemble_data_bridge.py" in body and "prepare_data.py" in body


def test_the_gate_rejects_a_stub_artifact(doc):
    """The exact regression of run 35042805806: a file that exists but is not a raster."""
    step = next(s for s in _steps(doc) if "Refuse to publish" in (s.get("name") or ""))
    body = _run(step)
    assert "100000" in body or "100 KB" in body, "the stub-size gate is the point of the step"
    assert "test -s" in body, "an empty or absent file must fail the step"


def test_validation_runs_on_every_emitted_file(doc):
    body = "\n".join(_run(s) for s in _steps(doc))
    assert body.count("validate_submission.py") >= 2, "the adopted artifact and the baseline must both be gated"
    assert "package_submission.py" in body, "the .zip the dialog accepts has to be produced and verified"
    assert "check_site_generator.py" in body, ("the page's in-browser generator is advertised by the "
                                               "same site; a run that ignores it can publish a stale payload")


def test_evidence_is_committed_but_bytes_are_not(doc):
    body = "\n".join(_run(s) for s in _steps(doc))
    assert "data/evidence/make_submission" in body
    assert not re.search(r"git add[^\n]*submission\.tif", body), \
        "one 569 KB raster per run is how a repo drowns; the .tif belongs in the artefact"
    upload = next(s for s in _steps(doc) if "Upload the artefacts" in (s.get("name") or ""))
    with_ = upload["with"]
    assert "out/submission.tif" in with_["path"]
    assert with_.get("retention-days", 0) >= 30, "90 days is the window a person has to enrol and upload"


def test_summary_reports_the_hash_a_human_will_check(doc):
    step = next(s for s in _steps(doc) if "Job summary" in (s.get("name") or ""))
    body = _run(step)
    assert "sha256sum" in body and "GITHUB_STEP_SUMMARY" in body
    assert "disclosure" in body.lower(), "the note field is not the §3.2 disclosure; the summary must say so"


def test_the_workflow_has_no_step_that_would_run_a_model_it_cannot_run(doc):
    """Honesty of the header: this file must not pretend to train the leaderboard configuration."""
    body = "\n".join(_run(s) for s in _steps(doc))
    assert "src.train" not in body, "training belongs to train-and-submit.yml; this route's promise is the file"
    install = next(s for s in _steps(doc) if "Install" in (s.get("name") or ""))
    lines = "\n".join(executable_lines(_run(install)))
    assert "torch" not in lines, \
        "installing torch in the route whose claim is that it needs neither would make the claim untrue"
    assert "scikit-learn" in lines, "the baseline classifier needs sklearn"
    assert "rasterio" in lines
    header = WF.read_text().split("on:")[0]
    for phrase in ("does not upload anything to DrivenData", "does not claim a score"):
        assert phrase in header, f"the header must state the limit: {phrase}"


def test_evidence_flags_come_from_exit_statuses_never_from_prose():
    """The first real run (35805002447) was green and still reported `payload_check: false`: the
    evidence builder grepped the word "matches" in a script whose success line says "reproduce". A
    grep of prose is a second source of truth, and it drifts the way this one did. So: record the
    exit status, read the number."""
    text = WF.read_text()
    assert 'in Path("out/' not in text, "an evidence flag is being derived by grepping a log"
    for name in re.findall(r'flag\("out/([\w.]+\.rc)"\)', text):
        assert f"> out/{name}" in text, f"out/{name} is read by the evidence builder but never written"
        assert f'echo "$st" > out/{name}' in text or f'echo "$gs" > out/{name}' in text, \
            f"out/{name} is written outside a recorded-status block"
    assert "test \"$st\" -eq 0" in text, \
        "a recorded status that is never re-raised turns a failed gate into a green run"


def test_a_failing_run_still_reports_itself(doc):
    """The measurement matters most on the run that failed, so reporting must not be skipped with it."""
    for frag in ("Job summary", "Commit the measurement", "Upload the artefacts"):
        step = next(s for s in _steps(doc) if frag in (s.get("name") or ""))
        assert step.get("if") == "always()", f"{frag} is skipped on a failing run"
