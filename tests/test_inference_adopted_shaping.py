"""The plain inference path must ship the MEASURED policy, not the in-domain calibration.

`python -m src.inference` used to shape with `manifest["shaping"]` — the floor calibrated against the
faults the model trained on.  Both prize phases score faults those labels do not contain, and the
decision record measures the two policies on that population: the calibrated floor is worth 0.0247
there, the measured candidate 0.1365.  Nothing in the run said which one it had used, so the defect
was silent.  These tests pin the precedence AND the provenance (source tags + summary fields), which
is what makes the difference auditable from the written raster alone.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import ADOPTED_EVIDENCE_PATH, adopted_shaping, effective_shaping  # noqa: E402

IN_DOMAIN = {"t0": 0.469674, "thin": False, "hard": True, "gamma": 1.0}


def test_the_committed_record_yields_the_measured_policy():
    """Not a synthetic fixture: the policy this repository actually ships comes out of the record."""
    a = adopted_shaping(ROOT / ADOPTED_EVIDENCE_PATH)
    assert a is not None, "the decision record must be readable and end in a SHIP verdict"
    assert (a["t0"], a["thin"], a["dilate"]) == (0.1, True, 0)
    assert a["source"] == "adopted_measured_policy"
    assert a["evidence"].endswith("emission_decision.json")


def test_config_nulls_means_adopt_the_measured_policy_not_the_calibration():
    """The default config path: nulls are an instruction to measure, not to fall back."""
    a = adopted_shaping(ROOT / ADOPTED_EVIDENCE_PATH)
    out = effective_shaping(IN_DOMAIN, {"t0": None, "thin": None, "hard": True, "gamma": 1.0}, a)
    assert out["source"] == "adopted_measured_policy"
    assert (out["t0"], out["thin"], out["dilate"]) == (0.1, True, 0)
    assert out["adopted"]["evidence"].endswith("emission_decision.json")
    # the counterfactual is recorded: without it, "adopted" cannot be told from "calibration agreed"
    assert out["in_domain_control"] == {"t0": 0.469674, "thin": False}
    assert out["t0"] != out["in_domain_control"]["t0"]


def test_an_explicit_config_floor_wins_but_is_never_silent():
    """An experiment may pin a floor; it may not pass for an adopted default."""
    a = adopted_shaping(ROOT / ADOPTED_EVIDENCE_PATH)
    out = effective_shaping(IN_DOMAIN, {"t0": 0.35, "hard": True, "gamma": 1.0}, a)
    assert out["source"] == "explicit_config_override"
    assert out["t0"] == 0.35
    assert out["overrode_adopted"]["t0"] == 0.1


def test_without_a_record_the_old_behaviour_is_kept_and_labelled():
    out = effective_shaping(IN_DOMAIN, {"t0": None}, None)
    assert out["source"] == "in_domain_manifest_calibration"
    assert out["t0"] == 0.469674


def test_a_record_that_does_not_say_ship_is_not_adopted(tmp_path):
    """A record mid-decision ('measured, not shipped') must not silently become the shipped policy."""
    rec = tmp_path / "emission_decision.json"
    rec.write_text(json.dumps({"verdict": {"conclusion": "measured, not shipped: condition 3",
                                           "best_measured_candidate": {"policy":
                                                                       "sweep_best_t0_0.1_width0px"}}}))
    assert adopted_shaping(rec) is None
    rec.write_text("{ not json")
    assert adopted_shaping(rec) is None
    assert adopted_shaping(tmp_path / "absent.json") is None


def test_the_written_submission_carries_its_policy_provenance():
    """The tags blend_submission.py writes must exist on this path too.

    Read from the source rather than from a run: inference needs a checkpoint and the full data
    bridge, and the property being pinned is that the tags are attached at the write, with the
    source and the emission width among them.
    """
    src = (ROOT / "src" / "inference.py").read_text()
    for tag in ("shaping_t0", "shaping_thin", "shaping_dilate", "shaping_source", "shaping_evidence"):
        assert f'"{tag}"' in src, tag
    summary = src[src.index('"shaping_source": shp.get'):]
    for field in ("shaping_source", "shaping_dilate", "shaping_adopted", "shaping_in_domain_control"):
        assert field in summary, field


def test_the_emission_width_reaches_optimize_submission():
    """`dilate` used to be dropped on this path, so a record adopting a non-zero width would have
    been applied as width 0 while the summary claimed the adopted policy.

    Parsed with `ast`, not with string slicing: the call's first argument contains a `)`
    (`R_px_infer(cfg)`), which is exactly the kind of text-level check that silently passes on the
    wrong span.
    """
    import ast

    tree = ast.parse((ROOT / "src" / "inference.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "optimize_submission"]
    assert calls, "no optimize_submission call found"
    kws = set()
    for c in calls:
        kws |= {k.arg for k in c.keywords}
    assert "dilate" in kws, sorted(kws)


def test_the_site_documents_the_adopted_policy_the_code_applies():
    """Docs and code must not disagree about what ships (the failure this repo audits for)."""
    rec = json.loads((ROOT / ADOPTED_EVIDENCE_PATH).read_text())
    policy = rec["verdict"]["best_measured_candidate"]["policy"]
    assert "t0_0.1_width0px" in policy, policy
    adopted = json.loads((ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/blend_report.json")
                         .read_text())["shaping"]
    assert adopted["source"] == "adopted_measured_policy"
    assert (adopted["t0"], adopted["thin"], adopted["dilate"]) == (0.1, True, 0)
    assert adopted["adopted"]["evidence"].endswith("emission_decision.json")


def test_reblend_is_still_the_only_workflow_that_pins_a_shaping(monkeypatch):
    """Sanity check on the other half of the story: the workflow passes the policy explicitly."""
    yml = (ROOT / ".github/workflows/reblend.yml").read_text()
    assert "SHAPING_T0" in yml and "SHAPING_DILATE" in yml
    assert "--shaping-t0" in yml
